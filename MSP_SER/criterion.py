import torch
import torch.nn as nn
import torch.nn.functional as F




class WeightedFocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            alpha = torch.ones(num_classes)
        else:
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[labels]        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
    
    
class FocalSMLoss(nn.Module):
    """
    Learnable convex combination of WeightedFocalLoss + WeightedSoftMarginLoss.
    Both alpha vectors and the blend scalar λ are trainable.
    """
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()

        if alpha is None:
            focal_init = torch.zeros(num_classes)   # softplus(0) ≈ 0.693
            sml_init   = torch.zeros(num_classes)
        else:
            alpha_t    = torch.tensor(alpha, dtype=torch.float32)
            inv        = torch.log(torch.exp(alpha_t) - 1 + 1e-6)  # inverse softplus
            focal_init = inv.clone()
            sml_init   = inv.clone()

        self.log_alpha_focal = nn.Parameter(focal_init)   # learnable α for focal
        self.log_alpha_sml   = nn.Parameter(sml_init)     # learnable α for SML
        self.raw_lambda      = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.gamma           = gamma
        self.num_classes     = num_classes
        self.reduction       = reduction

    def _focal(self, logits, labels, alpha):
        ce_loss    = F.cross_entropy(logits, labels, reduction='none')   # [B]
        pt         = torch.exp(-ce_loss)
        alpha_t    = alpha[labels]                                        # [B]
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

    def _sml(self, logits, labels, alpha):
        targets = torch.full_like(logits, -1.0)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)
        per_element      = torch.log(1.0 + torch.exp(-targets * logits))  # [B, C]
        weighted         = alpha.unsqueeze(0).expand_as(per_element) * per_element
        return weighted.mean() if self.reduction == 'mean' else weighted.sum()

    def forward(self, logits, labels):
        alpha_focal = F.softplus(self.log_alpha_focal)   # [C], always positive
        alpha_sml   = F.softplus(self.log_alpha_sml)     # [C], always positive
        lam         = torch.sigmoid(self.raw_lambda)     # scalar ∈ (0, 1)

        focal_loss  = self._focal(logits, labels, alpha_focal)
        sml_loss    = self._sml(logits, labels, alpha_sml)

        return lam * focal_loss + (1.0 - lam) * sml_loss
    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        # ce_loss = torch.clamp(ce_loss, min=1e-7, max=100.0)  # ← ADD THIS LINE ONLY
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RKDLoss(nn.Module):
    """Relational Knowledge Distillation: normalized pairwise L2 distance + Huber."""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    @staticmethod
    def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
        dist      = torch.cdist(x, x, p=2)
        mask      = ~torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
        mean_dist = dist.masked_select(mask).mean().clamp_min(1e-8)
        return dist / mean_dist

    def forward(self, z_S: torch.Tensor, z_T: torch.Tensor) -> torch.Tensor:
        z_S  = F.normalize(z_S, p=2, dim=1)
        z_T  = F.normalize(z_T, p=2, dim=1)
        d_S  = self._pairwise_dist(z_S)
        d_T  = self._pairwise_dist(z_T)
        mask = torch.triu(torch.ones_like(d_S, dtype=torch.bool), diagonal=1)
        return F.huber_loss(d_S[mask], d_T[mask], delta=self.delta, reduction='mean')


class CosineAlignmentLoss(nn.Module):
    """Embedding-level cosine alignment with learnable projection W_p."""
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.W_p = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, e_S: torch.Tensor, e_T: torch.Tensor) -> torch.Tensor:
        e_S_proj = F.normalize(self.W_p(e_S), p=2, dim=1)
        e_T      = F.normalize(e_T.detach(), p=2, dim=1)
        return (1 - (e_S_proj * e_T).sum(dim=1)).mean()

        
class SoftKLLoss(nn.Module):
    """
    Stable KL divergence for soft targets.
    logits : (B, C)  raw model output
    targets: (B, C)  per-annotator vote probabilities, sum to 1
    """
    def forward(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        p     = targets.clamp(min=1e-8)          # avoids log(0) → -inf
        return F.kl_div(log_q, p, reduction="batchmean")
    
    
# class WeightedFocalPrototypeLoss(nn.Module):
#     """
#     Combined Weighted Focal Loss + Prototype (Center) Loss.

#     Weighted Focal Loss:
#         L_wfl = α_t · (1 - p_t)^γ · CE(logits, labels)
#         Handles class imbalance via per-class alpha weights and
#         down-weights easy examples via the focal modulation term.

#     Prototype Loss:
#         L_proto = (1/N) Σ_i ||z_i - c_{y_i}||²
#         Pulls each embedding toward its class prototype (running centroid).
#         Prototypes are updated online via exponential moving average (EMA).

#     Total Loss:
#         L = L_wfl + λ · L_proto

#     Args:
#         num_classes  (int)   : Number of emotion classes (8).
#         embed_dim    (int)   : Dimensionality of embeddings fed into the loss.
#         alpha        (list)  : Per-class weights (inv_freq or cb_weight vector).
#         gamma        (float) : Focal modulation exponent. Default: 2.0.
#         proto_lambda (float) : Weight for prototype loss term. Default: 0.1.
#         proto_momentum (float): EMA momentum for prototype update. Default: 0.9.
#         reduction    (str)   : 'mean' or 'sum' for focal loss. Default: 'mean'.
#     """

#     def __init__(self, num_classes: int, embed_dim: int, alpha=None, gamma: float = 2.0, proto_lambda: float = 0.1,
#                  proto_momentum: float = 0.9, reduction: str = "mean",):
#         super().__init__()
        
#         if alpha is None:
#             alpha = torch.ones(num_classes)
#         else:
#             alpha = torch.tensor(alpha, dtype=torch.float32)
#         self.register_buffer("alpha", alpha)

#         self.gamma = gamma
#         self.reduction = reduction

#         self.num_classes = num_classes
#         self.embed_dim = embed_dim
#         self.proto_lambda = proto_lambda
#         self.proto_momentum = proto_momentum

#         self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
#         self.register_buffer("proto_initialized", torch.zeros(num_classes, dtype=torch.bool))

#     @torch.no_grad()
#     def _update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
#         """
#         For each class present in the batch, compute the mean embedding
#         and update the stored prototype via EMA:
#             c_k ← m · c_k + (1 - m) · batch_mean_k
#         First-time init: set prototype directly to the batch mean.
#         """
#         for cls in labels.unique():
#             mask = labels == cls
#             batch_mean = embeddings[mask].mean(dim=0)  # (embed_dim,)

#             if self.proto_initialized[cls]:
#                 self.prototypes[cls] = (self.proto_momentum * self.prototypes[cls] + (1.0 - self.proto_momentum) * batch_mean)
#             else:
#                 self.prototypes[cls] = batch_mean
#                 self.proto_initialized[cls] = True

#     def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         ce_loss = F.cross_entropy(logits, labels, reduction="none")  # (N,)
#         pt = torch.exp(-ce_loss)                                      # (N,)
#         alpha_t = self.alpha[labels]                                  # (N,)
#         focal = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss       # (N,)

#         return focal.mean() if self.reduction == "mean" else focal.sum()

#     def _prototype_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         self._update_prototypes(embeddings.detach(), labels)
#         assigned = self.prototypes[labels]              # (N, embed_dim)
#         diff = embeddings - assigned                    # (N, embed_dim)
#         dist_sq = (diff ** 2).sum(dim=1)                # (N,)
#         return dist_sq.mean()

#     def forward(self, logits: torch.Tensor, embeddings: torch.Tensor, labels: torch.Tensor,) -> tuple[torch.Tensor, dict]:
#         l_focal = self._focal_loss(logits, labels)
#         l_proto = self._prototype_loss(embeddings, labels)
#         total = l_focal + self.proto_lambda * l_proto
#         info = {
#             "loss_total":    total.item(),
#             "loss_focal":    l_focal.item(),
#             "loss_proto":    l_proto.item(),
#         }
#         return total, info


class WeightedFocalPrototypeLoss(nn.Module):
    """
    Combined Weighted Focal Loss + Prototype (Center) Loss.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        alpha=None,
        gamma: float = 2.0,
        proto_lambda: float = 1e-4,
        reduction: str = "mean",
        warmup_epochs: int = 2,
    ):
        super().__init__()

        if alpha is None:
            alpha = torch.ones(num_classes)
        else:
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.proto_lambda = proto_lambda
        self.warmup_epochs = warmup_epochs

        prototypes = F.normalize(torch.randn(num_classes, embed_dim), dim=-1)
        self.prototypes = nn.Parameter(prototypes)

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[labels]
        focal = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal.mean() if self.reduction == "mean" else focal.sum()

    def _prototype_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(embeddings, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        assigned = p_norm[labels]
        dist_sq = ((z_norm - assigned) ** 2).sum(dim=-1)
        return dist_sq.mean()

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        l_focal = self._focal_loss(logits, labels)

        if epoch < self.warmup_epochs:
            l_proto = torch.tensor(0.0, device=logits.device)
            proto_active = False
        else:
            l_proto = self._prototype_loss(embeddings, labels)
            proto_active = True

        total = l_focal + self.proto_lambda * l_proto

        info = {
            "loss_total": total.item(),
            "loss_focal": l_focal.item(),
            "loss_proto": l_proto.item(),
            "proto_active": proto_active,
        }
        return total, info
    
    
class WeightedSoftMarginLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, reduction='mean'):
        super().__init__()
        if alpha is None:
            alpha = torch.ones(num_classes)
        else:
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha)
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, labels):
        # Convert integer labels → one-hot {+1, -1} targets
        targets = torch.full_like(logits, -1.0)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)

        # Soft margin loss: log(1 + exp(-y * x)) per element
        per_element = torch.log(1.0 + torch.exp(-targets * logits))  # [B, C]

        # Apply per-class alpha weighting
        alpha_expanded = self.alpha.unsqueeze(0).expand_as(per_element)  # [B, C]
        weighted = alpha_expanded * per_element

        if self.reduction == 'mean':
            return weighted.mean()
        return weighted.sum()
    
    
class WeightedSoftMarginLossTrainable(nn.Module):
    def __init__(self, num_classes, alpha=None, reduction='mean'):
        super().__init__()
        if alpha is None:
            # Initialize raw logits to zero → softplus(0) ≈ 0.693 ≈ uniform-ish
            init = torch.zeros(num_classes)
        else:
            # Back-calculate raw logits from desired initial alpha values
            alpha_t = torch.tensor(alpha, dtype=torch.float32)
            init = torch.log(torch.exp(alpha_t) - 1 + 1e-6)  # inverse softplus

        # ← KEY CHANGE: nn.Parameter instead of register_buffer
        self.log_alpha = nn.Parameter(init)
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, labels):
        # Ensure alpha is always positive via softplus
        alpha = F.softplus(self.log_alpha)  # [C], all positive

        # One-hot {+1, -1} targets
        targets = torch.full_like(logits, -1.0)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)

        # Soft margin per element: log(1 + exp(-y * x))
        per_element = torch.log(1.0 + torch.exp(-targets * logits))  # [B, C]

        # Apply learnable per-class weighting
        weighted = alpha.unsqueeze(0).expand_as(per_element) * per_element

        return weighted.mean() if self.reduction == 'mean' else weighted.sum()