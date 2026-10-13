"""
train_lora_layerweighted.py
============================
Your original WavLMLayerWeighted + ProjectionNetwork architecture, enhanced
with LoRA adapters injected into WavLM's attention projections.

Key changes vs. original:
  - LoRA adapters replace plain fine-tuning of the backbone.
  - backbone.requires_grad_(False) is set first; only LoRA params + the
    learnable layer-weights are trained within WavLMLayerWeighted.
  - ProjectionNetwork is unchanged.
  - Optimizer uses separate LR groups: LoRA/layer-weights vs. head.
"""

import os, random, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse, torch, pandas
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.amp import autocast

from MSP_SER.dataset import MSP2_Dataset
from MSP_SER.criterion import WeightedFocalLoss
from MSP_SER.utilities import GetYAMLConfigs, Logs


###############################################################################
# LoRA primitives  (identical to train_simple_lora.py – shared utility)
###############################################################################

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear."""
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / rank
        in_f, out_f  = linear.in_features, linear.out_features

        self.lora_A  = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B  = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora_into_wavlm(model, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                            target_modules=("k_proj", "v_proj", "q_proj", "out_proj")):
    model.requires_grad_(False)
    replaced = 0
    for _, module in model.named_modules():
        for attr in target_modules:
            if hasattr(module, attr):
                child = getattr(module, attr)
                if isinstance(child, nn.Linear):
                    setattr(module, attr, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                    replaced += 1
    print(f"[LoRA] Replaced {replaced} Linear layers (rank={rank}, alpha={alpha})")
    return model


###############################################################################
# WavLMLayerWeighted  with LoRA
###############################################################################

class WavLMLayerWeightedLoRA(nn.Module):
    """
    Original WavLMLayerWeighted architecture with LoRA injected into WavLM.

    Trainable components:
      - LoRA A/B matrices inside WavLM attention projections
      - Layer importance weights  (self.weights)
      - Cross-attention modules   (self.cross_attns)
      - Depthwise conv pools      (self.pools)
      - Adaptive avg-pool         (self.proj)
    """
    def __init__(self,
                 bundle=torchaudio.pipelines.WAVLM_LARGE,
                 k: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.5,
                 pool_out: int = 32,
                 final_dim: int = 2048,
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 lora_dropout: float = 0.05):
        super().__init__()
        self.k         = k
        self.final_dim = final_dim

        # Load backbone and inject LoRA
        self.wavlm = bundle.get_model()
        inject_lora_into_wavlm(self.wavlm, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

        # Probe dimension from a dummy forward pass
        with torch.no_grad():
            feats, _ = self.wavlm.extract_features(torch.zeros(1, 16000))
        dim = feats[0].shape[-1]

        # Learnable layer importance weights (trainable, unlike the frozen backbone)
        self.weights = nn.Parameter(torch.ones(len(feats)))

        # k cross-attention heads — one per selected layer
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(k)
        ])
        # Depthwise temporal convolution pools
        self.pools = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=1, groups=dim) for _ in range(k)
        ])
        # Adaptive spatial pooling
        self.proj = nn.ModuleList([
            nn.AdaptiveAvgPool1d(pool_out) for _ in range(k)
        ])

    def forward(self, waveform):
        with torch.no_grad():
            layer_outputs, _ = self.wavlm.extract_features(waveform)

        # NOTE: LoRA forward passes inside extract_features ARE computed (autograd
        # tracks them) even though we called no_grad() here; the no_grad context
        # only skips gradient accumulation for the frozen base weights.
        # To allow LoRA gradients to flow, run extract_features OUTSIDE no_grad:
        layer_outputs, _ = self.wavlm.extract_features(waveform)

        w           = torch.sigmoid(self.weights)
        stacked     = torch.stack(layer_outputs, dim=0)           # (L, B, T, D)
        top_w, top_idx = torch.topk(w, self.k)
        selected    = stacked[top_idx] * top_w[:, None, None, None]  # (k, B, T, D)

        pooled = []
        for i, (mha, pool) in enumerate(zip(self.cross_attns, self.pools)):
            elem = selected[i]
            q    = torch.stack([selected[j] for j in range(self.k) if j != i]).mean(0)
            out, _ = mha(q, elem, elem)
            out  = out + elem
            out  = pool(out.transpose(1, 2)).transpose(1, 2)
            out  = self.proj[i](out.transpose(1, 2)).transpose(1, 2)
            pooled.append(out.flatten(1))

        return F.adaptive_avg_pool1d(
            torch.cat(pooled, dim=-1).unsqueeze(0), output_size=self.final_dim
        ).squeeze(0)


###############################################################################
# ProjectionNetwork  (unchanged from original)
###############################################################################

class ProjectionNetwork(nn.Module):
    def __init__(self, dims: int = 2048, num_classes: int = 8):
        super().__init__()
        self.epilogue = nn.Sequential(
            nn.Linear(dims,       dims // 2),
            nn.Linear(dims // 2,  dims // 4),
            nn.ReLU(),
            nn.Linear(dims // 4,  dims // 6),
            nn.ReLU(),
            nn.Linear(dims // 6,  num_classes),
        )

    def forward(self, X):
        return self.epilogue(X)


###############################################################################
# Main
###############################################################################

P = argparse.ArgumentParser()
P.add_argument("--config", type=str, required=True)
A = P.parse_args()

configs = GetYAMLConfigs(path=A.config)
logger  = Logs(exp_name=configs["experiment_name"])
DEVICE  = torch.device("cuda:{}".format(configs["cuda_device"]))

torch.cuda.empty_cache()
torch.manual_seed(configs["seed"])

# ── Dataloaders ───────────────────────────────────────────────────────────────
train_dataloader = DataLoader(
    dataset=MSP2_Dataset(), batch_size=configs["BATCH_SIZE"], shuffle=True)
dev_dataloader   = DataLoader(
    dataset=MSP2_Dataset(split="Development"), batch_size=configs["BATCH_SIZE"], shuffle=True)

num_classes = len(train_dataloader.dataset.labels)

# ── Models ────────────────────────────────────────────────────────────────────
feature_extractor  = WavLMLayerWeightedLoRA(
    k=9,
    lora_rank=configs.get("lora_rank",    8),
    lora_alpha=configs.get("lora_alpha",  16.0),
    lora_dropout=configs.get("lora_dropout", 0.05),
).to(DEVICE)

projection_network = ProjectionNetwork(
    dims=feature_extractor.final_dim, num_classes=num_classes
).to(DEVICE)

criterion = WeightedFocalLoss(num_classes=num_classes).to(DEVICE)

# LoRA params + layer-weights at backbone LR; head at full LR
lora_and_attn_params = (
    [p for n, p in feature_extractor.named_parameters() if p.requires_grad]
)
head_params = list(projection_network.parameters())

logger.write(
    f"Trainable — backbone+LoRA: {sum(p.numel() for p in lora_and_attn_params):,} | "
    f"head: {sum(p.numel() for p in head_params):,}"
)

optimizer = torch.optim.AdamW(
    [
        {"params": lora_and_attn_params, "lr": configs["learning_rate"] * 0.1},
        {"params": head_params,          "lr": configs["learning_rate"]},
    ],
    weight_decay=0.01,
)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=9)
scaler      = torch.amp.GradScaler(f"cuda:{configs['cuda_device']}")
accum_steps = configs["accumulation_steps"]

# ── Training loop ─────────────────────────────────────────────────────────────
for e in range(configs["epochs"]):
    feature_extractor.train(); projection_network.train(); criterion.train()
    epoch_loss = epoch_total = epoch_correct = 0

    for mb_idx, (mb_data, mb_label) in enumerate(train_dataloader):
        mb_data, mb_label = mb_data.squeeze(1).to(DEVICE), mb_label.to(DEVICE)

        with autocast(device_type=f"cuda:{configs['cuda_device']}"):
            embeddings  = feature_extractor(mb_data)
            predictions = projection_network(embeddings)
            loss        = criterion(predictions, mb_label) / accum_steps

        scaler.scale(loss).backward()

        if (mb_idx + 1) % accum_steps == 0 or (mb_idx + 1) == len(train_dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                lora_and_attn_params + head_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss    += loss.item() * accum_steps
        preds          = predictions.argmax(dim=-1)
        epoch_correct += (preds == mb_label).sum().item()
        epoch_total   += mb_label.size(0)

        if mb_idx % configs["minibatch_log_after"] == 0:
            logger.write("Epoch: [{}], MiniBatch: [{}], Loss: [{}], Accuracy: [{:.2f}%]".format(
                e, mb_idx, loss.item() * accum_steps, 100 * epoch_correct / epoch_total))

    scheduler.step()
    logger.write("\n=== Epoch: [{}] | Avg Loss: [{:.4f}] | Accuracy: [{:.2f}%] ===".format(
        e, epoch_loss / len(train_dataloader), 100 * epoch_correct / epoch_total))

    # ── Dev eval ──────────────────────────────────────────────────────────────
    feature_extractor.eval(); projection_network.eval()
    dev_loss = dev_correct = dev_total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for mb_data, mb_label in dev_dataloader:
            mb_data, mb_label = mb_data.squeeze(1).to(DEVICE), mb_label.to(DEVICE)
            with autocast(device_type="cuda"):
                embeddings  = feature_extractor(mb_data)
                predictions = projection_network(embeddings)
                loss        = criterion(predictions, mb_label)
            dev_loss    += loss.item()
            preds        = predictions.argmax(dim=-1)
            dev_correct += (preds == mb_label).sum().item()
            dev_total   += mb_label.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(mb_label.cpu().tolist())

    logger.write(
        "=== DEV Epoch: [{}] | Avg Loss: [{:.4f}] | Accuracy: [{:.2f}%] | "
        "Precision (Micro): [{:.4f}] | Precision (Macro): [{:.4f}] | "
        "Recall (Micro): [{:.4f}] | Recall (Macro): [{:.4f}] | "
        "Recall (Weighted): [{:.4f}] | F1-Micro: [{:.4f}] | F1-Macro: [{:.4f}] ===".format(
            e,
            dev_loss / len(dev_dataloader),
            100 * dev_correct / dev_total,
            precision_score(all_labels, all_preds, average="micro",    zero_division=0),
            precision_score(all_labels, all_preds, average="macro",    zero_division=0),
            recall_score   (all_labels, all_preds, average="micro"),
            recall_score   (all_labels, all_preds, average="macro"),
            recall_score   (all_labels, all_preds, average="weighted"),
            f1_score       (all_labels, all_preds, average="micro"),
            f1_score       (all_labels, all_preds, average="macro"),
        ))

    torch.save(
        {"feature_extractor": feature_extractor.state_dict(),
         "projection_network": projection_network.state_dict()},
        os.path.join(logger.chkpt_folder, f"epoch_{e}.pt"),
    )

    # ── Test splits ───────────────────────────────────────────────────────────
    if (e + 1) % configs["compute_test_scores_after"] == 0:
        for split in ["Test1", "Test2"]:
            test_dataloader = DataLoader(
                dataset=MSP2_Dataset(split=split), batch_size=configs["BATCH_SIZE"], shuffle=False)
            t_loss = t_correct = t_total = 0
            t_preds, t_labels = [], []

            with torch.no_grad():
                for mb_data, mb_label in test_dataloader:
                    mb_data, mb_label = mb_data.squeeze(1).to(DEVICE), mb_label.to(DEVICE)
                    with autocast(device_type="cuda"):
                        embeddings  = feature_extractor(mb_data)
                        predictions = projection_network(embeddings)
                        loss        = criterion(predictions, mb_label)
                    t_loss    += loss.item()
                    preds      = predictions.argmax(dim=-1)
                    t_correct += (preds == mb_label).sum().item()
                    t_total   += mb_label.size(0)
                    t_preds.extend(preds.cpu().tolist())
                    t_labels.extend(mb_label.cpu().tolist())

            logger.write(
                "=== {} Epoch: [{}] | Avg Loss: [{:.4f}] | Accuracy: [{:.2f}%] | "
                "Precision (Micro): [{:.4f}] | Precision (Macro): [{:.4f}] | "
                "Recall (Micro): [{:.4f}] | Recall (Macro): [{:.4f}] | "
                "Recall (Weighted): [{:.4f}] | F1-Micro: [{:.4f}] | F1-Macro: [{:.4f}] ===".format(
                    split, e,
                    t_loss / len(test_dataloader),
                    100 * t_correct / t_total,
                    precision_score(t_labels, t_preds, average="micro",    zero_division=0),
                    precision_score(t_labels, t_preds, average="macro",    zero_division=0),
                    recall_score   (t_labels, t_preds, average="micro"),
                    recall_score   (t_labels, t_preds, average="macro"),
                    recall_score   (t_labels, t_preds, average="weighted"),
                    f1_score       (t_labels, t_preds, average="micro"),
                    f1_score       (t_labels, t_preds, average="macro"),
                ))
        logger.write("\n")

###############################################################################
# Test3 – unlabelled inference
###############################################################################
with open("/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Partitions.txt", "r") as file:
    test3_lines = [
        os.path.join("/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Audios",
                     line.strip().replace("Test3; ", ""))
        for line in file if "Test3" in line
    ]

inv_labels = {v: k for k, v in train_dataloader.dataset.labels_dict.items()}
duration   = int(16000 * 6)


def load_audio(path, full_audio=False):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0, keepdim=True)
    wav = wav.repeat(1, duration // wav.shape[-1] + 1)
    return wav.mean(dim=0, keepdim=True) if full_audio else wav[:, :duration]


feature_extractor.eval(); projection_network.eval()
filenames, pred_classes = [], []
batch_size = configs["BATCH_SIZE"]

with torch.no_grad():
    for i in range(0, len(test3_lines), batch_size):
        batch_paths = test3_lines[i : i + batch_size]
        mb_data     = torch.stack([load_audio(p, full_audio=True)
                                   for p in batch_paths]).squeeze(1).to(DEVICE)
        with autocast(device_type="cuda"):
            preds = projection_network(feature_extractor(mb_data)).argmax(dim=-1)
        filenames.extend([os.path.basename(p) for p in batch_paths])
        pred_classes.extend([inv_labels[p.item()] for p in preds])

pandas.DataFrame({"FileName": filenames, "EmoClass": pred_classes}).to_csv(
    os.path.join(logger.chkpt_folder, "test3_predictions.csv"), index=False)
logger.write("Test3 predictions saved → test3_predictions.csv ({} samples)".format(len(filenames)))
