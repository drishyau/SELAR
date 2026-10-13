import torch, copy
import torch.nn as nn
import random, torchaudio, math
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq



"""
Crab style Wavlm layer selection with cross attention on selected layers
"""
class WavLMLayerWeighted(nn.Module):
    def __init__(self, bundle=torchaudio.pipelines.WAVLM_LARGE, k=3, num_heads=8, dropout=0.5, pool_out=32, final_dim=2048, asp=True):
        super().__init__()
        self.k = k
        self.final_dim = final_dim

        self.wavlm = bundle.get_model()
        self.wavlm.requires_grad_(False)
        
        with torch.no_grad():
            feats, _ = self.wavlm.extract_features(torch.zeros(1, 16000))
        dim = feats[0].shape[-1]
        self.weights = nn.Parameter(torch.ones(len(feats)))
        self.cross_attns = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True) for _ in range(k)])

        self.pools = nn.ModuleList([nn.Conv1d(dim, dim, kernel_size=1, groups=dim) for _ in range(k)])
        self.proj = nn.ModuleList([nn.AdaptiveAvgPool1d(pool_out) for _ in range(k)])

    def forward(self, waveform):
        with torch.no_grad():
            layer_outputs, _ = self.wavlm.extract_features(waveform)
        w = torch.sigmoid(self.weights)
        stacked = torch.stack(layer_outputs, dim=0)
        top_w, top_idx = torch.topk(w, self.k)
        selected = stacked[top_idx] * top_w[:, None, None, None]
        pooled = []
        for i, (mha, pool) in enumerate(zip(self.cross_attns, self.pools)):
            elem = selected[i]
            q = torch.stack([selected[j] for j in range(self.k) if j != i]).mean(0)
            out, _ = mha(q, elem, elem)
            out = out + elem
            out = pool(out.transpose(1, 2)).transpose(1, 2)
            out = self.proj[i](out.transpose(1, 2)).transpose(1, 2)
            pooled.append(out.flatten(1))
        return F.adaptive_avg_pool1d(torch.cat(pooled, dim=-1), output_size=self.final_dim)


class WhisperLayerWeighted(nn.Module):
    def __init__(self, model_name="openai/whisper-large-v3", k=3, num_heads=8,dropout=0.5, pool_out=32, final_dim=2048, asp=True, sr=16000,):
        super().__init__()
        self.k = k
        self.final_dim = final_dim
        self.sr = sr

        # ── Load Whisper processor + encoder ──────────────────────────────
        self.processor = AutoProcessor.from_pretrained(model_name)
        whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.encoder = whisper.model.encoder   # only the encoder half
        self.encoder.requires_grad_(False)

        # ── Probe hidden-state shape ───────────────────────────────────────
        with torch.no_grad():
            dummy_feats = self._wav_to_features(torch.zeros(1, sr))
            enc_out = self.encoder(
                input_features=dummy_feats, output_hidden_states=True
            )
        # enc_out.hidden_states is a tuple: (embedding_layer, layer1, ..., layer32)
        num_layers = len(enc_out.hidden_states)   # 33 for whisper-large-v3
        dim = enc_out.hidden_states[0].shape[-1]  # 1280 for whisper-large-v3

        self.weights = nn.Parameter(torch.ones(num_layers))
        self.cross_attns = nn.ModuleList(
            [nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
             for _ in range(k)]
        )
        self.pools = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=1, groups=dim) for _ in range(k)]
        )
        self.proj = nn.ModuleList(
            [nn.AdaptiveAvgPool1d(pool_out) for _ in range(k)]
        )

    # ── helper: raw waveform (B, T) → log-mel input_features (B, 128, 3000) ──
    def _wav_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        # processor expects numpy / list; move to CPU for processing
        wav_np = waveform.squeeze(0).cpu().numpy()   # (T,) for single sample
        inputs = self.processor(
            wav_np, sampling_rate=self.sr, return_tensors="pt"
        )
        return inputs["input_features"].to(device)   # (1, 128, 3000)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)  — process one sample at a time for the processor
        # For batch support we loop; swap to batched processor call if preferred.
        feats_list = [self._wav_to_features(waveform[i:i+1]) for i in range(waveform.shape[0])]
        input_features = torch.cat(feats_list, dim=0)   # (B, 128, 3000)

        with torch.no_grad():
            enc_out = self.encoder(
                input_features=input_features, output_hidden_states=True
            )
        # hidden_states: tuple of (B, T', 1280), length = num_layers
        layer_outputs = list(enc_out.hidden_states)

        w = torch.sigmoid(self.weights)
        stacked = torch.stack(layer_outputs, dim=0)        # (num_layers, B, T', dim)
        top_w, top_idx = torch.topk(w, self.k)
        selected = stacked[top_idx] * top_w[:, None, None, None]  # (k, B, T', dim)

        pooled = []
        for i, (mha, pool) in enumerate(zip(self.cross_attns, self.pools)):
            elem = selected[i]                             # (B, T', dim)
            q = torch.stack(
                [selected[j] for j in range(self.k) if j != i]
            ).mean(0)                                      # (B, T', dim)
            out, _ = mha(q, elem, elem)
            out = out + elem
            out = pool(out.transpose(1, 2)).transpose(1, 2)
            out = self.proj[i](out.transpose(1, 2)).transpose(1, 2)
            pooled.append(out.flatten(1))

        return F.adaptive_avg_pool1d(
            torch.cat(pooled, dim=-1), output_size=self.final_dim
        )
        
class WavLMLayerWeightedNew(nn.Module):
    def __init__(self, bundle=torchaudio.pipelines.WAVLM_LARGE, k=3,num_heads=8, dropout=0.5, pool_out=32, final_dim=2048, asp=True, weighted=True,):
        super().__init__()
        self.k = k
        self.final_dim = final_dim
        self.weighted = weighted

        self.wavlm = bundle.get_model()
        self.wavlm.requires_grad_(False)

        with torch.no_grad():
            feats, _ = self.wavlm.extract_features(torch.zeros(1, 16000))
        dim = feats[0].shape[-1]

        self.weights = nn.Parameter(torch.ones(len(feats)))
        self.cross_attns = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(k)
        ])

        self.pools = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=1, groups=dim)
            for _ in range(k)
        ])
        self.proj = nn.ModuleList([
            nn.AdaptiveAvgPool1d(pool_out)
            for _ in range(k)
        ])

    def _rank_weighted_mean(self, selected, exclude_idx):
        device = selected.device
        other_indices = [j for j in range(self.k) if j != exclude_idx]

        if len(other_indices) == 1:
            return selected[other_indices[0]]

        other_layers = selected[other_indices]  # [k-1, B, T, D]

        if self.weighted:
            # selected is already ordered by top-k rank because topk(sorted=True)
            # higher rank gets higher weight: [k-1, k-2, ..., 1]
            rank_weights = torch.arange(len(other_indices), 0, -1,dtype=other_layers.dtype, device=device)
            rank_weights = rank_weights / rank_weights.sum()
            q = (other_layers * rank_weights.view(-1, 1, 1, 1)).sum(dim=0)
        else:
            q = other_layers.mean(dim=0)

        print(q)
        return q

    def forward(self, waveform):
        with torch.no_grad():
            layer_outputs, _ = self.wavlm.extract_features(waveform)

        w = torch.sigmoid(self.weights)
        stacked = torch.stack(layer_outputs, dim=0)  # [L, B, T, D]

        top_w, top_idx = torch.topk(w, self.k, sorted=True)
        selected = stacked[top_idx] * top_w[:, None, None, None]  # [k, B, T, D]

        pooled = []
        for i, (mha, pool) in enumerate(zip(self.cross_attns, self.pools)):
            elem = selected[i]
            q = self._rank_weighted_mean(selected, exclude_idx=i)

            out, _ = mha(q, elem, elem)
            out = out + elem
            out = pool(out.transpose(1, 2)).transpose(1, 2)
            out = self.proj[i](out.transpose(1, 2)).transpose(1, 2)
            pooled.append(out.flatten(1))

        return F.adaptive_avg_pool1d(
            torch.cat(pooled, dim=-1), output_size=self.final_dim
        )
        
class WavLMLayerWeightedDualStream(nn.Module):
    def __init__(
        self,
        bundle=torchaudio.pipelines.WAVLM_LARGE,
        k=3,
        num_heads=8,
        dropout=0.5,
        pool_out=32,
        final_dim=2048,
    ):
        super().__init__()
        self.k         = k
        self.final_dim = final_dim

        # ── WavLM backbone (frozen) ──────────────────────────────────────────
        self.wavlm = bundle.get_model()
        self.wavlm.requires_grad_(False)

        with torch.no_grad():
            feats, _ = self.wavlm.extract_features(torch.zeros(1, 16000))

        total_layers = len(feats)            # 24
        self.split   = total_layers // 2     # 12
        dim          = feats[0].shape[-1]    # 1024

        # ── Per-stream layer weights ─────────────────────────────────────────
        self.weights_acoustic = nn.Parameter(torch.ones(self.split))
        self.weights_semantic = nn.Parameter(torch.ones(total_layers - self.split))

        # ── Per-slot, per-stream LayerNorms (NOT shared across slots) ────────
        # FIX: separate norm per slot prevents running-stat corruption under AMP
        self.norms_acoustic = nn.ModuleList([nn.LayerNorm(dim) for _ in range(k)])
        self.norms_semantic  = nn.ModuleList([nn.LayerNorm(dim) for _ in range(k)])

        # ── Cross-attention stacks ───────────────────────────────────────────
        self.cross_attns_acoustic = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(k)
        ])
        self.cross_attns_semantic = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(k)
        ])

        # ── Depthwise conv (per slot per stream) ─────────────────────────────
        self.pools_acoustic = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=1, groups=dim) for _ in range(k)
        ])
        self.pools_semantic = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=1, groups=dim) for _ in range(k)
        ])

        # ── Temporal compression → pool_out frames ───────────────────────────
        self.proj_acoustic = nn.ModuleList([nn.AdaptiveAvgPool1d(pool_out) for _ in range(k)])
        self.proj_semantic  = nn.ModuleList([nn.AdaptiveAvgPool1d(pool_out) for _ in range(k)])

        # stream_dim: what each stream outputs before ASP
        # shape: (B, dim, k*pool_out)  →  ASP  →  (B, 2*dim)
        asp_in_dim  = k * pool_out      # temporal frames fed into ASP (32*k)
        asp_out_dim = dim * 2           # ASP outputs mean+std → 2*dim = 2048

        # ── Attentive Stats Pooling (one per stream, actually used) ──────────
        # Input shape to ASP: (B, dim, k*pool_out)
        self.asp_acoustic = AttentiveStatsPool(in_dim=dim)
        self.asp_semantic  = AttentiveStatsPool(in_dim=dim)

        # After ASP each stream is (B, 2*dim) = (B, 2048)
        # Fused = (B, 4*dim) = (B, 4096)
        fused_dim = asp_out_dim * 2     # 4096

        # ── Fusion gate ──────────────────────────────────────────────────────
        # Bottleneck sized relative to fused_dim — no extreme compression
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )

        # ── Final projection ─────────────────────────────────────────────────
        # Input: gated combination of two ASP outputs → (B, asp_out_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(asp_out_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.GELU()
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _select_layers(self, layer_list, weight_param):
        w              = F.softmax(weight_param, dim=0)               # (N,)
        stacked        = torch.stack(layer_list, dim=0)               # (N, B, T, D)
        top_w, top_idx = torch.topk(w, self.k)
        return stacked[top_idx] * top_w[:, None, None, None]          # (k, B, T, D)

    # ─────────────────────────────────────────────────────────────────────────
    def _process_stream(self, selected, cross_attns, pools, projs, norms):
        """
        selected: (k, B, T, D)
        Each slot i:
          - norm_i(elem_i)  as key/value
          - norm_i(mean of others) as query
          - cross-attention + pre-norm residual
          - depthwise conv + adaptive pool → (B, pool_out, D)
        Returns: (B, D, k*pool_out)  ready for AttentiveStatsPool
        """
        pooled = []
        for i, (mha, pool, proj, norm) in enumerate(zip(cross_attns, pools, projs, norms)):
            elem    = norm(selected[i])                                # (B, T, D)
            others  = [norm(selected[j]) for j in range(self.k) if j != i]
            q       = torch.stack(others).mean(0) if others else elem

            out, _  = mha(q, elem, elem)                              # (B, T, D)
            out     = out + selected[i]                               # residual on raw (pre-norm)
            out     = pool(out.transpose(1, 2)).transpose(1, 2)       # depthwise conv
            out     = proj(out.transpose(1, 2)).transpose(1, 2)       # (B, pool_out, D)
            pooled.append(out)                                         # (B, pool_out, D)

        # Stack along time: (B, k*pool_out, D) → transpose → (B, D, k*pool_out)
        combined = torch.cat(pooled, dim=1)                           # (B, k*pool_out, D)
        return combined.transpose(1, 2)                               # (B, D, k*pool_out)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, waveform):
        with torch.no_grad():
            layer_outputs, _ = self.wavlm.extract_features(waveform)

        acoustic_layers = layer_outputs[:self.split]
        semantic_layers = layer_outputs[self.split:]

        # ── Stream A ─────────────────────────────────────────────────────────
        selected_a = self._select_layers(acoustic_layers, self.weights_acoustic)
        feat_a     = self._process_stream(
            selected_a, self.cross_attns_acoustic,
            self.pools_acoustic, self.proj_acoustic, self.norms_acoustic
        )                                                              # (B, D, k*pool_out)
        out_a = self.asp_acoustic(feat_a)                             # (B, 2*D)

        # ── Stream B ─────────────────────────────────────────────────────────
        selected_s = self._select_layers(semantic_layers, self.weights_semantic)
        feat_s     = self._process_stream(
            selected_s, self.cross_attns_semantic,
            self.pools_semantic, self.proj_semantic, self.norms_semantic
        )                                                              # (B, D, k*pool_out)
        out_s = self.asp_semantic(feat_s)                             # (B, 2*D)

        # ── Fusion gate ───────────────────────────────────────────────────────
        fused = torch.cat([out_a, out_s], dim=-1)                     # (B, 4*D)
        gate  = self.fusion_gate(fused)                               # (B, 2)

        # Soft gated sum of streams
        out_gated = gate[:, 0:1] * out_a + gate[:, 1:2] * out_s      # (B, 2*D)

        return self.out_proj(out_gated)                               # (B, final_dim)



"""
Best wavlm model on MSP 1 wavlm + attentivestatspool + classifier
"""

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, kernel_size=1),   # W·h_t + b
            nn.Tanh(),
            nn.Conv1d(bottleneck, in_dim, kernel_size=1),   # per-channel scores
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        e = self.attention(x)               # (B, C, T) — one score per channel & frame
        if mask is not None:
            e = e.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(e, dim=-1)    # (B, C, T)
        mean = (alpha * x).sum(dim=-1)      # (B, C)
        mean_sq = (alpha * x.pow(2)).sum(dim=-1)   # (B, C)
        std = (mean_sq - mean.pow(2)).clamp(min=1e-9).sqrt()  # (B, C)
        return torch.cat([mean, std], dim=-1)       # (B, 2*C)



class WavLM_Finetune_N(nn.Module):
    def __init__(self, bundle=torchaudio.pipelines.WAVLM_LARGE, finetune=(9,23), asp_in_dim=1024, asp_bottleneck=256,
                 droput=0.3, num_classifier_layers=3, classes=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.wavlm = bundle.get_model()
        self.wavlm.requires_grad_(False)
        if type(finetune) == tuple:
            for i in range(finetune[0], finetune[1]+1):
                self.wavlm.model.encoder.transformer.layers[i].requires_grad_(True)
        else:
            raise Exception("'finetune' should be like (9,15)")
        
        self.pooling_layer = AttentiveStatsPool(asp_in_dim, asp_bottleneck)
        
        linear_in = asp_in_dim*2
        self.classifier_layer = nn.Sequential(
            nn.Linear(linear_in, linear_in),
            nn.LayerNorm(linear_in),
            nn.ReLU(),
            nn.Dropout(droput)
        )
        self.classifier = nn.ModuleList([copy.deepcopy(self.classifier_layer) for _ in range(num_classifier_layers)])
        self.classifier.append(nn.Linear(linear_in, classes))
    def forward(self, X):
        o, _ = self.wavlm(X)
        asp_in = o.transpose(1,2)
        o = self.pooling_layer(asp_in)
        for i in self.classifier:
            o = i(o)
        return o

    
    
class HuBERT_Finetune_Full(nn.Module):
    def __init__(self, bundle=torchaudio.pipelines.HUBERT_LARGE, asp_in_dim=1024, asp_bottleneck=256,
                 dropout=0.3, num_classifier_layers=3, classes=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hubert = bundle.get_model()
        self.hubert.requires_grad_(True)  # Full fine-tune — all layers unfrozen

        self.pooling_layer = AttentiveStatsPool(asp_in_dim, asp_bottleneck)

        linear_in = asp_in_dim * 2
        self.classifier_layer = nn.Sequential(
            nn.Linear(linear_in, linear_in),
            nn.LayerNorm(linear_in),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.ModuleList(
            [copy.deepcopy(self.classifier_layer) for _ in range(num_classifier_layers)]
        )
        self.classifier.append(nn.Linear(linear_in, classes))

    def forward(self, X):
        o, _ = self.hubert(X)
        asp_in = o.transpose(1, 2)
        o = self.pooling_layer(asp_in)
        for i in self.classifier:
            o = i(o)
        return o
    
class ProjectionNetwork(nn.Module):
    def __init__(self, dims=98000, num_classes=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.epilogue = nn.Sequential(
            nn.Linear(in_features=dims, out_features=dims//2),
            nn.Linear(in_features=dims//2, out_features=dims//4),
            nn.ReLU(),
            nn.Linear(in_features=dims//4, out_features=dims//6),
            nn.ReLU(),
            nn.Linear(in_features=dims//6, out_features=num_classes)
        )
    
    def forward(self, X):
        return self.epilogue(X)
    

# class WavLMLayerWeighted_FN_Layers(nn.Module):
#     def __init__(
#         self,
#         bundle=torchaudio.pipelines.WAVLM_LARGE,
#         k=3,
#         num_heads=8,
#         dropout=0.3,
#         pool_out=8,
#         final_dim=1024,
#         candidate_range=(9, 15),
#         select_n=3,
#         seed=None,
#         finetune_wavlm=False,   # False by default because selective internal unfreezing is version-dependent
#     ):
#         super().__init__()

#         self.k = k
#         self.final_dim = final_dim
#         self.candidate_start, self.candidate_end = candidate_range
#         self.num_candidate_layers = self.candidate_end - self.candidate_start + 1

#         if seed is not None:
#             random.seed(seed)

#         self.selected_layers = sorted(random.sample(range(self.candidate_start, self.candidate_end + 1), select_n))
#         self.wavlm = bundle.get_model()

#         # Freeze or unfreeze full wavlm
#         if finetune_wavlm:
#             self.wavlm.requires_grad_(True)
#         else:
#             self.wavlm.requires_grad_(False)


#         with torch.no_grad():
#             feats, _ = self.wavlm.extract_features(torch.zeros(1, 16000), num_layers=self.candidate_end + 1)
#         dim = feats[0].shape[-1]
#         self.hidden_dim = dim

#         self.weights = nn.Parameter(torch.ones(self.num_candidate_layers))
#         self.cross_attns = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True) for _ in range(k)])
#         self.pools = nn.ModuleList([nn.Conv1d(dim, dim, kernel_size=1, groups=dim) for _ in range(k)])
#         self.proj = nn.ModuleList([nn.AdaptiveAvgPool1d(pool_out) for _ in range(k)])
#         self.out_proj = nn.Linear(k * dim * pool_out, final_dim)

#     def forward(self, waveform):
#         layer_outputs, _ = self.wavlm.extract_features(waveform, num_layers=self.candidate_end + 1)

#         candidate_outputs = layer_outputs[self.candidate_start:self.candidate_end + 1]
#         w = torch.sigmoid(self.weights)
#         selected_local_idx = [layer_id - self.candidate_start for layer_id in self.selected_layers]
#         selected_weights = w[selected_local_idx]

#         selected = [candidate_outputs[idx] * selected_weights[i] for i, idx in enumerate(selected_local_idx)]
#         pooled = []

#         for i, (mha, pool, proj) in enumerate(zip(self.cross_attns, self.pools, self.proj)):
#             elem = selected[i]
#             others = [selected[j] for j in range(len(selected)) if j != i]
#             q = torch.stack(others, dim=0).mean(0) if len(others) > 0 else elem
#             out, _ = mha(q, elem, elem)
#             out = out + elem
#             out = pool(out.transpose(1, 2)).transpose(1, 2)
#             out = proj(out.transpose(1, 2)).transpose(1, 2)
#             pooled.append(out.flatten(1))

#         fused = torch.cat(pooled, dim=-1)
#         fused = self.out_proj(fused)
#         return fused

    






class SERes2Block(nn.Module):
    def __init__(self, channels, kernel_size, dilation, scale=8):
        super().__init__()
        self.width = int(math.floor(channels / scale))
        self.nums  = scale - 1
        num_pad    = math.floor(kernel_size / 2) * dilation

        self.conv1 = nn.Conv1d(channels, self.width * scale, kernel_size=1)
        self.bn1   = nn.BatchNorm1d(self.width * scale)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=kernel_size,
                      dilation=dilation, padding=num_pad)
            for _ in range(self.nums)
        ])
        self.bns  = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(self.width * scale, channels, kernel_size=1)
        self.bn3   = nn.BatchNorm1d(channels)
        self.se    = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, channels, kernel_size=1), nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        sp, parts = None, []
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            parts.append(sp)
        out = torch.cat(parts + [spx[self.nums]], dim=1)
        out = F.relu(self.bn3(self.conv3(out)))
        return out * self.se(out) + residual
    
    
class Student_ECAPA_TDNN(nn.Module):
    def __init__(self, C, num_blocks=3, scale=8, embd_dim=192):
        super().__init__()
        dilations = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
        assert num_blocks <= len(dilations), f"num_blocks must be <= {len(dilations)}"
        
        coef = 0.97
        self.register_buffer('pe_filter',
            torch.FloatTensor([-coef, 1.]).unsqueeze(0).unsqueeze(0))
        self.torchfbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80)

        self.freq_mask_width  = (0, 8)
        self.time_mask_width  = (0, 10)

        self.conv1 = nn.Conv1d(80, C, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(C)
        self.relu  = nn.ReLU()
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            SERes2Block(C, kernel_size=3, dilation=dilations[i], scale=scale) for i in range(num_blocks)
            ])

        self.layer4    = nn.Conv1d(num_blocks * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(1536 * 3, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, embd_dim)
        self.bn6 = nn.BatchNorm1d(embd_dim)

    def _fbank(self, x):
        x = F.pad(x.unsqueeze(1), (1, 0), 'reflect')
        x = F.conv1d(x, self.pe_filter).squeeze(1)
        x = self.torchfbank(x) + 1e-6
        x = x.log()
        return x - x.mean(dim=-1, keepdim=True)

    def _specaug(self, x):
        for dim, width_range in [(2, self.time_mask_width), (1, self.freq_mask_width)]:
            B, F, T = x.shape
            D = T if dim == 2 else F
            mlen = torch.randint(*width_range, (B, 1), device=x.device).unsqueeze(2)
            mpos = torch.randint(0, max(1, D - mlen.max()), (B, 1), device=x.device).unsqueeze(2)
            ar   = torch.arange(D, device=x.device).view(1, 1, -1)
            mask = ((mpos <= ar) & (ar < mpos + mlen)).any(1)
            x.masked_fill_(mask.unsqueeze(2 if dim == 1 else 1), 0.0)
        return x

    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self._fbank(x)
            if aug:
                x = self._specaug(x)

        x = self.relu(self.bn1(self.conv1(x)))
        outs, acc = [], x
        for i, blk in enumerate(self.blocks):
            acc = acc if i == 0 else acc + outs[-1]
            outs.append(blk(acc))

        x = self.relu(self.layer4(torch.cat(outs, dim=1)))
        t = x.size(-1)
        mu_rep  = x.mean(dim=2, keepdim=True).expand(-1, -1, t)
        std_rep = x.var(dim=2, keepdim=True).clamp(min=1e-4).sqrt().expand(-1, -1, t)
        w = self.attention(torch.cat([x, mu_rep, std_rep], dim=1))
        mu = (x * w).sum(dim=2)
        sg = ((x**2 * w).sum(dim=2) - mu**2).clamp(min=1e-4).sqrt()
        x = self.bn5(torch.cat([mu, sg], dim=1))
        return self.bn6(self.fc6(x))