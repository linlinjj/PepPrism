"""Deep sequence baselines and the PepPrismNet dual-branch gated fusion architecture."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pepprism.biophysics.scales import PAD_IDX, VOCAB_SIZE


class MaskedAdditiveAttention(nn.Module):
    """Temperature-scaled additive attention producing normalized residue importance weights."""

    def __init__(self, hidden_dim: int, attn_dim: int = 48, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.scorer = nn.Linear(attn_dim, 1, bias=False)
        self.temperature = temperature

    def forward(
        self,
        seq_hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # seq_hidden: (B, L, H), mask: (B, L) bool
        energy = self.scorer(torch.tanh(self.proj(seq_hidden))).squeeze(-1) / self.temperature
        energy = energy.masked_fill(~mask, -1e9)
        weights = torch.softmax(energy, dim=-1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pooled = torch.bmm(weights.unsqueeze(1), seq_hidden).squeeze(1)
        return pooled, weights


class MultiScaleDilatedConvBlock(nn.Module):
    """Parallel dilated 1D convolutions + Squeeze-and-Excitation channel attention."""

    def __init__(self, dim: int, dilations: tuple[int, int, int] = (1, 1, 2), dropout: float = 0.2) -> None:
        super().__init__()
        b_dim = dim // 3
        rem = dim - 2 * b_dim
        d1, d2, d3 = dilations
        self.conv1 = nn.Conv1d(dim, b_dim, kernel_size=3, padding=d1, dilation=d1)
        self.conv2 = nn.Conv1d(dim, b_dim, kernel_size=5, padding=2 * d2, dilation=d2)
        self.conv3 = nn.Conv1d(dim, rem, kernel_size=3, padding=d3, dilation=d3)
        self.se = nn.Sequential(
            nn.Linear(dim, max(8, dim // 4)),
            nn.GELU(),
            nn.Linear(max(8, dim // 4), dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        xt = x.transpose(1, 2)
        h1 = F.gelu(self.conv1(xt))
        h2 = F.gelu(self.conv2(xt))
        h3 = F.gelu(self.conv3(xt))
        cat = torch.cat([h1, h2, h3], dim=1).transpose(1, 2)
        cat = cat * mask.unsqueeze(-1).float()
        # Masked Squeeze-and-Excitation channel attention
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        ch_mean = cat.sum(dim=1) / lengths
        ch_scale = self.se(ch_mean).unsqueeze(1)
        cat = cat * ch_scale
        out = self.norm(x + self.drop(self.proj(cat)))
        return out * mask.unsqueeze(-1).float()


class SeqCNN1DBaseline(nn.Module):
    """Pure sequence multi-kernel 1D-CNN baseline (AMPScanner-style convolutional pooling)."""

    def __init__(self, embed_dim: int = 48, num_filters: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        channels: torch.Tensor,
        mask: torch.Tensor,
        bio_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del channels, bio_features
        x = self.emb(tokens).transpose(1, 2)
        mask_f = mask.unsqueeze(1).float()
        pooled_branches = []
        for conv in self.convs:
            h = F.relu(conv(x)) * mask_f + (~mask.unsqueeze(1)).float() * (-1e9)
            pooled_branches.append(h.max(dim=-1).values)
        feat = torch.cat(pooled_branches, dim=-1)
        logits = self.head(feat).squeeze(-1)
        uniform_attn = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        return {"logits": logits, "attention": uniform_attn}


class BiGRUAttentionBaseline(nn.Module):
    """Sequence-only BiGRU + additive attention baseline (AMPlify-style architecture)."""

    def __init__(self, embed_dim: int = 48, hidden_dim: int = 48, dropout: float = 0.25) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = MaskedAdditiveAttention(hidden_dim * 2, attn_dim=48)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        channels: torch.Tensor,
        mask: torch.Tensor,
        bio_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del channels, bio_features
        x = self.drop(self.emb(tokens))
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=tokens.size(1)
        )
        pooled, weights = self.attn(h, mask)
        logits = self.head(pooled).squeeze(-1)
        return {"logits": logits, "attention": weights}


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to residue representations."""

    def __init__(self, dim: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class PepPrismNet(nn.Module):
    """Dual-branch Multi-Scale Biophysical-Sequence Network with Gated Fusion (GBSF).

    Branch A combines residue embeddings + 7 biophysical channels, sinusoidal position
    encoding, stacked Multi-Scale Dilated Squeeze-and-Excitation 1D Convolutions,
    and dual (additive attention + global max) pooling.
    Branch B encodes 17 first-principles biophysical descriptors via a residual MLP.
    """

    def __init__(
        self,
        bio_dim: int = 17,
        embed_dim: int = 48,
        residue_ch_dim: int = 7,
        seq_dim: int = 64,
        fusion_dim: int = 64,
        dropout: float = 0.25,
        mode: Literal["gbsf", "concat", "seq_only", "bio_only"] = "gbsf",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.fusion_dim = fusion_dim

        # Branch A: Residue sequence + biophysical prior channels
        self.emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.seq_in = nn.Sequential(
            nn.Linear(embed_dim + residue_ch_dim, seq_dim),
            nn.LayerNorm(seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = SinusoidalPositionalEncoding(seq_dim, max_len=256)
        self.ms_conv1 = MultiScaleDilatedConvBlock(seq_dim, dilations=(1, 1, 2), dropout=dropout)
        self.ms_conv2 = MultiScaleDilatedConvBlock(seq_dim, dilations=(2, 2, 4), dropout=dropout)
        self.attn = MaskedAdditiveAttention(seq_dim, attn_dim=48, temperature=1.0)
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Branch B: Global biophysical descriptor residual MLP
        self.bio_in = nn.Sequential(
            nn.Linear(bio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bio_res = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gated fusion or concatenation fusion
        self.gate_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        self.concat_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    def encode_sequence(
        self,
        tokens: torch.Tensor,
        channels: torch.Tensor,
        mask: torch.Tensor,
        override_seq_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run sequence branch and return projected representation and residue attention."""
        emb = override_seq_embed if override_seq_embed is not None else self.emb(tokens)
        x = torch.cat([emb, channels], dim=-1)
        x = self.seq_in(x) * mask.unsqueeze(-1).float()
        x = self.pos_enc(x) * mask.unsqueeze(-1).float()
        x = self.ms_conv1(x, mask)
        x = self.ms_conv2(x, mask)

        attn_pooled, weights = self.attn(x, mask)
        masked_for_max = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pooled = masked_for_max.max(dim=1).values
        combined = torch.cat([attn_pooled, max_pooled], dim=-1)
        return self.seq_proj(combined), weights

    def encode_biophysics(self, bio_features: torch.Tensor) -> torch.Tensor:
        """Run residual MLP on standardized global biophysical descriptors."""
        h = self.bio_in(bio_features)
        return h + self.bio_res(h)

    def forward(
        self,
        tokens: torch.Tensor,
        channels: torch.Tensor,
        mask: torch.Tensor,
        bio_features: torch.Tensor,
        override_seq_embed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.mode == "bio_only":
            h_bio = self.encode_biophysics(bio_features)
            fused = h_bio
            gate = torch.zeros_like(h_bio)
            attn_weights = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            h_seq, attn_weights = self.encode_sequence(
                tokens, channels, mask, override_seq_embed=override_seq_embed
            )
            if self.mode == "seq_only":
                fused = h_seq
                gate = torch.ones_like(h_seq)
            elif self.mode == "concat":
                h_bio = self.encode_biophysics(bio_features)
                cat = torch.cat([h_seq, h_bio], dim=-1)
                fused = self.concat_proj(cat)
                gate = torch.full_like(h_seq, 0.5)
            else:  # "gbsf"
                h_bio = self.encode_biophysics(bio_features)
                cat = torch.cat([h_seq, h_bio], dim=-1)
                gate = torch.sigmoid(self.gate_layer(cat))
                fused = gate * h_seq + (1.0 - gate) * h_bio

        logits = self.classifier(fused).squeeze(-1)
        return {
            "logits": logits,
            "attention": attn_weights,
            "gate": gate,
            "seq_gate_ratio": gate.mean(dim=-1),
        }
