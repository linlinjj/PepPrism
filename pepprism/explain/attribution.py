"""Multi-perspective residue attribution (attention, gradient, alanine-scan) and motif mining."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from pepprism.biophysics.descriptors import (
    batch_encode_sequences,
    compute_biophysical_profile,
    sanitize_sequence,
)
from pepprism.biophysics.scales import AA_GROUP
from pepprism.models.trainer import TrainedPepPrismBundle


@dataclass(frozen=True)
class ResidueAttribution:
    """Per-residue interpretability metrics."""

    position: int
    amino_acid: str
    group: str
    attention: float
    alanine_scan_delta: float
    grad_saliency: float
    combined_importance: float


@dataclass(frozen=True)
class PeptideExplanation:
    """Full multi-modal explanation for a single peptide prediction."""

    sequence: str
    probability: float
    uncertainty_std: float
    seq_gate_ratio: float
    bio_gate_ratio: float
    residue_attributions: list[ResidueAttribution]
    top_hotspots: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _compute_gradient_saliency(
    bundle: TrainedPepPrismBundle,
    sequence: str,
) -> np.ndarray:
    """Compute input-embedding x gradient saliency norm per residue."""
    tokens_np, channels_np, masks_np = batch_encode_sequences(
        [sequence], max_len=bundle.max_len
    )
    bio_raw = compute_biophysical_profile(sequence).to_feature_vector()[None, :]
    bio_norm = bundle.standardize_bio(bio_raw)

    tokens = torch.from_numpy(tokens_np)
    channels = torch.from_numpy(channels_np)
    masks = torch.from_numpy(masks_np)
    bio = torch.from_numpy(bio_norm)

    bundle.model.eval()
    if hasattr(bundle.model, "emb"):
        emb = bundle.model.emb(tokens).detach().requires_grad_(True)
        if hasattr(bundle.model, "encode_sequence"):
            out = bundle.model(tokens, channels, masks, bio, override_seq_embed=emb)
        else:
            out = bundle.model(tokens, channels, masks, bio)
        logit = out["logits"].sum()
        bundle.model.zero_grad()
        logit.backward()
        if emb.grad is not None:
            sal = (emb.grad * emb).abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
        else:
            sal = np.zeros(bundle.max_len, dtype=np.float32)
    else:
        sal = np.zeros(bundle.max_len, dtype=np.float32)

    seq_len = min(len(sequence), bundle.max_len)
    sal = sal[:seq_len]
    denom = float(sal.max()) + 1e-8
    return (sal / denom).astype(np.float32)


def explain_peptide(
    bundle: TrainedPepPrismBundle,
    sequence: str,
    mc_samples: int = 20,
) -> PeptideExplanation:
    """Generate attention, gradient saliency, and in silico alanine-scan attribution."""
    seq = sanitize_sequence(sequence, strict=True)
    seq_len = min(len(seq), bundle.max_len)

    base_res = bundle.predict_sequences([seq], mc_samples=mc_samples)
    p_base = float(base_res["probabilities"][0])
    p_std = float(base_res["uncertainty_std"][0])
    attn = base_res["attention"][0][:seq_len]
    gate_seq = float(base_res["seq_gate_ratio"][0])

    # In silico alanine scanning: mutate each position to neutral Alanine (or Glycine if already A)
    ala_variants: list[str] = []
    for i in range(seq_len):
        sub_aa = "G" if seq[i] == "A" else "A"
        ala_variants.append(seq[:i] + sub_aa + seq[i + 1 :])

    ala_res = bundle.predict_sequences(ala_variants, mc_samples=1)
    ala_probs = ala_res["probabilities"]
    ala_deltas = p_base - ala_probs  # Positive means original residue supports activity

    grad_sal = _compute_gradient_saliency(bundle, seq)

    attn_norm = attn / (float(attn.max()) + 1e-8)
    delta_pos = np.maximum(ala_deltas, 0.0)
    delta_norm = delta_pos / (float(delta_pos.max()) + 1e-8)
    combined = 0.45 * attn_norm + 0.30 * grad_sal + 0.25 * delta_norm
    if float(combined.max()) > 0:
        combined = combined / float(combined.max())

    residues: list[ResidueAttribution] = []
    for i in range(seq_len):
        residues.append(
            ResidueAttribution(
                position=i + 1,
                amino_acid=seq[i],
                group=AA_GROUP[seq[i]],
                attention=round(float(attn[i]), 4),
                alanine_scan_delta=round(float(ala_deltas[i]), 4),
                grad_saliency=round(float(grad_sal[i]), 4),
                combined_importance=round(float(combined[i]), 4),
            )
        )

    ranked = sorted(residues, key=lambda r: r.combined_importance, reverse=True)[:5]
    hotspots = [
        {
            "position": r.position,
            "amino_acid": r.amino_acid,
            "group": r.group,
            "combined_importance": r.combined_importance,
            "alanine_scan_delta": r.alanine_scan_delta,
        }
        for r in ranked
    ]

    return PeptideExplanation(
        sequence=seq,
        probability=round(p_base, 4),
        uncertainty_std=round(p_std, 4),
        seq_gate_ratio=round(gate_seq, 4),
        bio_gate_ratio=round(1.0 - gate_seq, 4),
        residue_attributions=residues,
        top_hotspots=hotspots,
    )


def mine_high_attention_motifs(
    bundle: TrainedPepPrismBundle,
    sequences: Sequence[str],
    labels: np.ndarray,
    k: int = 4,
    top_n: int = 15,
    prob_threshold: float = 0.7,
) -> list[dict[str, object]]:
    """Extract recurrent k-mer motifs centered at peak-attention positions in active peptides."""
    pos_seqs = [s for s, y in zip(sequences, labels) if int(y) == 1 and len(s) >= k]
    if not pos_seqs:
        return []

    pred = bundle.predict_sequences(pos_seqs, mc_samples=1)
    probs = pred["probabilities"]
    attns = pred["attention"]

    motif_counts: Counter[str] = Counter()
    motif_scores: dict[str, list[float]] = {}

    for seq, prob, attn in zip(pos_seqs, probs, attns):
        if float(prob) < prob_threshold:
            continue
        valid_len = min(len(seq), bundle.max_len)
        if valid_len < k:
            continue
        seq_attn = attn[:valid_len]
        # Score each k-mer window by mean attention
        window_scores = [
            (float(seq_attn[i : i + k].mean()), seq[i : i + k])
            for i in range(valid_len - k + 1)
        ]
        window_scores.sort(reverse=True, key=lambda item: item[0])
        for score, kmer in window_scores[:2]:
            motif_counts[kmer] += 1
            motif_scores.setdefault(kmer, []).append(score)

    ranked = []
    for kmer, count in motif_counts.most_common(top_n):
        ranked.append(
            {
                "motif": kmer,
                "count": int(count),
                "mean_attention": round(float(np.mean(motif_scores[kmer])), 4),
            }
        )
    return ranked
