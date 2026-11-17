"""Training, calibration, evaluation metrics, and checkpoint serialization for PepPrism."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
from typing import Literal, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pepprism.biophysics.descriptors import (
    batch_encode_sequences,
    compute_biophysical_profile,
)
from pepprism.models.pepprism_net import (
    BiGRUAttentionBaseline,
    PepPrismNet,
    SeqCNN1DBaseline,
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Comprehensive binary classification and probability calibration metrics."""

    accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    f1: float
    mcc: float
    auroc: float
    auprc: float
    brier: float
    ece: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error across uniform probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= low) & (y_prob <= high)
        else:
            in_bin = (y_prob >= low) & (y_prob < high)
        cnt = int(in_bin.sum())
        if cnt > 0:
            acc_bin = float(y_true[in_bin].mean())
            conf_bin = float(y_prob[in_bin].mean())
            ece += (cnt / n) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Compute full classification, discrimination, and calibration metrics."""
    y_true_i = np.asarray(y_true, dtype=np.int64)
    y_prob_f = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_f >= threshold).astype(np.int64)

    tn = int(((y_true_i == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_i == 0) & (y_pred == 1)).sum())
    specificity = tn / float(tn + fp) if (tn + fp) > 0 else 0.0

    has_both = len(np.unique(y_true_i)) == 2
    auroc = float(roc_auc_score(y_true_i, y_prob_f)) if has_both else 0.5
    auprc = float(average_precision_score(y_true_i, y_prob_f)) if has_both else 0.0

    return EvaluationMetrics(
        accuracy=round(float(accuracy_score(y_true_i, y_pred)), 4),
        precision=round(float(precision_score(y_true_i, y_pred, zero_division=0)), 4),
        sensitivity=round(float(recall_score(y_true_i, y_pred, zero_division=0)), 4),
        specificity=round(float(specificity), 4),
        f1=round(float(f1_score(y_true_i, y_pred, zero_division=0)), 4),
        mcc=round(float(matthews_corrcoef(y_true_i, y_pred)), 4) if has_both else 0.0,
        auroc=round(auroc, 4),
        auprc=round(auprc, 4),
        brier=round(float(brier_score_loss(y_true_i, y_prob_f)), 4),
        ece=round(compute_ece(y_true_i, y_prob_f), 4),
    )


def set_deterministic_seed(seed: int = 42) -> None:
    """Set reproducible seeds across random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainedPepPrismBundle:
    """Self-contained model weights and feature scaler for inference and explanation."""

    model: nn.Module
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    max_len: int
    arch: str
    task: str

    def standardize_bio(self, bio_raw: np.ndarray) -> np.ndarray:
        return ((bio_raw - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    def predict_sequences(
        self,
        sequences: Sequence[str],
        mc_samples: int = 1,
        batch_size: int = 512,
    ) -> dict[str, np.ndarray]:
        """Run deterministic or MC-Dropout uncertainty-aware inference on sequences."""
        tokens_np, channels_np, masks_np = batch_encode_sequences(
            sequences, max_len=self.max_len
        )
        bio_raw = np.vstack(
            [compute_biophysical_profile(s).to_feature_vector() for s in sequences]
        ).astype(np.float32)
        bio_norm = self.standardize_bio(bio_raw)

        n_total = len(sequences)
        probs_all = np.zeros(n_total, dtype=np.float64)
        stds_all = np.zeros(n_total, dtype=np.float64)
        attn_all = np.zeros((n_total, self.max_len), dtype=np.float32)
        gate_all = np.zeros(n_total, dtype=np.float32)

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            tokens = torch.from_numpy(tokens_np[start:end])
            channels = torch.from_numpy(channels_np[start:end])
            masks = torch.from_numpy(masks_np[start:end])
            bio = torch.from_numpy(bio_norm[start:end])
            l_max = max(1, int(masks.sum(dim=1).max().item()))
            t_trim, c_trim, m_trim = tokens[:, :l_max], channels[:, :l_max], masks[:, :l_max]

            if mc_samples <= 1:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(t_trim, c_trim, m_trim, bio)
                    p = torch.sigmoid(out["logits"]).cpu().numpy()
                    a = out["attention"].cpu().numpy()
                    g = (
                        out["seq_gate_ratio"].cpu().numpy()
                        if "seq_gate_ratio" in out
                        else np.full_like(p, 0.5, dtype=np.float32)
                    )
                probs_all[start:end] = p
                attn_all[start:end, :l_max] = a
                gate_all[start:end] = g
            else:
                self.model.train()
                sample_probs = []
                attn_sum = None
                gate_sum = None
                with torch.no_grad():
                    for _ in range(mc_samples):
                        out = self.model(t_trim, c_trim, m_trim, bio)
                        p = torch.sigmoid(out["logits"]).cpu().numpy()
                        sample_probs.append(p)
                        a = out["attention"].cpu().numpy()
                        attn_sum = a if attn_sum is None else (attn_sum + a)
                        if "seq_gate_ratio" in out:
                            g = out["seq_gate_ratio"].cpu().numpy()
                            gate_sum = g if gate_sum is None else (gate_sum + g)
                self.model.eval()
                stacked = np.stack(sample_probs, axis=0)
                probs_all[start:end] = stacked.mean(axis=0)
                stds_all[start:end] = stacked.std(axis=0)
                attn_all[start:end, :l_max] = attn_sum / float(mc_samples)
                gate_all[start:end] = (
                    (gate_sum / float(mc_samples))
                    if gate_sum is not None
                    else np.full(end - start, 0.5, dtype=np.float32)
                )

        return {
            "probabilities": probs_all,
            "uncertainty_std": stds_all,
            "attention": attn_all,
            "seq_gate_ratio": gate_all,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "scaler_mean": self.scaler_mean,
                "scaler_scale": self.scaler_scale,
                "max_len": self.max_len,
                "arch": self.arch,
                "task": self.task,
            },
            p,
        )

    @classmethod
    def load(cls, path: Path | str) -> "TrainedPepPrismBundle":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        arch = ckpt["arch"]
        bio_dim = len(ckpt["scaler_mean"])
        model = build_deep_architecture(arch, bio_dim=bio_dim)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(
            model=model,
            scaler_mean=ckpt["scaler_mean"],
            scaler_scale=ckpt["scaler_scale"],
            max_len=int(ckpt["max_len"]),
            arch=arch,
            task=ckpt["task"],
        )


def build_deep_architecture(
    arch: Literal[
        "SeqCNN1D",
        "BiGRUAttention",
        "PepPrism_SeqOnly",
        "PepPrism_BioOnly",
        "PepPrism_Concat",
        "PepPrismNet",
    ],
    bio_dim: int = 17,
) -> nn.Module:
    """Instantiate one of the deep baselines, ablations, or full PepPrismNet."""
    if arch == "SeqCNN1D":
        return SeqCNN1DBaseline()
    if arch == "BiGRUAttention":
        return BiGRUAttentionBaseline()
    if arch == "PepPrism_SeqOnly":
        return PepPrismNet(bio_dim=bio_dim, mode="seq_only")
    if arch == "PepPrism_BioOnly":
        return PepPrismNet(bio_dim=bio_dim, mode="bio_only")
    if arch == "PepPrism_Concat":
        return PepPrismNet(bio_dim=bio_dim, mode="concat")
    if arch == "PepPrismNet":
        return PepPrismNet(bio_dim=bio_dim, mode="gbsf")
    raise ValueError(f"Unsupported deep architecture '{arch}'.")


def predict_pepprism_probabilities(
    model: nn.Module,
    tokens: np.ndarray,
    channels: np.ndarray,
    masks: np.ndarray,
    bio: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run batched deterministic evaluation inference."""
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(tokens),
            torch.from_numpy(channels),
            torch.from_numpy(masks),
            torch.from_numpy(bio),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for b_tok, b_ch, b_mask, b_bio in loader:
            l_max = max(1, int(b_mask.sum(dim=1).max().item()))
            out = model(b_tok[:, :l_max], b_ch[:, :l_max], b_mask[:, :l_max], b_bio)
            preds.append(torch.sigmoid(out["logits"]).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float64)


def train_deep_model(
    arch: str,
    train_sequences: Sequence[str],
    train_labels: np.ndarray,
    train_bio_raw: np.ndarray,
    task: str = "AMP",
    max_len: int = 128,
    epochs: int = 28,
    batch_size: int = 128,
    lr: float = 1.8e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    seed: int = 42,
) -> TrainedPepPrismBundle:
    """Train a deep baseline or PepPrismNet with internal stratified early stopping."""
    set_deterministic_seed(seed)
    # Limit intra-op threads to prevent OpenMP spin-wait slowdown on small sequence batches
    torch.set_num_threads(min(4, max(1, (os.cpu_count() or 4) // 4)))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, val_idx = next(splitter.split(np.zeros(len(train_labels)), train_labels))

    # Fit biophysical scaler strictly on training split
    mean = train_bio_raw[tr_idx].mean(axis=0)
    scale = train_bio_raw[tr_idx].std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    bio_norm = ((train_bio_raw - mean) / scale).astype(np.float32)

    tokens, channels, masks = batch_encode_sequences(train_sequences, max_len=max_len)

    tr_ds = TensorDataset(
        torch.from_numpy(tokens[tr_idx]),
        torch.from_numpy(channels[tr_idx]),
        torch.from_numpy(masks[tr_idx]),
        torch.from_numpy(bio_norm[tr_idx]),
        torch.from_numpy(train_labels[tr_idx].astype(np.float32)),
    )
    eff_batch = 512 if len(tr_idx) > 25000 and batch_size == 128 else batch_size
    if len(tr_idx) > 25000:
        # Sort by sequence length before creating batches to reduce padding overhead on CPU
        lens = masks[tr_idx].sum(axis=1)
        order = np.argsort(lens)
        tr_idx_ordered = tr_idx[order]
        tr_ds = TensorDataset(
            torch.from_numpy(tokens[tr_idx_ordered]),
            torch.from_numpy(channels[tr_idx_ordered]),
            torch.from_numpy(masks[tr_idx_ordered]),
            torch.from_numpy(bio_norm[tr_idx_ordered]),
            torch.from_numpy(train_labels[tr_idx_ordered].astype(np.float32)),
        )
        base_batches = [
            list(range(i, min(i + eff_batch, len(tr_idx_ordered))))
            for i in range(0, len(tr_idx_ordered), eff_batch)
        ]

        class _ShuffledBatchSampler:
            def __init__(self, b_list: list[list[int]], rng_seed: int) -> None:
                self._batches = b_list
                self._rng = random.Random(rng_seed)

            def __iter__(self):
                order_idx = list(range(len(self._batches)))
                self._rng.shuffle(order_idx)
                for idx in order_idx:
                    yield self._batches[idx]

            def __len__(self) -> int:
                return len(self._batches)

        tr_loader = DataLoader(tr_ds, batch_sampler=_ShuffledBatchSampler(base_batches, seed))
    else:
        tr_loader = DataLoader(tr_ds, batch_size=eff_batch, shuffle=True)

    model = build_deep_architecture(arch, bio_dim=bio_norm.shape[1])

    n_pos = float((train_labels[tr_idx] == 1).sum())
    n_neg = float((train_labels[tr_idx] == 0).sum())
    pos_weight_val = min(max(n_neg / max(n_pos, 1.0), 1.0), 6.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    best_state = None
    no_improve = 0

    for _epoch in range(epochs):
        model.train()
        for b_tok, b_ch, b_mask, b_bio, b_y in tr_loader:
            l_max = max(1, int(b_mask.sum(dim=1).max().item()))
            optimizer.zero_grad()
            out = model(b_tok[:, :l_max], b_ch[:, :l_max], b_mask[:, :l_max], b_bio)
            loss = criterion(out["logits"], b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

        val_probs = predict_pepprism_probabilities(
            model,
            tokens[val_idx],
            channels[val_idx],
            masks[val_idx],
            bio_norm[val_idx],
        )
        val_y = train_labels[val_idx]
        if len(np.unique(val_y)) == 2:
            val_score = 0.5 * roc_auc_score(val_y, val_probs) + 0.5 * average_precision_score(
                val_y, val_probs
            )
        else:
            val_score = float(accuracy_score(val_y, (val_probs >= 0.5).astype(int)))

        if val_score > best_score + 1e-4:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return TrainedPepPrismBundle(
        model=model,
        scaler_mean=mean.astype(np.float32),
        scaler_scale=scale.astype(np.float32),
        max_len=max_len,
        arch=arch,
        task=task,
    )
