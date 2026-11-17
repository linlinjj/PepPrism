"""First-principles biophysical descriptor and sequence feature computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

from Bio.SeqUtils import ProtParamData
import numpy as np

from pepprism.biophysics.scales import (
    AA_SET,
    AA_TO_IDX,
    CHOU_FASMAN_HELIX,
    CHOU_FASMAN_SHEET,
    EISENBERG_HYDROPHOBICITY,
    HOPP_WOODS,
    KYTE_DOOLITTLE,
    PAD_IDX,
    PKA_C_TERMINAL,
    PKA_NEG,
    PKA_N_TERMINAL,
    PKA_POS,
    RADZICKA_WOLFENDEN,
    RESIDUE_CHANNEL_TABLE,
    RESIDUE_MASS,
    STANDARD_AAS,
    UNK_IDX,
)

# Precompute standardized scales for Chou (2001) PseAAC correlation factors
def _zscore_scale(scale: dict[str, float]) -> dict[str, float]:
    vals = np.array([scale[aa] for aa in STANDARD_AAS], dtype=np.float64)
    mean, std = float(vals.mean()), float(vals.std()) + 1e-12
    return {aa: (scale[aa] - mean) / std for aa in STANDARD_AAS}


_PSE_HYDROPHOBICITY = _zscore_scale(KYTE_DOOLITTLE)
_PSE_HYDROPHILICITY = _zscore_scale(HOPP_WOODS)
_PSE_MASS = _zscore_scale(RESIDUE_MASS)

# Precompute 20x20 DIWV matrix for fast O(L) Guruprasad instability index calculation
_DIWV_MAT = np.zeros((20, 20), dtype=np.float64)
for _i, _aa1 in enumerate(STANDARD_AAS):
    _row = ProtParamData.DIWV.get(_aa1, {})
    for _j, _aa2 in enumerate(STANDARD_AAS):
        _DIWV_MAT[_i, _j] = float(_row.get(_aa2, 1.0))

_AA_FLAT_IDX = {aa: idx for idx, aa in enumerate(STANDARD_AAS)}

BIOPHYSICAL_FEATURE_NAMES: tuple[str, ...] = (
    "length",
    "molecular_weight",
    "net_charge_ph74",
    "charge_density_ph74",
    "isoelectric_point",
    "gravy_kd",
    "mean_eisenberg_h",
    "hydrophobic_moment_alpha",
    "max_window_moment_alpha",
    "hydrophobic_moment_beta",
    "boman_index",
    "instability_index",
    "aliphatic_index",
    "aromaticity",
    "helix_propensity",
    "sheet_propensity",
    "cationic_fraction",
)


@dataclass(frozen=True)
class BiophysicalProfile:
    """Comprehensive biophysical profile of a peptide sequence."""

    sequence: str
    length: int
    molecular_weight: float
    net_charge_ph74: float
    charge_density_ph74: float
    isoelectric_point: float
    gravy_kd: float
    mean_eisenberg_h: float
    hydrophobic_moment_alpha: float
    max_window_moment_alpha: float
    hydrophobic_moment_beta: float
    boman_index: float
    instability_index: float
    aliphatic_index: float
    aromaticity: float
    helix_propensity: float
    sheet_propensity: float
    cationic_fraction: float
    hemolysis_risk_heuristic: float
    ph_titration_curve: list[dict[str, float]]

    def to_feature_vector(self) -> np.ndarray:
        """Return the 17 numeric biophysical features aligned with BIOPHYSICAL_FEATURE_NAMES."""
        return np.array(
            [getattr(self, name) for name in BIOPHYSICAL_FEATURE_NAMES],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize profile to a JSON-compatible dictionary."""
        return asdict(self)


def sanitize_sequence(sequence: str, strict: bool = True) -> str:
    """Normalize whitespace/case and validate standard 20 amino acid composition."""
    cleaned = "".join(sequence.strip().split()).upper()
    if not cleaned:
        raise ValueError("Peptide sequence cannot be empty.")
    invalid = sorted(set(cleaned) - AA_SET)
    if invalid and strict:
        raise ValueError(
            f"Unsupported non-standard amino acid symbol(s): {', '.join(invalid)}"
        )
    return cleaned


def is_valid_standard_sequence(sequence: str, min_len: int = 5, max_len: int = 200) -> bool:
    """Return True if sequence uses only the 20 standard amino acids within length bounds."""
    cleaned = "".join(sequence.strip().split()).upper()
    return min_len <= len(cleaned) <= max_len and all(ch in AA_SET for ch in cleaned)


def _count_aas(seq: str) -> dict[str, int]:
    counts = {aa: 0 for aa in STANDARD_AAS}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
    return counts


def net_charge_at_ph(
    counts: dict[str, int],
    ph: float,
    n_term_aa: str = "",
    c_term_aa: str = "",
) -> float:
    """Compute peptide net charge at pH via Bjellqvist / Henderson-Hasselbalch model."""
    nterm_pka = PKA_N_TERMINAL.get(n_term_aa, PKA_POS["N_term"])
    cterm_pka = PKA_C_TERMINAL.get(c_term_aa, PKA_NEG["C_term"])

    pos_charge = 1.0 / (1.0 + 10.0 ** (ph - nterm_pka))
    for aa in ("K", "R", "H"):
        cnt = counts.get(aa, 0)
        if cnt:
            pos_charge += cnt / (1.0 + 10.0 ** (ph - PKA_POS[aa]))

    neg_charge = 1.0 / (1.0 + 10.0 ** (cterm_pka - ph))
    for aa in ("D", "E", "C", "Y"):
        cnt = counts.get(aa, 0)
        if cnt:
            neg_charge += cnt / (1.0 + 10.0 ** (PKA_NEG[aa] - ph))

    return pos_charge - neg_charge


def compute_isoelectric_point(
    counts: dict[str, int],
    n_term_aa: str = "",
    c_term_aa: str = "",
    tol: float = 1e-5,
) -> float:
    """Solve net_charge_at_ph(pI) = 0 on [0, 14] using bisection."""
    low, high = 0.0, 14.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if net_charge_at_ph(counts, mid, n_term_aa=n_term_aa, c_term_aa=c_term_aa) > 0.0:
            low = mid
        else:
            high = mid
        if (high - low) < tol:
            break
    return 0.5 * (low + high)


def compute_hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """Compute Eisenberg (1984) normalized mean hydrophobic moment at periodic angle delta."""
    n = len(seq)
    if n == 0:
        return 0.0
    delta = math.radians(angle_deg)
    sum_cos = 0.0
    sum_sin = 0.0
    for idx, aa in enumerate(seq):
        h = EISENBERG_HYDROPHOBICITY[aa]
        theta = idx * delta
        sum_cos += h * math.cos(theta)
        sum_sin += h * math.sin(theta)
    return math.hypot(sum_cos, sum_sin) / n


def compute_max_window_moment(seq: str, window: int = 11, angle_deg: float = 100.0) -> float:
    """Compute maximum sliding-window Eisenberg hydrophobic moment."""
    n = len(seq)
    if n <= window:
        return compute_hydrophobic_moment(seq, angle_deg=angle_deg)
    best = 0.0
    for start in range(n - window + 1):
        m = compute_hydrophobic_moment(seq[start : start + window], angle_deg=angle_deg)
        if m > best:
            best = m
    return best


def compute_instability_index(seq: str) -> float:
    """Compute Guruprasad et al. (1990) dipeptide instability index."""
    n = len(seq)
    if n < 2:
        return 0.0
    indices = [_AA_FLAT_IDX[ch] for ch in seq]
    score = sum(_DIWV_MAT[indices[i], indices[i + 1]] for i in range(n - 1))
    return (10.0 / n) * score


def compute_hemolysis_risk_heuristic(
    gravy: float,
    mu_h_alpha: float,
    net_charge_ph74: float,
    length: int,
) -> float:
    """Estimate physicochemical hemolysis risk in [0, 1] from amphipathicity/hydrophobicity balance.

    Excessive global hydrophobicity combined with strong alpha-helical hydrophobic moment
    increases non-selective insertion into zwitterionic mammalian erythrocyte membranes,
    whereas moderate cationic charge favors electrostatic selectivity toward anionic bacterial membranes.
    """
    charge_density = net_charge_ph74 / max(length, 1)
    logit = 1.65 * gravy + 3.20 * mu_h_alpha - 2.40 * max(charge_density, 0.0) - 1.15
    return float(1.0 / (1.0 + math.exp(-max(min(logit, 20.0), -20.0))))


def compute_biophysical_profile(sequence: str) -> BiophysicalProfile:
    """Compute full first-principles biophysical profile for one peptide sequence."""
    seq = sanitize_sequence(sequence, strict=True)
    n = len(seq)
    counts = _count_aas(seq)

    mw = sum(RESIDUE_MASS[aa] * cnt for aa, cnt in counts.items()) + 18.01528
    nterm_aa, cterm_aa = seq[0], seq[-1]
    charge_74 = net_charge_at_ph(counts, 7.4, n_term_aa=nterm_aa, c_term_aa=cterm_aa)
    pi_val = compute_isoelectric_point(counts, n_term_aa=nterm_aa, c_term_aa=cterm_aa)
    gravy = sum(KYTE_DOOLITTLE[aa] * cnt for aa, cnt in counts.items()) / n
    mean_eis = sum(EISENBERG_HYDROPHOBICITY[aa] * cnt for aa, cnt in counts.items()) / n
    mu_alpha = compute_hydrophobic_moment(seq, angle_deg=100.0)
    max_win_alpha = compute_max_window_moment(seq, window=11, angle_deg=100.0)
    mu_beta = compute_hydrophobic_moment(seq, angle_deg=180.0)
    boman = -sum(RADZICKA_WOLFENDEN[aa] * cnt for aa, cnt in counts.items()) / n
    instability = compute_instability_index(seq)
    aliphatic = 100.0 * (
        counts["A"] + 2.9 * counts["V"] + 3.9 * (counts["I"] + counts["L"])
    ) / n
    aromaticity = (counts["F"] + counts["W"] + counts["Y"]) / n
    helix_prop = sum(CHOU_FASMAN_HELIX[aa] * cnt for aa, cnt in counts.items()) / n
    sheet_prop = sum(CHOU_FASMAN_SHEET[aa] * cnt for aa, cnt in counts.items()) / n
    cationic_frac = (counts["K"] + counts["R"] + counts["H"]) / n
    hemolysis = compute_hemolysis_risk_heuristic(gravy, mu_alpha, charge_74, n)

    ph_curve = [
        {
            "ph": float(ph),
            "net_charge": round(
                net_charge_at_ph(counts, float(ph), n_term_aa=nterm_aa, c_term_aa=cterm_aa),
                4,
            ),
        }
        for ph in range(2, 13)
    ]

    return BiophysicalProfile(
        sequence=seq,
        length=n,
        molecular_weight=round(mw, 4),
        net_charge_ph74=round(charge_74, 4),
        charge_density_ph74=round(charge_74 / n, 4),
        isoelectric_point=round(pi_val, 4),
        gravy_kd=round(gravy, 4),
        mean_eisenberg_h=round(mean_eis, 4),
        hydrophobic_moment_alpha=round(mu_alpha, 4),
        max_window_moment_alpha=round(max_win_alpha, 4),
        hydrophobic_moment_beta=round(mu_beta, 4),
        boman_index=round(boman, 4),
        instability_index=round(instability, 4),
        aliphatic_index=round(aliphatic, 4),
        aromaticity=round(aromaticity, 4),
        helix_propensity=round(helix_prop, 4),
        sheet_propensity=round(sheet_prop, 4),
        cationic_fraction=round(cationic_frac, 4),
        hemolysis_risk_heuristic=round(hemolysis, 4),
        ph_titration_curve=ph_curve,
    )


def compute_aac(seq: str) -> np.ndarray:
    """Compute 20-dimensional normalized amino acid composition vector."""
    vec = np.zeros(20, dtype=np.float32)
    n = len(seq)
    if n == 0:
        return vec
    for ch in seq:
        idx = _AA_FLAT_IDX.get(ch)
        if idx is not None:
            vec[idx] += 1.0
    vec /= float(n)
    return vec


def compute_dpc(seq: str) -> np.ndarray:
    """Compute 400-dimensional normalized dipeptide composition vector."""
    vec = np.zeros(400, dtype=np.float32)
    n = len(seq)
    if n < 2:
        return vec
    for i in range(n - 1):
        a = _AA_FLAT_IDX.get(seq[i])
        b = _AA_FLAT_IDX.get(seq[i + 1])
        if a is not None and b is not None:
            vec[a * 20 + b] += 1.0
    vec /= float(n - 1)
    return vec


def compute_pseaac(seq: str, lam: int = 5, weight: float = 0.05) -> np.ndarray:
    """Compute Chou (2001) Type-1 Pseudo Amino Acid Composition (20 + lam dimensions)."""
    aac = compute_aac(seq).astype(np.float64)
    n = len(seq)
    thetas = np.zeros(lam, dtype=np.float64)
    for k in range(1, lam + 1):
        if n <= k:
            thetas[k - 1] = 0.0
            continue
        acc = 0.0
        for i in range(n - k):
            r1, r2 = seq[i], seq[i + k]
            d_h = (_PSE_HYDROPHOBICITY[r1] - _PSE_HYDROPHOBICITY[r2]) ** 2
            d_p = (_PSE_HYDROPHILICITY[r1] - _PSE_HYDROPHILICITY[r2]) ** 2
            d_m = (_PSE_MASS[r1] - _PSE_MASS[r2]) ** 2
            acc += (d_h + d_p + d_m) / 3.0
        thetas[k - 1] = acc / (n - k)

    denom = float(aac.sum() + weight * thetas.sum()) + 1e-12
    head = aac / denom
    tail = (weight * thetas) / denom
    return np.concatenate([head, tail]).astype(np.float32)


def extract_feature_matrix(
    sequences: Sequence[str],
    include_dpc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned biophysical matrix (N, 17) and classical ML matrix (N, 462 or 62)."""
    n_seqs = len(sequences)
    bio_mat = np.zeros((n_seqs, len(BIOPHYSICAL_FEATURE_NAMES)), dtype=np.float32)
    classic_parts: list[np.ndarray] = []

    for i, seq in enumerate(sequences):
        profile = compute_biophysical_profile(seq)
        bio_vec = profile.to_feature_vector()
        bio_mat[i] = bio_vec

        aac = compute_aac(profile.sequence)
        pseaac = compute_pseaac(profile.sequence, lam=5, weight=0.05)
        if include_dpc:
            dpc = compute_dpc(profile.sequence)
            classic_parts.append(np.concatenate([aac, dpc, pseaac, bio_vec]))
        else:
            classic_parts.append(np.concatenate([aac, pseaac, bio_vec]))

    return bio_mat, np.vstack(classic_parts).astype(np.float32)


def sequence_to_indices(sequence: str, max_len: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Encode sequence into padded integer indices and boolean valid-residue mask."""
    seq = sanitize_sequence(sequence, strict=False)
    tokens = np.full(max_len, PAD_IDX, dtype=np.int64)
    mask = np.zeros(max_len, dtype=np.bool_)
    truncated = seq[:max_len]
    for i, ch in enumerate(truncated):
        tokens[i] = AA_TO_IDX.get(ch, UNK_IDX)
        mask[i] = True
    return tokens, mask


def sequence_to_residue_channels(sequence: str, max_len: int = 128) -> np.ndarray:
    """Return (max_len, 7) normalized residue physicochemical channel matrix."""
    tokens, _ = sequence_to_indices(sequence, max_len=max_len)
    return RESIDUE_CHANNEL_TABLE[tokens]


def batch_encode_sequences(
    sequences: Iterable[str],
    max_len: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a collection of sequences into (tokens, channels, masks)."""
    seq_list = list(sequences)
    n = len(seq_list)
    tokens = np.zeros((n, max_len), dtype=np.int64)
    channels = np.zeros((n, max_len, 7), dtype=np.float32)
    masks = np.zeros((n, max_len), dtype=np.bool_)
    for i, seq in enumerate(seq_list):
        idx, m = sequence_to_indices(seq, max_len=max_len)
        tokens[i] = idx
        masks[i] = m
        channels[i] = RESIDUE_CHANNEL_TABLE[idx]
    return tokens, channels, masks
