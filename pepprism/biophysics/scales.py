"""Standard amino acid physicochemical scales and indexing tables."""

from __future__ import annotations

import numpy as np

STANDARD_AAS: str = "ACDEFGHIKLMNPQRSTVWY"
AA_SET: frozenset[str] = frozenset(STANDARD_AAS)

# Index 0 is reserved for padding; 1..20 map to STANDARD_AAS; 21 is reserved for unknown.
AA_TO_IDX: dict[str, int] = {aa: idx + 1 for idx, aa in enumerate(STANDARD_AAS)}
IDX_TO_AA: dict[int, str] = {idx + 1: aa for idx, aa in enumerate(STANDARD_AAS)}
PAD_IDX: int = 0
UNK_IDX: int = 21
VOCAB_SIZE: int = 22

# Kyte & Doolittle (1982) hydropathy scale
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Eisenberg et al. (1984) normalized consensus hydrophobicity scale
EISENBERG_HYDROPHOBICITY: dict[str, float] = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}

# Radzicka & Wolfenden (1988) cyclohexane-to-water transfer free energies (kcal/mol)
# Boman (2003) index = -sum(RADZICKA_WOLFENDEN[aa]) / length
RADZICKA_WOLFENDEN: dict[str, float] = {
    "A": 1.81, "C": 1.28, "D": -8.72, "E": -6.81, "F": 2.98,
    "G": 0.94, "H": -4.66, "I": 4.92, "K": -5.55, "L": 4.92,
    "M": 2.35, "N": -6.64, "P": 0.00, "Q": -5.54, "R": -14.92,
    "S": -3.40, "T": -2.57, "V": 4.04, "W": 2.33, "Y": -0.14,
}

# Hopp & Woods (1981) hydrophilicity scale (used in PseAAC)
HOPP_WOODS: dict[str, float] = {
    "A": -0.5, "C": -1.0, "D": 3.0, "E": 3.0, "F": -2.5,
    "G": 0.0, "H": -0.5, "I": -1.8, "K": 3.0, "L": -1.8,
    "M": -1.3, "N": 0.2, "P": 0.0, "Q": 0.2, "R": 3.0,
    "S": 0.3, "T": -0.4, "V": -1.5, "W": -3.4, "Y": -2.3,
}

# Chou & Fasman (1978) alpha-helix and beta-sheet conformational parameters
CHOU_FASMAN_HELIX: dict[str, float] = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}

CHOU_FASMAN_SHEET: dict[str, float] = {
    "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38,
    "G": 0.75, "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30,
    "M": 1.05, "N": 0.89, "P": 0.55, "Q": 1.10, "R": 0.93,
    "S": 0.75, "T": 1.19, "V": 1.70, "W": 1.37, "Y": 1.47,
}

# Grantham (1974) side-chain polarity and Zamyatnin (1972) residue molecular volume (A^3)
GRANTHAM_POLARITY: dict[str, float] = {
    "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2,
    "G": 9.0, "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9,
    "M": 5.7, "N": 11.6, "P": 8.0, "Q": 10.5, "R": 10.5,
    "S": 9.2, "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2,
}

RESIDUE_VOLUME: dict[str, float] = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

# Standard monoisotopic/average residue masses (Da, in-chain mass; +18.01528 Da for H2O)
RESIDUE_MASS: dict[str, float] = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}

# Bjellqvist et al. (1993, 1994) / ExPASy ProtParam pKa table (matching Biopython)
PKA_POS: dict[str, float] = {"N_term": 7.50, "K": 10.00, "R": 12.00, "H": 5.98}
PKA_NEG: dict[str, float] = {"C_term": 3.55, "D": 4.05, "E": 4.45, "C": 9.00, "Y": 10.00}
PKA_N_TERMINAL: dict[str, float] = {
    "A": 7.59, "M": 7.00, "S": 6.93, "P": 8.36, "T": 6.82, "V": 7.44, "E": 7.70,
}
PKA_C_TERMINAL: dict[str, float] = {"D": 4.55, "E": 4.75}

# Physicochemical group classification for visualization & helical wheel coloring
AA_GROUP: dict[str, str] = {
    "A": "hydrophobic", "V": "hydrophobic", "I": "hydrophobic", "L": "hydrophobic",
    "M": "hydrophobic", "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "K": "positive", "R": "positive", "H": "positive",
    "D": "negative", "E": "negative",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
    "C": "special", "G": "special", "P": "special",
}


def build_normalized_channel_table() -> np.ndarray:
    """Build (VOCAB_SIZE, 7) lookup table of standardized per-residue biophysical channels.

    Channels:
        0: Kyte-Doolittle hydropathy
        1: Eisenberg consensus hydrophobicity
        2: Nominal side-chain charge at pH 7.4 (+1 K/R, +0.11 H, -1 D/E)
        3: Chou-Fasman alpha-helix propensity
        4: Chou-Fasman beta-sheet propensity
        5: Grantham polarity
        6: Residue side-chain volume
    """
    raw = np.zeros((20, 7), dtype=np.float32)
    nominal_charge = {"K": 1.0, "R": 1.0, "H": 0.11, "D": -1.0, "E": -1.0}
    for i, aa in enumerate(STANDARD_AAS):
        raw[i, 0] = KYTE_DOOLITTLE[aa]
        raw[i, 1] = EISENBERG_HYDROPHOBICITY[aa]
        raw[i, 2] = nominal_charge.get(aa, 0.0)
        raw[i, 3] = CHOU_FASMAN_HELIX[aa]
        raw[i, 4] = CHOU_FASMAN_SHEET[aa]
        raw[i, 5] = GRANTHAM_POLARITY[aa]
        raw[i, 6] = RESIDUE_VOLUME[aa]

    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-6
    normed = (raw - mean) / std

    table = np.zeros((VOCAB_SIZE, 7), dtype=np.float32)
    table[1:21] = normed
    return table


RESIDUE_CHANNEL_TABLE: np.ndarray = build_normalized_channel_table()
