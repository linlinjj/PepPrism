"""Explainability and motif discovery modules for PepPrism."""

from pepprism.explain.attribution import (
    PeptideExplanation,
    ResidueAttribution,
    explain_peptide,
    mine_high_attention_motifs,
)

__all__ = [
    "PeptideExplanation",
    "ResidueAttribution",
    "explain_peptide",
    "mine_high_attention_motifs",
]
