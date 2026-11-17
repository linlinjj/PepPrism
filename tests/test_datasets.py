"""Unit tests for sequence parsing, anti-leakage filtering, and length-matched sampling."""

import numpy as np

from pepprism.data.datasets import (
    audit_length_bias,
    create_length_matched_subset,
    parse_fasta_or_lines,
    remove_cross_split_leakage,
)


def test_parse_fasta_and_plain_lines() -> None:
    fasta = ">pep1\nGIGKFLHSAKK\nFGKAFV\n>pep2\nILPWKWPWWPWRR\n"
    assert parse_fasta_or_lines(fasta) == ["GIGKFLHSAKKFGKAFV", "ILPWKWPWWPWRR"]

    plain = "FAKKFAKKFKK\nglfdlvkkvv\n"
    assert parse_fasta_or_lines(plain) == ["FAKKFAKKFKK", "GLFDLVKKVV"]


def test_remove_cross_split_leakage() -> None:
    train = ["GIGKFLHSAKKFGKAFVGEIMNS", "ILPWKWPWWPWRR"]
    test = [
        "GIGKFLHSAKKFGKAFVGEIMNS",      # exact duplicate
        "GIGKFLHSAKKFGKAFVGEIMNA",      # 1-aa near duplicate (high 4-mer Jaccard)
        "KWKLFKKIEKVGQNIRDGIIKAGPAV",   # genuinely distinct
    ]
    kept, exact_rm, near_rm = remove_cross_split_leakage(train, test, jaccard_threshold=0.80)
    assert kept == ["KWKLFKKIEKVGQNIRDGIIKAGPAV"]
    assert exact_rm == 1
    assert near_rm == 1


def test_length_matched_subset_reduces_bias() -> None:
    pos = ["GIGKFLHSAKKFGKAFVG", "LLGDFFRKSKEKIGKEFK", "KWKLFKKIEKVGQNIRDG"]
    neg = [
        "AAAAAAAA",
        "CCCCCCCC",
        "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
        "EEEEEEEEEEEEEEEEEE",
        "FFFFFFFFFFFFFFFFFF",
        "GGGGGGGGGGGGGGGGGG",
    ]
    seqs = pos + neg
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    sub_s, sub_y = create_length_matched_subset(seqs, labels, seed=42)
    assert (sub_y == 1).sum() == 3
    assert (sub_y == 0).sum() == 3
    pos_sub = [s for s, y in zip(sub_s, sub_y) if y == 1]
    neg_sub = [s for s, y in zip(sub_s, sub_y) if y == 0]
    ks_stat, _ = audit_length_bias(pos_sub, neg_sub)
    assert ks_stat == 0.0
