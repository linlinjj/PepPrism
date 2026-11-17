"""Public therapeutic peptide benchmark loader, parser, and anti-leakage auditor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import requests

from pepprism.biophysics.descriptors import is_valid_standard_sequence


@dataclass(frozen=True)
class BenchmarkSpec:
    """URL and task specification for a public peptide benchmark."""

    name: str
    task: str
    description: str
    train_pos_urls: tuple[str, ...]
    train_neg_urls: tuple[str, ...]
    test_pos_urls: tuple[str, ...]
    test_neg_urls: tuple[str, ...]


BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {
    "amplify_balanced": BenchmarkSpec(
        name="amplify_balanced",
        task="AMP",
        description="BirolLab/AMPlify balanced AMP vs non-AMP benchmark",
        train_pos_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_AMP_train_common.fa",
        ),
        train_neg_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_non_AMP_train_balanced.fa",
        ),
        test_pos_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_AMP_test_common.fa",
        ),
        test_neg_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_non_AMP_test_balanced.fa",
        ),
    ),
    "amplify_imbalanced": BenchmarkSpec(
        name="amplify_imbalanced",
        task="AMP",
        description="BirolLab/AMPlify imbalanced proteome-screening AMP benchmark",
        train_pos_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_AMP_train_common.fa",
        ),
        train_neg_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_non_AMP_train_imbalanced.fa",
        ),
        test_pos_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_AMP_test_common.fa",
        ),
        test_neg_urls=(
            "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data/AMPlify_non_AMP_test_imbalanced.fa",
        ),
    ),
    "ampscanner_v2": BenchmarkSpec(
        name="ampscanner_v2",
        task="AMP",
        description="Veltri et al. (2018) AMPScanner v2 original AMP/DECOY benchmark",
        train_pos_urls=(
            "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/master/original-dataset/AMP.tr.fa",
            "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/master/original-dataset/AMP.eval.fa",
        ),
        train_neg_urls=(
            "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/master/original-dataset/DECOY.tr.fa",
            "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/master/original-dataset/DECOY.eval.fa",
        ),
        test_pos_urls=(
            "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/master/original-dataset/AMP.te.fa",
        ),
        test_neg_urls=(
            "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/master/original-dataset/DECOY.te.fa",
        ),
    ),
    "anticp2_main": BenchmarkSpec(
        name="anticp2_main",
        task="ACP",
        description="Agrawal et al. (2021) AntiCP 2.0 Main dataset (ACP vs AMP hard negatives)",
        train_pos_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/pos_train_main",
        ),
        train_neg_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/neg_train_main",
        ),
        test_pos_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/pos_test_main",
        ),
        test_neg_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/neg_test_main",
        ),
    ),
    "anticp2_alternate": BenchmarkSpec(
        name="anticp2_alternate",
        task="ACP",
        description="Agrawal et al. (2021) AntiCP 2.0 Alternate dataset (ACP vs random peptides)",
        train_pos_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/pos_train_alternate",
        ),
        train_neg_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/neg_train_alternate",
        ),
        test_pos_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/pos_test_alternate",
        ),
        test_neg_urls=(
            "https://raw.githubusercontent.com/raghavagps/anticp2/master/datasets/neg_test_alternate",
        ),
    ),
}


@dataclass(frozen=True)
class DatasetAudit:
    """Data hygiene, deduplication, and length distribution audit metrics."""

    name: str
    task: str
    raw_train_pos: int
    raw_train_neg: int
    raw_test_pos: int
    raw_test_neg: int
    clean_train_pos: int
    clean_train_neg: int
    clean_test_pos: int
    clean_test_neg: int
    exact_leak_removed: int
    near_dup_removed: int
    train_pos_mean_len: float
    train_neg_mean_len: float
    test_pos_mean_len: float
    test_neg_mean_len: float
    length_ks_stat: float
    length_ks_pvalue: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PeptideDatasetSplit:
    """Cleaned, leak-free train/test split for a peptide benchmark."""

    name: str
    task: str
    train_sequences: list[str]
    train_labels: np.ndarray
    test_sequences: list[str]
    test_labels: np.ndarray
    audit: DatasetAudit


def parse_fasta_or_lines(text: str) -> list[str]:
    """Parse either FASTA format or plain one-sequence-per-line text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    has_fasta_headers = any(ln.startswith(">") for ln in lines)
    if not has_fasta_headers:
        return ["".join(ln.split()).upper() for ln in lines]

    sequences: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if ln.startswith(">"):
            if buf:
                sequences.append("".join(buf).upper())
                buf.clear()
        else:
            buf.append("".join(ln.split()))
    if buf:
        sequences.append("".join(buf).upper())
    return sequences


def _download_or_load(url: str, cache_dir: Path, timeout: int = 30) -> str:
    """Fetch a remote sequence file with deterministic local caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "__".join(url.split("/")[-3:])
    local_path = cache_dir / safe_name
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path.read_text(encoding="utf-8", errors="ignore")

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    local_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _kmer_set(seq: str, k: int = 4) -> frozenset[str]:
    if len(seq) < k:
        return frozenset({seq})
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a) + len(b) - inter)


def _filter_and_deduplicate(
    sequences: Iterable[str],
    min_len: int = 5,
    max_len: int = 200,
) -> list[str]:
    """Keep valid standard-AA sequences within length bounds, preserving first-seen order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for seq in sequences:
        s = "".join(seq.strip().split()).upper()
        if is_valid_standard_sequence(s, min_len=min_len, max_len=max_len) and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def audit_length_bias(
    pos_sequences: Iterable[str],
    neg_sequences: Iterable[str],
) -> tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov test comparing positive vs negative length."""
    from scipy.stats import ks_2samp

    pos_lens = np.array([len(s) for s in pos_sequences], dtype=np.float64)
    neg_lens = np.array([len(s) for s in neg_sequences], dtype=np.float64)
    if len(pos_lens) == 0 or len(neg_lens) == 0:
        return 0.0, 1.0
    res = ks_2samp(pos_lens, neg_lens)
    return float(res.statistic), float(res.pvalue)


def remove_cross_split_leakage(
    train_seqs: list[str],
    test_seqs: list[str],
    jaccard_threshold: float = 0.85,
) -> tuple[list[str], int, int]:
    """Remove exact matches and near-duplicates via exact prefix-filtered 5-mer index."""
    from collections import defaultdict
    import math

    train_set = set(train_seqs)
    k = 5
    train_kmers: list[frozenset[str]] = []
    train_sizes: list[int] = []
    kmer_index: dict[str, list[int]] = defaultdict(list)

    if jaccard_threshold < 1.0:
        for idx, seq in enumerate(train_seqs):
            km = _kmer_set(seq, k=k)
            train_kmers.append(km)
            train_sizes.append(len(km))
            for token in km:
                kmer_index[token].append(idx)

    kept: list[str] = []
    exact_removed = 0
    near_removed = 0

    for seq in test_seqs:
        if seq in train_set:
            exact_removed += 1
            continue
        is_near = False
        if jaccard_threshold < 1.0:
            t_km = _kmer_set(seq, k=k)
            t_size = len(t_km)
            min_overlap = max(1, math.ceil(jaccard_threshold * t_size))
            max_tr_size = int(math.floor(t_size / jaccard_threshold))
            min_tr_size = min_overlap

            prefix_len = t_size - min_overlap + 1
            sorted_tokens = sorted(t_km, key=lambda tok: len(kmer_index.get(tok, ())))
            candidates: set[int] = set()
            for token in sorted_tokens[:prefix_len]:
                for tr_idx in kmer_index.get(token, ()):
                    tr_size = train_sizes[tr_idx]
                    if min_tr_size <= tr_size <= max_tr_size:
                        candidates.add(tr_idx)

            for tr_idx in candidates:
                tr_km = train_kmers[tr_idx]
                inter = len(t_km & tr_km)
                if inter / float(t_size + train_sizes[tr_idx] - inter) >= jaccard_threshold:
                    is_near = True
                    break
        if is_near:
            near_removed += 1
        else:
            kept.append(seq)

    return kept, exact_removed, near_removed


def create_length_matched_subset(
    sequences: list[str],
    labels: np.ndarray,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Subsample negative sequences to match positive sequence length distribution."""
    rng = random.Random(seed)
    pos_indices = [i for i, y in enumerate(labels) if int(y) == 1]
    neg_indices = [i for i, y in enumerate(labels) if int(y) == 0]
    if not pos_indices or not neg_indices:
        return sequences, labels

    used_neg: set[int] = set()
    neg_by_len: dict[int, list[int]] = {}
    for idx in neg_indices:
        neg_by_len.setdefault(len(sequences[idx]), []).append(idx)
    for lst in neg_by_len.values():
        rng.shuffle(lst)

    matched_negs: list[int] = []
    for p_idx in pos_indices:
        target_len = len(sequences[p_idx])
        found = None
        # Search outward in length tolerance +/- 5 residues
        for delta in range(0, 12):
            for candidate_len in (target_len - delta, target_len + delta):
                pool = neg_by_len.get(candidate_len)
                if pool:
                    while pool:
                        cand = pool.pop()
                        if cand not in used_neg:
                            found = cand
                            break
                if found is not None:
                    break
            if found is not None:
                break
        if found is not None:
            used_neg.add(found)
            matched_negs.append(found)

    chosen = pos_indices + matched_negs
    rng.shuffle(chosen)
    sub_seqs = [sequences[i] for i in chosen]
    sub_labels = np.array([int(labels[i]) for i in chosen], dtype=np.int64)
    return sub_seqs, sub_labels


def load_benchmark_dataset(
    name: str,
    cache_dir: Path | str = Path("data/raw"),
    min_len: int = 5,
    max_len: int = 200,
    jaccard_threshold: float = 0.85,
) -> PeptideDatasetSplit:
    """Download, clean, deduplicate, and audit a registered benchmark dataset."""
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_REGISTRY)}")

    spec = BENCHMARK_REGISTRY[name]
    cache = Path(cache_dir) / name

    def _fetch_many(urls: tuple[str, ...]) -> list[str]:
        all_seqs: list[str] = []
        for url in urls:
            raw_text = _download_or_load(url, cache)
            all_seqs.extend(parse_fasta_or_lines(raw_text))
        return all_seqs

    raw_tr_pos = _fetch_many(spec.train_pos_urls)
    raw_tr_neg = _fetch_many(spec.train_neg_urls)
    raw_te_pos = _fetch_many(spec.test_pos_urls)
    raw_te_neg = _fetch_many(spec.test_neg_urls)

    tr_pos = _filter_and_deduplicate(raw_tr_pos, min_len=min_len, max_len=max_len)
    tr_neg = _filter_and_deduplicate(raw_tr_neg, min_len=min_len, max_len=max_len)
    # Exclude ambiguous sequences appearing in both positive and negative train sets
    pos_train_set = set(tr_pos)
    tr_neg = [s for s in tr_neg if s not in pos_train_set]

    te_pos = _filter_and_deduplicate(raw_te_pos, min_len=min_len, max_len=max_len)
    te_neg = _filter_and_deduplicate(raw_te_neg, min_len=min_len, max_len=max_len)
    pos_test_set = set(te_pos)
    te_neg = [s for s in te_neg if s not in pos_test_set]

    all_train = tr_pos + tr_neg
    te_pos_clean, ex_p, nr_p = remove_cross_split_leakage(
        all_train, te_pos, jaccard_threshold=jaccard_threshold
    )
    te_neg_clean, ex_n, nr_n = remove_cross_split_leakage(
        all_train, te_neg, jaccard_threshold=jaccard_threshold
    )

    train_seqs = tr_pos + tr_neg
    train_labels = np.array([1] * len(tr_pos) + [0] * len(tr_neg), dtype=np.int64)
    test_seqs = te_pos_clean + te_neg_clean
    test_labels = np.array([1] * len(te_pos_clean) + [0] * len(te_neg_clean), dtype=np.int64)

    ks_stat, ks_p = audit_length_bias(tr_pos, tr_neg)

    audit = DatasetAudit(
        name=spec.name,
        task=spec.task,
        raw_train_pos=len(raw_tr_pos),
        raw_train_neg=len(raw_tr_neg),
        raw_test_pos=len(raw_te_pos),
        raw_test_neg=len(raw_te_neg),
        clean_train_pos=len(tr_pos),
        clean_train_neg=len(tr_neg),
        clean_test_pos=len(te_pos_clean),
        clean_test_neg=len(te_neg_clean),
        exact_leak_removed=ex_p + ex_n,
        near_dup_removed=nr_p + nr_n,
        train_pos_mean_len=round(float(np.mean([len(s) for s in tr_pos])), 2),
        train_neg_mean_len=round(float(np.mean([len(s) for s in tr_neg])), 2),
        test_pos_mean_len=round(float(np.mean([len(s) for s in te_pos_clean])), 2)
        if te_pos_clean
        else 0.0,
        test_neg_mean_len=round(float(np.mean([len(s) for s in te_neg_clean])), 2)
        if te_neg_clean
        else 0.0,
        length_ks_stat=round(ks_stat, 4),
        length_ks_pvalue=float(ks_p),
    )

    return PeptideDatasetSplit(
        name=spec.name,
        task=spec.task,
        train_sequences=train_seqs,
        train_labels=train_labels,
        test_sequences=test_seqs,
        test_labels=test_labels,
        audit=audit,
    )
