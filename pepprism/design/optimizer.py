"""Constrained in silico directed evolution and Pareto multi-objective peptide optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pepprism.biophysics.descriptors import compute_biophysical_profile, sanitize_sequence
from pepprism.biophysics.scales import STANDARD_AAS
from pepprism.models.trainer import TrainedPepPrismBundle


@dataclass(frozen=True)
class MutationCandidate:
    """Single candidate mutant with predicted activity and biophysical constraints."""

    sequence: str
    mutations: str
    edit_distance: int
    probability: float
    delta_probability: float
    net_charge_ph74: float
    hydrophobic_moment_alpha: float
    gravy_kd: float
    instability_index: float
    boman_index: float
    hemolysis_risk_heuristic: float
    constraints_satisfied: bool
    pareto_optimal: bool
    composite_score: float


@dataclass(frozen=True)
class OptimizationReport:
    """Complete optimization results including landscape matrix and Pareto candidates."""

    parent_sequence: str
    parent_probability: float
    parent_hemolysis_risk: float
    parent_charge_ph74: float
    parent_moment_alpha: float
    candidates: list[MutationCandidate]
    single_scan_heatmap: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _score_candidate(
    prob: float,
    hemolysis: float,
    mu_alpha: float,
    charge: float,
    instability: float,
    charge_bounds: tuple[float, float],
    max_instability: float,
) -> tuple[bool, float]:
    c_min, c_max = charge_bounds
    penalty = 0.0
    ok = True
    if charge < c_min:
        penalty += 0.15 * (c_min - charge)
        ok = False
    elif charge > c_max:
        penalty += 0.15 * (charge - c_max)
        ok = False
    if instability > max_instability:
        penalty += 0.01 * (instability - max_instability)
        ok = False

    score = prob - 0.35 * hemolysis + 0.15 * min(mu_alpha, 0.85) - penalty
    return ok, float(score)


def _mark_pareto_front(items: list[dict[str, object]]) -> set[int]:
    """Identify non-dominated indices maximizing probability and minimizing hemolysis."""
    pareto_idx: set[int] = set()
    for i, a in enumerate(items):
        dominated = False
        for j, b in enumerate(items):
            if i == j:
                continue
            if (
                float(b["probability"]) >= float(a["probability"])
                and float(b["hemolysis_risk_heuristic"]) <= float(a["hemolysis_risk_heuristic"])
                and (
                    float(b["probability"]) > float(a["probability"])
                    or float(b["hemolysis_risk_heuristic"]) < float(a["hemolysis_risk_heuristic"])
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto_idx.add(i)
    return pareto_idx


def optimize_peptide(
    bundle: TrainedPepPrismBundle,
    sequence: str,
    max_mutations: int = 2,
    top_k: int = 12,
    beam_width: int = 12,
    charge_bounds: tuple[float, float] = (2.0, 9.0),
    max_instability: float = 40.0,
) -> OptimizationReport:
    """Perform constrained in silico directed mutagenesis (1-point and optional 2-point)."""
    parent = sanitize_sequence(sequence, strict=True)
    parent_profile = compute_biophysical_profile(parent)
    parent_prob = float(bundle.predict_sequences([parent], mc_samples=1)["probabilities"][0])

    # Stage 1: Full single-point saturation mutagenesis scan
    single_mutants: list[tuple[str, str, int, str]] = []
    for pos, orig in enumerate(parent):
        for aa in STANDARD_AAS:
            if aa == orig:
                continue
            mut_seq = parent[:pos] + aa + parent[pos + 1 :]
            mut_label = f"{orig}{pos + 1}{aa}"
            single_mutants.append((mut_seq, mut_label, pos + 1, aa))

    mut_seqs = [m[0] for m in single_mutants]
    single_probs = bundle.predict_sequences(mut_seqs, mc_samples=1)["probabilities"]

    heatmap_entries: list[dict[str, object]] = []
    evaluated: dict[str, dict[str, object]] = {}

    for (mut_seq, mut_label, pos, aa), prob in zip(single_mutants, single_probs):
        p_val = float(prob)
        heatmap_entries.append(
            {
                "position": pos,
                "orig_aa": parent[pos - 1],
                "mut_aa": aa,
                "mutation": mut_label,
                "probability": round(p_val, 4),
                "delta_probability": round(p_val - parent_prob, 4),
            }
        )
        prof = compute_biophysical_profile(mut_seq)
        ok, score = _score_candidate(
            p_val,
            prof.hemolysis_risk_heuristic,
            prof.hydrophobic_moment_alpha,
            prof.net_charge_ph74,
            prof.instability_index,
            charge_bounds,
            max_instability,
        )
        evaluated[mut_seq] = {
            "sequence": mut_seq,
            "mutations": mut_label,
            "edit_distance": 1,
            "probability": p_val,
            "delta_probability": p_val - parent_prob,
            "net_charge_ph74": prof.net_charge_ph74,
            "hydrophobic_moment_alpha": prof.hydrophobic_moment_alpha,
            "gravy_kd": prof.gravy_kd,
            "instability_index": prof.instability_index,
            "boman_index": prof.boman_index,
            "hemolysis_risk_heuristic": prof.hemolysis_risk_heuristic,
            "constraints_satisfied": ok,
            "composite_score": score,
        }

    # Stage 2: Constrained 2-point beam expansion around top 1-point variants
    if max_mutations >= 2 and len(parent) >= 2:
        top_seeds = sorted(
            evaluated.values(),
            key=lambda item: float(item["composite_score"]),
            reverse=True,
        )[:beam_width]

        double_candidates: dict[str, str] = {}
        for seed in top_seeds:
            seed_seq = str(seed["sequence"])
            seed_mut = str(seed["mutations"])
            first_pos = int("".join(ch for ch in seed_mut if ch.isdigit())) - 1
            for pos, orig in enumerate(seed_seq):
                if pos == first_pos:
                    continue
                for aa in STANDARD_AAS:
                    if aa == orig:
                        continue
                    cand_seq = seed_seq[:pos] + aa + seed_seq[pos + 1 :]
                    if cand_seq == parent or cand_seq in evaluated or cand_seq in double_candidates:
                        continue
                    double_candidates[cand_seq] = f"{seed_mut}+{orig}{pos + 1}{aa}"

        if double_candidates:
            d_seqs = list(double_candidates.keys())
            d_probs = bundle.predict_sequences(d_seqs, mc_samples=1)["probabilities"]
            # Score top 60 probabilities thoroughly with full biophysical profiles
            top_d_idx = sorted(range(len(d_seqs)), key=lambda i: float(d_probs[i]), reverse=True)[
                :60
            ]
            for idx in top_d_idx:
                cand_seq = d_seqs[idx]
                p_val = float(d_probs[idx])
                prof = compute_biophysical_profile(cand_seq)
                ok, score = _score_candidate(
                    p_val,
                    prof.hemolysis_risk_heuristic,
                    prof.hydrophobic_moment_alpha,
                    prof.net_charge_ph74,
                    prof.instability_index,
                    charge_bounds,
                    max_instability,
                )
                evaluated[cand_seq] = {
                    "sequence": cand_seq,
                    "mutations": double_candidates[cand_seq],
                    "edit_distance": 2,
                    "probability": p_val,
                    "delta_probability": p_val - parent_prob,
                    "net_charge_ph74": prof.net_charge_ph74,
                    "hydrophobic_moment_alpha": prof.hydrophobic_moment_alpha,
                    "gravy_kd": prof.gravy_kd,
                    "instability_index": prof.instability_index,
                    "boman_index": prof.boman_index,
                    "hemolysis_risk_heuristic": prof.hemolysis_risk_heuristic,
                    "constraints_satisfied": ok,
                    "composite_score": score,
                }

    ranked_raw = sorted(
        evaluated.values(),
        key=lambda item: (bool(item["constraints_satisfied"]), float(item["composite_score"])),
        reverse=True,
    )
    feasible_pool = [r for r in ranked_raw if r["constraints_satisfied"]] or ranked_raw
    pareto_set = _mark_pareto_front(feasible_pool[:80])

    top_candidates: list[MutationCandidate] = []
    # Include Pareto-optimal feasible candidates first, then top composite score
    selected_indices: list[int] = []
    for idx in sorted(pareto_set):
        selected_indices.append(idx)
    for idx in range(len(feasible_pool)):
        if idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= top_k:
            break

    for idx in selected_indices[:top_k]:
        r = feasible_pool[idx]
        top_candidates.append(
            MutationCandidate(
                sequence=str(r["sequence"]),
                mutations=str(r["mutations"]),
                edit_distance=int(r["edit_distance"]),
                probability=round(float(r["probability"]), 4),
                delta_probability=round(float(r["delta_probability"]), 4),
                net_charge_ph74=round(float(r["net_charge_ph74"]), 3),
                hydrophobic_moment_alpha=round(float(r["hydrophobic_moment_alpha"]), 4),
                gravy_kd=round(float(r["gravy_kd"]), 4),
                instability_index=round(float(r["instability_index"]), 3),
                boman_index=round(float(r["boman_index"]), 4),
                hemolysis_risk_heuristic=round(float(r["hemolysis_risk_heuristic"]), 4),
                constraints_satisfied=bool(r["constraints_satisfied"]),
                pareto_optimal=(idx in pareto_set),
                composite_score=round(float(r["composite_score"]), 4),
            )
        )

    return OptimizationReport(
        parent_sequence=parent,
        parent_probability=round(parent_prob, 4),
        parent_hemolysis_risk=parent_profile.hemolysis_risk_heuristic,
        parent_charge_ph74=parent_profile.net_charge_ph74,
        parent_moment_alpha=parent_profile.hydrophobic_moment_alpha,
        candidates=top_candidates,
        single_scan_heatmap=heatmap_entries,
    )
