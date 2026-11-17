"""Reproducible multi-benchmark training, ablation, bias audit, and figure pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from pepprism.biophysics.descriptors import extract_feature_matrix
from pepprism.data.datasets import (
    create_length_matched_subset,
    load_benchmark_dataset,
)
from pepprism.design.optimizer import optimize_peptide
from pepprism.explain.attribution import explain_peptide, mine_high_attention_motifs
from pepprism.models.baselines import CLASSICAL_MODEL_NAMES, train_evaluate_classical_baseline
from pepprism.models.trainer import (
    compute_classification_metrics,
    train_deep_model,
)
from pepprism.viz.plots import (
    plot_ablation_and_calibration,
    plot_dataset_length_audit,
    plot_interpretability_case,
    plot_optimization_pareto,
    plot_roc_pr_curves,
    plot_system_architecture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
ASSETS_DIR = REPO_ROOT / "assets"
CKPT_DIR = REPO_ROOT / "checkpoints"
DATA_CACHE = REPO_ROOT / "data" / "raw"

BENCHMARK_NAMES: tuple[str, ...] = (
    "amplify_balanced",
    "amplify_imbalanced",
    "ampscanner_v2",
    "anticp2_main",
    "anticp2_alternate",
)


def verify_existing_outputs() -> None:
    """Verify all expected result JSONs, checkpoints, and figures exist and contain valid data."""
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("audits", "main_benchmarks", "five_fold_cv", "ablation", "length_bias_test", "motifs", "cases"):
        if key not in summary or not summary[key]:
            raise ValueError(f"Missing or empty section '{key}' in {summary_path}")

    for ckpt in (CKPT_DIR / "pepprism_amp.pt", CKPT_DIR / "pepprism_acp.pt"):
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    expected_figs = (
        "fig1_system_architecture.png",
        "fig2_dataset_length_audit.png",
        "fig3_roc_pr_benchmarks.png",
        "fig4_ablation_calibration.png",
        "fig5_interpretability_case.png",
        "fig6_optimization_pareto.png",
    )
    for fig_name in expected_figs:
        fig_path = ASSETS_DIR / fig_name
        if not fig_path.exists() or fig_path.stat().st_size < 5000:
            raise FileNotFoundError(f"Missing or invalid figure: {fig_path}")
    print("[VERIFY OK] All benchmark JSONs, checkpoints, and 6 publication figures verified.")


def run_all_experiments(seed: int = 42, epochs: int = 24) -> dict[str, object]:
    """Execute complete reproducible pipeline across all datasets, ablations, and case studies."""
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & audit datasets
    print("=== [1/6] Loading and auditing public peptide benchmarks ===")
    splits = {}
    audits = {}
    audit_plot_data = []
    for name in BENCHMARK_NAMES:
        split = load_benchmark_dataset(name, cache_dir=DATA_CACHE)
        splits[name] = split
        audits[name] = split.audit.to_dict()
        print(
            f"  {name}: train={len(split.train_sequences)} (pos={split.audit.clean_train_pos}, "
            f"neg={split.audit.clean_train_neg}), test={len(split.test_sequences)} "
            f"(exact_leak_rm={split.audit.exact_leak_removed}, near_rm={split.audit.near_dup_removed}), "
            f"KS={split.audit.length_ks_stat:.3f}"
        )
        if name in ("amplify_balanced", "amplify_imbalanced", "ampscanner_v2", "anticp2_main"):
            pos_lens = [len(s) for s, y in zip(split.train_sequences, split.train_labels) if y == 1]
            neg_lens = [len(s) for s, y in zip(split.train_sequences, split.train_labels) if y == 0]
            audit_plot_data.append(
                {
                    "title": name,
                    "pos_lengths": pos_lens,
                    "neg_lengths": neg_lens,
                    "ks_stat": split.audit.length_ks_stat,
                }
            )

    plot_system_architecture(ASSETS_DIR / "fig1_system_architecture.png")
    plot_dataset_length_audit(audit_plot_data, ASSETS_DIR / "fig2_dataset_length_audit.png")

    # Extract cached feature matrices per dataset
    features = {}
    for name, split in splits.items():
        tr_bio, tr_cls = extract_feature_matrix(split.train_sequences, include_dpc=True)
        te_bio, te_cls = extract_feature_matrix(split.test_sequences, include_dpc=True)
        features[name] = {
            "tr_bio": tr_bio,
            "tr_cls": tr_cls,
            "te_bio": te_bio,
            "te_cls": te_cls,
        }

    # 2. Main Multi-Model Benchmark Comparison
    print("=== [2/6] Training & evaluating Classical ML + Deep baselines + PepPrismNet ===")
    main_benchmarks: dict[str, dict[str, dict[str, float]]] = {}
    roc_curves_for_plot: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    trained_amp_bundle = None
    trained_acp_bundle = None
    amp_test_models: dict[str, object] = {}

    for name in BENCHMARK_NAMES:
        split = splits[name]
        feat = features[name]
        main_benchmarks[name] = {}
        collect_curves = name in ("amplify_balanced", "amplify_imbalanced", "ampscanner_v2", "anticp2_main")
        if collect_curves:
            roc_curves_for_plot[name] = {}

        # Classical ML baselines
        for cls_name in CLASSICAL_MODEL_NAMES:
            cw = "balanced"
            pipe, probs = train_evaluate_classical_baseline(
                cls_name,
                feat["tr_cls"],
                split.train_labels,
                feat["te_cls"],
                seed=seed,
                class_weight=cw,
            )
            m = compute_classification_metrics(split.test_labels, probs)
            main_benchmarks[name][cls_name] = m.to_dict()
            if collect_curves:
                roc_curves_for_plot[name][cls_name] = (split.test_labels, probs)
            if name == "amplify_balanced":
                amp_test_models[cls_name] = pipe
            print(f"  [{name}] {cls_name:20s} AUROC={m.auroc:.4f} AUPRC={m.auprc:.4f} MCC={m.mcc:.4f}")

        # Deep models
        ds_epochs = 5 if len(split.train_sequences) > 30000 else epochs
        for deep_name in ("SeqCNN1D", "BiGRUAttention", "PepPrismNet"):
            t_m = time.time()
            bundle = train_deep_model(
                arch=deep_name,
                train_sequences=split.train_sequences,
                train_labels=split.train_labels,
                train_bio_raw=feat["tr_bio"],
                task=split.task,
                epochs=ds_epochs,
                seed=seed,
            )
            mc = 15 if deep_name == "PepPrismNet" else 1
            pred = bundle.predict_sequences(split.test_sequences, mc_samples=mc)
            probs = pred["probabilities"]
            m = compute_classification_metrics(split.test_labels, probs)
            main_benchmarks[name][deep_name] = m.to_dict()
            if collect_curves:
                roc_curves_for_plot[name][deep_name] = (split.test_labels, probs)
            if name == "amplify_balanced":
                amp_test_models[deep_name] = bundle
                if deep_name == "PepPrismNet":
                    trained_amp_bundle = bundle
                    bundle.save(CKPT_DIR / "pepprism_amp.pt")
            if name == "anticp2_main" and deep_name == "PepPrismNet":
                trained_acp_bundle = bundle
                bundle.save(CKPT_DIR / "pepprism_acp.pt")
            print(f"  [{name}] {deep_name:20s} AUROC={m.auroc:.4f} AUPRC={m.auprc:.4f} MCC={m.mcc:.4f} ECE={m.ece:.4f} ({time.time()-t_m:.1f}s)", flush=True)

    plot_roc_pr_curves(roc_curves_for_plot, ASSETS_DIR / "fig3_roc_pr_benchmarks.png")

    # 3. 5-Fold Stratified Cross-Validation on AMPlify Balanced training set
    print("=== [3/6] Running 5-Fold Stratified Cross-Validation on AMPlify Balanced ===")
    amp_split = splits["amplify_balanced"]
    amp_feat = features["amplify_balanced"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_models = ("RandomForest", "HistGradientBoosting", "BiGRUAttention", "PepPrismNet")
    cv_raw: dict[str, list[dict[str, float]]] = {m: [] for m in cv_models}

    for fold_idx, (tr_i, va_i) in enumerate(skf.split(amp_split.train_sequences, amp_split.train_labels), start=1):
        tr_seqs = [amp_split.train_sequences[i] for i in tr_i]
        va_seqs = [amp_split.train_sequences[i] for i in va_i]
        tr_y = amp_split.train_labels[tr_i]
        va_y = amp_split.train_labels[va_i]

        for cls_name in ("RandomForest", "HistGradientBoosting"):
            cw = "balanced" if cls_name == "RandomForest" else None
            _, probs = train_evaluate_classical_baseline(
                cls_name,
                amp_feat["tr_cls"][tr_i],
                tr_y,
                amp_feat["tr_cls"][va_i],
                seed=seed + fold_idx,
                class_weight=cw,
            )
            cv_raw[cls_name].append(compute_classification_metrics(va_y, probs).to_dict())

        for deep_name in ("BiGRUAttention", "PepPrismNet"):
            b = train_deep_model(
                arch=deep_name,
                train_sequences=tr_seqs,
                train_labels=tr_y,
                train_bio_raw=amp_feat["tr_bio"][tr_i],
                task="AMP",
                epochs=min(epochs, 18),
                seed=seed + fold_idx,
            )
            probs = b.predict_sequences(va_seqs, mc_samples=1)["probabilities"]
            cv_raw[deep_name].append(compute_classification_metrics(va_y, probs).to_dict())

    five_fold_cv: dict[str, dict[str, dict[str, float]]] = {}
    for m_name, fold_list in cv_raw.items():
        five_fold_cv[m_name] = {}
        for metric_key in ("accuracy", "f1", "mcc", "auroc", "auprc"):
            vals = [f[metric_key] for f in fold_list]
            five_fold_cv[m_name][metric_key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
        print(
            f"  [5-Fold CV] {m_name:20s} AUROC={five_fold_cv[m_name]['auroc']['mean']:.4f}±{five_fold_cv[m_name]['auroc']['std']:.4f}, "
            f"MCC={five_fold_cv[m_name]['mcc']['mean']:.4f}±{five_fold_cv[m_name]['mcc']['std']:.4f}"
        )

    # 4. Ablation & Calibration Study on AMPlify Balanced & AntiCP 2.0 Main
    print("=== [4/6] Running Architecture Ablation & Calibration Study ===")
    ablation_variants = (
        ("PepPrism_BioOnly", "仅理化分支 (BioOnly)"),
        ("PepPrism_SeqOnly", "仅序列分支 (SeqOnly)"),
        ("PepPrism_Concat", "双分支拼接 (Concat)"),
        ("PepPrismNet", "门控融合 (GBSF, 完整模型)"),
    )
    ablation_results: dict[str, list[dict[str, object]]] = {}
    cal_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ds_key in ("amplify_balanced", "anticp2_main"):
        sp = splits[ds_key]
        ft = features[ds_key]
        rows = []
        for arch_key, label_zh in ablation_variants:
            if ds_key == "amplify_balanced" and arch_key == "PepPrismNet" and trained_amp_bundle is not None:
                b = trained_amp_bundle
            elif ds_key == "anticp2_main" and arch_key == "PepPrismNet" and trained_acp_bundle is not None:
                b = trained_acp_bundle
            else:
                b = train_deep_model(
                    arch=arch_key,
                    train_sequences=sp.train_sequences,
                    train_labels=sp.train_labels,
                    train_bio_raw=ft["tr_bio"],
                    task=sp.task,
                    epochs=epochs,
                    seed=seed,
                )
            mc = 20 if arch_key == "PepPrismNet" else 1
            pred = b.predict_sequences(sp.test_sequences, mc_samples=mc)
            probs = pred["probabilities"]
            m = compute_classification_metrics(sp.test_labels, probs)
            row = {"arch": arch_key, "variant": label_zh, **m.to_dict()}
            rows.append(row)
            if ds_key == "amplify_balanced":
                cal_curves[arch_key] = (sp.test_labels, probs)
            print(f"  [Ablation {ds_key}] {arch_key:18s} AUROC={m.auroc:.4f} MCC={m.mcc:.4f} ECE={m.ece:.4f}")
        ablation_results[ds_key] = rows

    plot_ablation_and_calibration(
        ablation_results["amplify_balanced"],
        cal_curves,
        ASSETS_DIR / "fig4_ablation_calibration.png",
    )

    # 5. Negative-Sampling Length-Bias Robustness Evaluation
    print("=== [5/6] Running Length-Matched Negative Sampling Robustness Test ===")
    lm_seqs, lm_labels = create_length_matched_subset(
        amp_split.test_sequences, amp_split.test_labels, seed=seed
    )
    _, lm_cls_mat = extract_feature_matrix(lm_seqs, include_dpc=True)
    length_bias_test: dict[str, dict[str, float]] = {}
    for m_name, model_obj in amp_test_models.items():
        if m_name in CLASSICAL_MODEL_NAMES:
            probs = model_obj.predict_proba(lm_cls_mat)[:, 1]
        else:
            probs = model_obj.predict_sequences(lm_seqs, mc_samples=1)["probabilities"]
        m = compute_classification_metrics(lm_labels, probs)
        length_bias_test[m_name] = m.to_dict()
        print(f"  [Length-Matched Test] {m_name:20s} AUROC={m.auroc:.4f} MCC={m.mcc:.4f}")

    # 6. Biological Interpretability, Motif Mining & Constrained Optimization Cases
    print("=== [6/6] Running Explainability, Motif Mining & Pareto Optimization Cases ===")
    assert trained_amp_bundle is not None
    motifs = mine_high_attention_motifs(
        trained_amp_bundle,
        amp_split.train_sequences,
        amp_split.train_labels,
        k=4,
        top_n=12,
    )

    magainin2 = "GIGKFLHSAKKFGKAFVGEIMNS"
    melittin = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    ll37 = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"

    expl_mag = explain_peptide(trained_amp_bundle, magainin2, mc_samples=20)
    expl_ll37 = explain_peptide(trained_amp_bundle, ll37, mc_samples=20)
    plot_interpretability_case(
        expl_mag,
        "Magainin-2",
        ASSETS_DIR / "fig5_interpretability_case.png",
    )

    opt_melittin = optimize_peptide(
        trained_amp_bundle,
        melittin,
        max_mutations=2,
        top_k=12,
        beam_width=12,
        charge_bounds=(3.0, 9.0),
        max_instability=40.0,
    )
    opt_magainin = optimize_peptide(
        trained_amp_bundle,
        magainin2,
        max_mutations=2,
        top_k=12,
        beam_width=12,
        charge_bounds=(2.0, 9.0),
        max_instability=40.0,
    )
    plot_optimization_pareto(
        opt_melittin.parent_probability,
        opt_melittin.parent_hemolysis_risk,
        [vars(c) if hasattr(c, "__dict__") else c for c in opt_melittin.candidates],
        ASSETS_DIR / "fig6_optimization_pareto.png",
    )

    summary = {
        "runtime_seconds": round(time.time() - t0, 2),
        "seed": seed,
        "epochs": epochs,
        "audits": audits,
        "main_benchmarks": main_benchmarks,
        "five_fold_cv": five_fold_cv,
        "ablation": ablation_results,
        "length_bias_test": {
            "n_samples": len(lm_seqs),
            "metrics": length_bias_test,
        },
        "motifs": motifs,
        "cases": {
            "magainin2_explanation": expl_mag.to_dict(),
            "ll37_explanation": expl_ll37.to_dict(),
            "melittin_optimization": opt_melittin.to_dict(),
            "magainin2_optimization": opt_magainin.to_dict(),
        },
    }
    (RESULTS_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    verify_existing_outputs()
    print(f"All experiments completed in {summary['runtime_seconds']}s.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PepPrism reproducible scientific benchmarks.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing artifacts only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()

    if args.verify_only:
        verify_existing_outputs()
    else:
        run_all_experiments(seed=args.seed, epochs=args.epochs)


if __name__ == "__main__":
    main()
