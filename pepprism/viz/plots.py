"""Publication-grade Matplotlib visualization generators for PepPrism."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from pepprism.biophysics.helical_wheel import compute_helical_wheel
from pepprism.explain.attribution import PeptideExplanation

# Configure Chinese + Latin academic typography cleanly on Windows/Linux
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

GROUP_COLORS: dict[str, str] = {
    "hydrophobic": "#eab308",
    "aromatic": "#f97316",
    "positive": "#2563eb",
    "negative": "#dc2626",
    "polar": "#10b981",
    "special": "#6b7280",
}


def _ensure_parent(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_system_architecture(out_path: Path | str) -> None:
    """Render clean dual-branch PepPrism system architecture diagram."""
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f8fafc", ec="#334155", title_color="#0f172a"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.14",
            facecolor=fc, edgecolor=ec, linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - 0.14, body, ha="center", va="center",
                fontsize=8.6, color="#334155", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.6))

    box(0.25, 1.85, 1.85, 1.6, "多肽序列输入", "FASTA / 单条序列\n标准氨基酸清洗\n抗泄漏去重校验", fc="#eff6ff", ec="#2563eb")
    box(2.65, 3.15, 2.75, 1.75, "分支 A：多尺度序列编码器", "残基嵌入 + 7维理化先验\n膨胀1D-CNN (d=1,2,4)\nSE通道注意 + 双池化归因", fc="#ecfeff", ec="#0891b2")
    box(2.65, 0.55, 2.75, 1.85, "分支 B：第一性原理生物物理", "Henderson-Hasselbalch净电荷\nEisenberg α/β 疏水矩 μH\nBoman / 不稳定指数 / GRAVY", fc="#fefce8", ec="#ca8a04")
    box(5.95, 1.85, 2.15, 1.65, "门控融合模块 (GBSF)", "跨模态动态门控 g=σ(W[h_s;h_b])\n模态权重自适应平衡\nMC-Dropout 不确定度估计", fc="#f5f3ff", ec="#7c3aed")
    box(8.65, 3.25, 2.1, 1.6, "可解释性归因引擎", "残基注意力分布 α_i\n输入梯度 + 丙氨酸扫描\n高注意 k-mer 基序挖掘", fc="#ecfdf5", ec="#059669")
    box(8.65, 0.7, 2.1, 1.75, "约束定向优化器", "1/2点饱和突变扫描\n电荷/稳定性/溶血风险约束\nPareto 前沿筛选与螺旋轮", fc="#fff1f2", ec="#e11d48")

    arrow(2.15, 2.95, 2.62, 3.95)
    arrow(2.15, 2.35, 2.62, 1.50)
    arrow(5.45, 3.95, 5.92, 2.95)
    arrow(5.45, 1.50, 5.92, 2.40)
    arrow(8.15, 3.00, 8.62, 4.00)
    arrow(8.15, 2.35, 8.62, 1.60)
    arrow(9.70, 3.22, 9.70, 2.50)

    ax.set_title("PepPrism 双分支门控融合与理性设计系统架构图", fontsize=12.5, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_length_audit(
    datasets_info: Sequence[dict[str, object]],
    out_path: Path | str,
) -> None:
    """Plot peptide length distributions (positive vs negative) across benchmarks."""
    p = _ensure_parent(out_path)
    n = len(datasets_info)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, info in zip(axes, datasets_info):
        pos_lens = np.asarray(info["pos_lengths"])
        neg_lens = np.asarray(info["neg_lengths"])
        bins = np.linspace(5, min(140, max(pos_lens.max(), neg_lens.max())), 26)
        ax.hist(pos_lens, bins=bins, density=True, alpha=0.58, color="#2563eb", label=f"阳性样本 (n={len(pos_lens)})")
        ax.hist(neg_lens, bins=bins, density=True, alpha=0.50, color="#ef4444", label=f"阴性样本 (n={len(neg_lens)})")
        ax.set_title(
            f"{info['title']}\nKS={info['ks_stat']:.3f}",
            fontsize=10.5,
            fontweight="bold",
            linespacing=1.35,
            pad=8,
        )
        ax.set_xlabel("多肽序列长度 (aa)", fontsize=9.5)
        ax.set_ylabel("概率密度", fontsize=9.5)
        ax.legend(fontsize=8.2, frameon=True)
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("公开基准数据集正/负样本序列长度分布与采样偏差审计", fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    benchmark_curves: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    out_path: Path | str,
) -> None:
    """Plot ROC and Precision-Recall curves across benchmarks and models."""
    p = _ensure_parent(out_path)
    names = list(benchmark_curves.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 7.6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    palette = {
        "PepPrismNet": ("#1d4ed8", "-", 2.3),
        "BiGRUAttention": ("#0891b2", "--", 1.7),
        "SeqCNN1D": ("#10b981", "-.", 1.6),
        "HistGradientBoosting": ("#d97706", ":", 1.8),
        "RandomForest": ("#7c3aed", "--", 1.4),
        "SVM_RBF": ("#be185d", "-.", 1.4),
        "LogisticRegression": ("#64748b", ":", 1.3),
    }

    for col, ds_name in enumerate(names):
        models = benchmark_curves[ds_name]
        ax_roc = axes[0, col]
        ax_pr = axes[1, col]

        for model_name, (y_true, y_prob) in models.items():
            color, ls, lw = palette.get(model_name, ("#334155", "-", 1.4))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                        label=f"{model_name} ({roc_auc:.3f})")

            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(rec, prec)
            ax_pr.plot(rec, prec, color=color, linestyle=ls, linewidth=lw,
                       label=f"{model_name} ({pr_auc:.3f})")

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1.0)
        ax_roc.set_title(f"{ds_name} - ROC 曲线", fontsize=10.5, fontweight="bold")
        ax_roc.set_xlabel("假阳性率 (FPR)", fontsize=9)
        ax_roc.set_ylabel("真阳性率 (TPR)", fontsize=9)
        ax_roc.legend(fontsize=7.3, loc="lower right", frameon=True)
        ax_roc.grid(alpha=0.25, linestyle="--")

        ax_pr.set_title(f"{ds_name} - PR 曲线", fontsize=10.5, fontweight="bold")
        ax_pr.set_xlabel("召回率 (Recall)", fontsize=9)
        ax_pr.set_ylabel("精确率 (Precision)", fontsize=9)
        ax_pr.legend(fontsize=7.3, loc="lower left", frameon=True)
        ax_pr.grid(alpha=0.25, linestyle="--")

    fig.suptitle("多基准测试集上不同模型 ROC 与 Precision-Recall 性能对比", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_and_calibration(
    ablation_rows: Sequence[dict[str, object]],
    calibration_curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path | str,
) -> None:
    """Plot ablation bar comparison (AUROC/MCC) and probability calibration reliability diagram."""
    p = _ensure_parent(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    labels = [str(r["variant"]) for r in ablation_rows]
    aurocs = [float(r["auroc"]) for r in ablation_rows]
    mccs = [float(r["mcc"]) for r in ablation_rows]
    x = np.arange(len(labels))
    w = 0.34

    b1 = ax1.bar(x - w / 2, aurocs, width=w, color="#2563eb", label="AUROC")
    b2 = ax1.bar(x + w / 2, mccs, width=w, color="#0d9488", label="MCC")
    for rect in list(b1) + list(b2):
        val = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=7.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.8)
    ax1.set_ylim(0.0, 1.18)
    ax1.set_ylabel("指标得分", fontsize=9.5)
    ax1.set_title("双分支架构消融实验对比", fontsize=10.8, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper right", ncol=2, frameon=True)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    ax2.plot([0, 1], [0, 1], "k--", lw=1.1, label="理想校准线")
    colors = ["#1d4ed8", "#059669", "#d97706", "#dc2626"]
    for idx, (m_name, (y_true, y_prob)) in enumerate(calibration_curves.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")
        ax2.plot(prob_pred, prob_true, marker="o", lw=1.8,
                 color=colors[idx % len(colors)], label=m_name)

    ax2.set_xlabel("模型预测平均置信概率", fontsize=9.5)
    ax2.set_ylabel("实际阳性样本比例", fontsize=9.5)
    ax2.set_title("预测概率可靠性校准曲线 (Reliability Diagram)", fontsize=10.8, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_interpretability_case(
    explanation: PeptideExplanation,
    peptide_name: str,
    out_path: Path | str,
) -> None:
    """Plot combined per-residue attribution bar chart and alpha-helical wheel projection."""
    p = _ensure_parent(out_path)
    wheel = compute_helical_wheel(explanation.sequence, max_residues=36)

    fig = plt.figure(figsize=(11.8, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_wheel = fig.add_subplot(gs[0, 1])

    residues = explanation.residue_attributions
    positions = [r.position for r in residues]
    aa_labels = [f"{r.amino_acid}{r.position}" for r in residues]
    imp = [r.combined_importance for r in residues]
    ala_d = [r.alanine_scan_delta for r in residues]
    bar_colors = [GROUP_COLORS.get(r.group, "#64748b") for r in residues]

    ax_bar.bar(positions, imp, color=bar_colors, alpha=0.88, label="综合归因重要性")
    ax_bar.plot(positions, np.clip(ala_d, -0.3, 1.0), color="#0f172a", marker="o",
                markersize=3.5, lw=1.3, label="丙氨酸扫描活性变化 (ΔP)")
    ax_bar.axhline(0.0, color="#94a3b8", lw=0.8)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(aa_labels, rotation=65, fontsize=7.0)
    ax_bar.set_ylabel("归一化归因强度 / ΔP", fontsize=9.5)
    ax_bar.set_title(
        f"{peptide_name} 逐残基可解释性归因 (P={explanation.probability:.3f}±{explanation.uncertainty_std:.3f}, 序列门控={explanation.seq_gate_ratio:.2f})",
        fontsize=10.2, fontweight="bold",
    )
    ax_bar.legend(fontsize=8.0, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--")

    # Helical wheel projection
    circle1 = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="#cbd5e1", lw=1.0)
    circle2 = plt.Circle((0, 0), 1.24, fill=False, linestyle=":", color="#e2e8f0", lw=1.0)
    ax_wheel.add_patch(circle1)
    ax_wheel.add_patch(circle2)

    if len(wheel.points) > 1:
        xs = [pt.x for pt in wheel.points]
        ys = [pt.y for pt in wheel.points]
        ax_wheel.plot(xs, ys, color="#94a3b8", lw=0.8, alpha=0.65, zorder=1)

    for pt in wheel.points:
        c = GROUP_COLORS.get(pt.group, "#64748b")
        ax_wheel.scatter([pt.x], [pt.y], s=210, color=c, edgecolors="#0f172a", lw=0.8, zorder=3)
        ax_wheel.text(pt.x, pt.y, f"{pt.amino_acid}{pt.position}", ha="center", va="center",
                      fontsize=6.2, fontweight="bold", color="white", zorder=4)

    # Draw hydrophobic moment arrow scaled for visibility
    scale_v = 1.1 / max(wheel.moment_magnitude, 0.25)
    ax_wheel.arrow(0, 0, wheel.moment_vx * scale_v * 0.45, wheel.moment_vy * scale_v * 0.45,
                   head_width=0.08, head_length=0.10, fc="#dc2626", ec="#dc2626", lw=2.0, zorder=5)

    ax_wheel.set_xlim(-1.55, 1.55)
    ax_wheel.set_ylim(-1.55, 1.55)
    ax_wheel.set_aspect("equal")
    ax_wheel.axis("off")
    ax_wheel.set_title(
        f"α-螺旋轮空间投影 (μH={wheel.moment_magnitude:.3f}, 方向={wheel.moment_angle_deg:.1f}°)",
        fontsize=10.2, fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS["positive"], label="阳离子 (K/R/H)"),
        mpatches.Patch(color=GROUP_COLORS["hydrophobic"], label="疏水脂肪族"),
        mpatches.Patch(color=GROUP_COLORS["aromatic"], label="芳香族 (F/W/Y)"),
        mpatches.Patch(color=GROUP_COLORS["polar"], label="极性中性"),
    ]
    ax_wheel.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.10),
                    ncol=2, fontsize=7.5, frameon=False)

    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_optimization_pareto(
    parent_prob: float,
    parent_hemolysis: float,
    candidates: Sequence[dict[str, object]],
    out_path: Path | str,
) -> None:
    """Plot directed mutagenesis activity vs hemolysis-risk Pareto frontier."""
    p = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    feas_x = [1.0 - float(c["hemolysis_risk_heuristic"]) for c in candidates if not c["pareto_optimal"]]
    feas_y = [float(c["probability"]) for c in candidates if not c["pareto_optimal"]]

    par_pts = [c for c in candidates if c["pareto_optimal"]]
    par_pts.sort(key=lambda c: float(c["hemolysis_risk_heuristic"]))
    par_x = [1.0 - float(c["hemolysis_risk_heuristic"]) for c in par_pts]
    par_y = [float(c["probability"]) for c in par_pts]

    if feas_x:
        ax.scatter(feas_x, feas_y, c="#64748b", alpha=0.65, s=45, label="约束可行突变体")
    if par_x:
        ax.plot(par_x, par_y, "r--", lw=1.4, alpha=0.8)
        ax.scatter(par_x, par_y, c="#dc2626", s=80, marker="*", zorder=4, label="Pareto 最优前沿")
        # Annotate representative diverse Pareto endpoints without overlapping labels
        picks = [par_pts[0], par_pts[len(par_pts) // 2], par_pts[-1]] if len(par_pts) >= 3 else par_pts
        offsets = [(-44, 6), (-86, -15), (6, 9)]
        for idx_p, c in enumerate(picks):
            dx, dy = offsets[idx_p % len(offsets)]
            ax.annotate(
                str(c["mutations"]),
                (1.0 - float(c["hemolysis_risk_heuristic"]), float(c["probability"])),
                textcoords="offset points", xytext=(dx, dy), fontsize=8.0, fontweight="bold", color="#991b1b",
            )

    ax.scatter([1.0 - parent_hemolysis], [parent_prob], c="#2563eb", s=110, marker="D", zorder=5, label="原始母本肽")
    ax.annotate("母本肽", (1.0 - parent_hemolysis, parent_prob),
                textcoords="offset points", xytext=(8, -8), fontsize=8.5, fontweight="bold", color="#1d4ed8")

    y_vals = feas_y + par_y + [parent_prob]
    y_min, y_max = min(y_vals), max(y_vals)
    y_span = max(y_max - y_min, 1e-4)
    ax.set_ylim(y_min - 0.08 * y_span, min(1.0005, y_max + 0.16 * y_span))
    ax.set_xlabel("生物相容安全性指数 (1 - 溶血风险启发分)", fontsize=9.8)
    ax.set_ylabel("预测治疗活性概率 (P_active)", fontsize=9.8)
    ax.set_title("约束定向进化突变体活性-安全性 Pareto 优化分布", fontsize=11.2, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
