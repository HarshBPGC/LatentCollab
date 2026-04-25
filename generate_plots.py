"""
Generate comprehensive matplotlib visualizations for all project components:
  1. AQI Module — dimension scores, composite AQI, radar chart
  2. Latent Steering — fit loss curves, distance reduction
  3. Stackelberg Regret — multi-scenario comparison, sub-metric breakdown
  4. PettingZoo Pipeline — full pipeline metrics, drift tracking, BRG analysis
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pettingzoo.mpe import simple_spread_v3

from aqi_module import (
    AQIResult,
    AlignmentScorer,
    AQIPipeline,
    evaluate_alignment_profiles,
    simple_embed_fn,
    _make_well_aligned_model,
    _make_partially_aligned_model,
    _make_poorly_aligned_model,
)
from latent_steering import LatentSteeringAdapter, SteeringFitStats
from stackelberg_regret_eval import (
    AgentStep,
    EpisodeBuffer,
    StackelbergRegretEvaluator,
    StackelbergRegretResult,
    make_mock_agent_step,
)
from pettingzoo_sr_eval import (
    AgentNetwork,
    train_agents,
    select_baseline_agent,
    fit_latent_steering,
    collect_aligned_latents,
    run_episode,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})

COLORS = {
    "well": "#2ecc71",
    "partial": "#f39c12",
    "poor": "#e74c3c",
    "aligned": "#2ecc71",
    "drifted": "#e74c3c",
    "steered": "#3498db",
    "random": "#95a5a6",
}

FIGURES_DIR = Path("figures")


def save_plot(save_path: str | Path) -> None:
    """Save the current Matplotlib figure and create the output directory."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: AQI Module Results
# ═══════════════════════════════════════════════════════════════

def plot_aqi_results(
    aqi_results: list[AQIResult],
    save_path: str | Path = FIGURES_DIR / "fig1_aqi_results.png",
):
    sorted_results = sorted(aqi_results, key=lambda r: r.aqi, reverse=True)
    model_names = [r.model_name for r in sorted_results]
    dimensions = list(sorted_results[0].dimension_scores.keys())
    short_names = [n.replace("agent_", "Agent ") for n in model_names]
    profile_labels = []
    for r in sorted_results:
        profile_labels.append(r.model_name)

    colors_list = []
    for r in sorted_results:
        p = getattr(r, "_profile", "")
        if "well" in r.model_name or r.aqi > 0.7:
            colors_list.append(COLORS["well"])
        elif "poor" in r.model_name or r.aqi < 0.4:
            colors_list.append(COLORS["poor"])
        else:
            colors_list.append(COLORS["partial"])

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel A: Composite AQI Bar Chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(short_names, [r.aqi for r in sorted_results], color=colors_list,
                   edgecolor="white", linewidth=1.5, width=0.6)
    for bar, r in zip(bars, sorted_results):
        label = f"{r.aqi:.3f}"
        if r.is_baseline:
            label += " ★"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 label, ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax1.set_ylabel("AQI Score")
    ax1.set_title("(a) Composite Alignment Quality Index")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Midpoint")
    ax1.legend(fontsize=8)

    # --- Panel B: Per-Dimension Grouped Bar Chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(dimensions))
    width = 0.25
    for i, r in enumerate(sorted_results):
        scores = [r.dimension_scores[d].score for d in dimensions]
        offset = (i - 1) * width
        ax2.bar(x + offset, scores, width, label=short_names[i], color=colors_list[i],
                edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace("_", "\n") for d in dimensions], fontsize=8)
    ax2.set_ylabel("Score")
    ax2.set_title("(b) Per-Dimension Scores")
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8, loc="upper right")

    # --- Panel C: Radar Chart ---
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    dim_labels = [d.replace("_", "\n").title() for d in dimensions]
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(dim_labels, fontsize=8)
    ax3.set_ylim(0, 1.0)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)

    for i, r in enumerate(sorted_results):
        values = [r.dimension_scores[d].score for d in dimensions]
        values += values[:1]
        ax3.plot(angles, values, "o-", linewidth=1.8, label=short_names[i],
                 color=colors_list[i], markersize=4)
        ax3.fill(angles, values, alpha=0.1, color=colors_list[i])
    ax3.set_title("(c) Alignment Radar", y=1.08)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # --- Panel D: AQI Weight Breakdown (stacked bar) ---
    ax4 = fig.add_subplot(gs[1, 1])
    weights = {"harmlessness": 0.30, "helpfulness": 0.25, "honesty": 0.20,
               "consistency": 0.15, "instruction_fidelity": 0.10}
    dim_colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    bottom = np.zeros(len(sorted_results))
    for j, dim in enumerate(dimensions):
        contrib = [r.dimension_scores[dim].score * weights.get(dim, 0.2) for r in sorted_results]
        ax4.bar(short_names, contrib, bottom=bottom, label=dim.replace("_", " ").title(),
                color=dim_colors[j], edgecolor="white", linewidth=0.8, width=0.6)
        bottom += np.array(contrib)
    ax4.set_ylabel("Weighted Contribution to AQI")
    ax4.set_title("(d) AQI Decomposition (Weighted)")
    ax4.legend(fontsize=7, loc="upper right")
    ax4.set_ylim(0, 1.05)

    fig.suptitle("Figure 1: Alignment Quality Index (AQI) — Model Evaluation", fontsize=14, fontweight="bold", y=0.98)
    save_plot(save_path)


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Latent Steering Adapter Results
# ═══════════════════════════════════════════════════════════════

def plot_steering_results(
    fit_stats: dict[str, SteeringFitStats],
    loss_curves: dict[str, list[float]],
    save_path: str | Path = FIGURES_DIR / "fig2_latent_steering.png",
):
    agent_names = list(fit_stats.keys())
    short_names = [n.replace("agent_", "Agent ") for n in agent_names]
    agent_colors = ["#3498db", "#e74c3c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: Loss Curves ---
    ax = axes[0, 0]
    for i, name in enumerate(agent_names):
        curve = loss_curves[name]
        ax.plot(curve, label=short_names[i], color=agent_colors[i % len(agent_colors)], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE + Regularization)")
    ax.set_title("(a) Steering Adapter Training Loss")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- Panel B: Before/After Loss ---
    ax = axes[0, 1]
    x = np.arange(len(agent_names))
    width = 0.35
    before_loss = [fit_stats[n].initial_loss for n in agent_names]
    after_loss = [fit_stats[n].final_loss for n in agent_names]
    bars1 = ax.bar(x - width / 2, before_loss, width, label="Before Steering", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width / 2, after_loss, width, label="After Steering", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("MSE Loss")
    ax.set_title("(b) Latent Prediction Loss: Before vs After")
    ax.legend(fontsize=9)
    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    # --- Panel C: Target Distance Reduction ---
    ax = axes[1, 0]
    before_dist = [fit_stats[n].target_distance_before for n in agent_names]
    after_dist = [fit_stats[n].target_distance_after for n in agent_names]
    bars1 = ax.bar(x - width / 2, before_dist, width, label="Before", color="#e67e22", alpha=0.8)
    bars2 = ax.bar(x + width / 2, after_dist, width, label="After", color="#27ae60", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Mean L2 Distance to Baseline")
    ax.set_title("(c) Latent Distance to Baseline: Before vs After Steering")
    ax.legend(fontsize=9)
    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    # --- Panel D: Distance Reduction Percentage ---
    ax = axes[1, 1]
    reductions = []
    for name in agent_names:
        b = fit_stats[name].target_distance_before
        a = fit_stats[name].target_distance_after
        reductions.append(((b - a) / b) * 100 if b > 0 else 0)
    bars = ax.bar(short_names, reductions, color=["#3498db", "#e74c3c"][:len(agent_names)],
                  edgecolor="white", width=0.5)
    for b, pct in zip(bars, reductions):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Distance Reduction (%)")
    ax.set_title("(d) Steering Effectiveness (% Distance Reduction)")
    ax.set_ylim(0, max(reductions) * 1.2 if reductions else 100)

    fig.suptitle("Figure 2: Latent Steering Adapter — Training & Effectiveness",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(save_path)


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Stackelberg Regret — 4-Scenario Comparison
# ═══════════════════════════════════════════════════════════════

def plot_scenario_comparison(
    aligned_results: list[StackelbergRegretResult],
    drifted_results: list[StackelbergRegretResult],
    steered_results: list[StackelbergRegretResult],
    random_results: list[StackelbergRegretResult],
    save_path: str | Path = FIGURES_DIR / "fig3_scenario_comparison.png",
):
    def avg(results, attr):
        if not results:
            return 0.0
        return float(np.mean([getattr(r, attr) for r in results]))

    def std(results, attr):
        if not results:
            return 0.0
        return float(np.std([getattr(r, attr) for r in results]))

    scenarios = ["Aligned", "Drifted", "Steered", "Random"]
    scenario_colors = [COLORS["aligned"], COLORS["drifted"], COLORS["steered"], COLORS["random"]]
    all_results = [aligned_results, drifted_results, steered_results, random_results]

    metrics = ["behavioral_sr", "latent_sr", "nash_deviation", "latent_cone_rate", "composite_sr"]
    metric_labels = ["Behavioral SR", "Latent SR", "Nash Deviation", "Cone Rate", "Composite SR"]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # --- Panel A: Composite SR Comparison ---
    ax = fig.add_subplot(gs[0, 0])
    means = [avg(r, "composite_sr") for r in all_results]
    stds = [std(r, "composite_sr") for r in all_results]
    bars = ax.bar(scenarios, means, yerr=stds, color=scenario_colors,
                  edgecolor="white", linewidth=1.5, width=0.6, capsize=4)
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.6, label="Alignment Threshold")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Composite SR")
    ax.set_title("(a) Composite Stackelberg Regret")
    ax.legend(fontsize=8)

    # --- Panel B: All Sub-Metrics Grouped ---
    ax = fig.add_subplot(gs[0, 1:])
    x = np.arange(len(metrics))
    width = 0.2
    for i, (label, color, results) in enumerate(zip(scenarios, scenario_colors, all_results)):
        vals = [avg(results, m) for m in metrics]
        errs = [std(results, m) for m in metrics]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=label, color=color,
               edgecolor="white", linewidth=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Metric Value")
    ax.set_title("(b) All SR Sub-Metrics by Scenario")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel C: Behavioral SR Box Plot ---
    ax = fig.add_subplot(gs[1, 0])
    bsr_data = [[r.behavioral_sr for r in res] if res else [0] for res in all_results]
    bp = ax.boxplot(bsr_data, tick_labels=scenarios, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], scenario_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Behavioral SR")
    ax.set_title("(c) Behavioral SR Distribution")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)

    # --- Panel D: Latent Cone Rate Comparison ---
    ax = fig.add_subplot(gs[1, 1])
    cone_means = [avg(r, "latent_cone_rate") for r in all_results]
    cone_stds = [std(r, "latent_cone_rate") for r in all_results]
    bars = ax.bar(scenarios, cone_means, yerr=cone_stds, color=scenario_colors,
                  edgecolor="white", linewidth=1.5, width=0.6, capsize=4)
    for b, m in zip(bars, cone_means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Cone Membership Rate")
    ax.set_title("(d) Latent Cone Membership")
    ax.set_ylim(0, 1.1)

    # --- Panel E: Per-Follower BRG Heatmap ---
    ax = fig.add_subplot(gs[1, 2])
    all_fids = set()
    for results in all_results:
        for r in results:
            all_fids.update(r.per_follower_brg.keys())
    fids = sorted(all_fids)

    if fids:
        brg_matrix = np.zeros((len(scenarios), len(fids)))
        for i, results in enumerate(all_results):
            for j, fid in enumerate(fids):
                vals = [r.per_follower_brg.get(fid, 0) for r in results]
                brg_matrix[i, j] = np.mean(vals) if vals else 0

        im = ax.imshow(brg_matrix, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(len(fids)))
        ax.set_xticklabels([f"Follower {f}" for f in fids], fontsize=8)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=9)
        for i in range(len(scenarios)):
            for j in range(len(fids)):
                ax.text(j, i, f"{brg_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if brg_matrix[i, j] > brg_matrix.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, label="BRG")
    ax.set_title("(e) Per-Follower Best-Response Gap")

    fig.suptitle("Figure 3: Stackelberg Regret — Multi-Scenario Evaluation",
                 fontsize=14, fontweight="bold", y=0.98)
    save_plot(save_path)


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Drift Tracking Over Episodes
# ═══════════════════════════════════════════════════════════════

def plot_drift_tracking(
    aligned_results: list[StackelbergRegretResult],
    evaluator: StackelbergRegretEvaluator,
    save_path: str | Path = FIGURES_DIR / "fig4_drift_tracking.png",
):
    if len(aligned_results) < 3:
        print("  Skipping drift tracking (need >= 3 episodes)")
        return

    drift = evaluator.track_drift(aligned_results, window=3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_colors = {
        "composite_sr": "#8e44ad",
        "behavioral_sr": "#2980b9",
        "latent_sr": "#e74c3c",
        "nash_deviation": "#f39c12",
        "cone_rate": "#27ae60",
    }
    metric_labels = {
        "composite_sr": "Composite SR",
        "behavioral_sr": "Behavioral SR",
        "latent_sr": "Latent SR",
        "nash_deviation": "Nash Deviation",
        "cone_rate": "Cone Membership Rate",
    }

    for idx, (key, vals) in enumerate(drift.items()):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        raw_vals = [getattr(r, key if key != "cone_rate" else "latent_cone_rate")
                    for r in aligned_results]
        episodes = range(1, len(vals) + 1)
        ax.plot(episodes, raw_vals, "o", alpha=0.4, color=metric_colors[key], markersize=5, label="Raw")
        ax.plot(episodes, vals, "-", linewidth=2, color=metric_colors[key], label="Smoothed (w=3)")
        ax.fill_between(episodes, vals, alpha=0.15, color=metric_colors[key])
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_labels[key])
        ax.set_title(metric_labels[key])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Use last panel for a summary trend table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for key, vals in drift.items():
        trend = "Stable" if abs(vals[-1] - vals[0]) < 0.05 else (
            "↑ Increasing" if vals[-1] > vals[0] else "↓ Decreasing")
        table_data.append([metric_labels[key], f"{vals[0]:.4f}", f"{vals[-1]:.4f}", trend])

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "First", "Last", "Trend"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("Drift Summary", fontsize=11, fontweight="bold", pad=20)

    fig.suptitle("Figure 4: Alignment Drift Tracking Over Episodes",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(save_path)


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Steering Impact — Side-by-Side SR Decomposition
# ═══════════════════════════════════════════════════════════════

def plot_steering_impact(
    drifted_results: list[StackelbergRegretResult],
    steered_results: list[StackelbergRegretResult],
    save_path: str | Path = FIGURES_DIR / "fig5_steering_impact.png",
):
    def avg(results, attr):
        return float(np.mean([getattr(r, attr) for r in results])) if results else 0.0

    metrics = ["behavioral_sr", "latent_sr", "nash_deviation", "composite_sr"]
    labels = ["Behavioral SR", "Latent SR", "Nash Deviation", "Composite SR"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel A: Before vs After Steering ---
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    drifted_vals = [avg(drifted_results, m) for m in metrics]
    steered_vals = [avg(steered_results, m) for m in metrics]
    ax.bar(x - width / 2, drifted_vals, width, label="Drifted (No Steering)",
           color=COLORS["drifted"], edgecolor="white")
    ax.bar(x + width / 2, steered_vals, width, label="Drifted + Steered",
           color=COLORS["steered"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Metric Value")
    ax.set_title("(a) Steering Reduces Misalignment Metrics")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel B: Improvement Arrows ---
    ax = axes[1]
    improvements = [(d - s) / abs(d) * 100 if abs(d) > 1e-6 else 0
                    for d, s in zip(drifted_vals, steered_vals)]
    bar_colors = ["#27ae60" if imp > 0 else "#e74c3c" for imp in improvements]
    bars = ax.barh(labels, improvements, color=bar_colors, edgecolor="white", height=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
    for b, imp in zip(bars, improvements):
        sign = "+" if imp > 0 else ""
        xpos = b.get_width() + (1 if imp >= 0 else -1)
        ax.text(xpos, b.get_y() + b.get_height() / 2,
                f"{sign}{imp:.1f}%", va="center", fontweight="bold", fontsize=9)
    ax.set_xlabel("% Improvement (positive = better)")
    ax.set_title("(b) Steering Improvement (%)")

    # --- Panel C: Cone Rate Recovery ---
    ax = axes[2]
    cone_drifted = avg(drifted_results, "latent_cone_rate")
    cone_steered = avg(steered_results, "latent_cone_rate")
    categories = ["Drifted", "Steered"]
    vals = [cone_drifted, cone_steered]
    bar_colors = [COLORS["drifted"], COLORS["steered"]]
    bars = ax.bar(categories, vals, color=bar_colors, edgecolor="white", width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Cone Membership Rate")
    ax.set_title("(c) Latent Cone Recovery After Steering")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3, label="Perfect")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 5: Latent Steering Impact — Misalignment Correction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(save_path)


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Training Reward Curve + Episode-Level SR Scatter
# ═══════════════════════════════════════════════════════════════

def plot_episode_level_analysis(
    aligned_results: list[StackelbergRegretResult],
    drifted_results: list[StackelbergRegretResult],
    steered_results: list[StackelbergRegretResult],
    random_results: list[StackelbergRegretResult],
    save_path: str | Path = FIGURES_DIR / "fig6_episode_analysis.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel A: Composite SR per Episode ---
    ax = axes[0]
    labels_map = [
        ("Aligned", aligned_results, COLORS["aligned"]),
        ("Drifted", drifted_results, COLORS["drifted"]),
        ("Steered", steered_results, COLORS["steered"]),
        ("Random", random_results, COLORS["random"]),
    ]
    for label, results, color in labels_map:
        if results:
            vals = [r.composite_sr for r in results]
            ax.plot(range(1, len(vals) + 1), vals, "o-", label=label, color=color,
                    markersize=5, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Eval Episode")
    ax.set_ylabel("Composite SR")
    ax.set_title("(a) Composite SR Per Evaluation Episode")
    ax.legend(fontsize=9)
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.4, label="Threshold")
    ax.grid(True, alpha=0.3)

    # --- Panel B: Behavioral vs Latent SR Scatter ---
    ax = axes[1]
    for label, results, color in labels_map:
        if results:
            bsr = [r.behavioral_sr for r in results]
            lsr = [r.latent_sr for r in results]
            ax.scatter(bsr, lsr, label=label, color=color, alpha=0.7, s=50, edgecolors="white")
    ax.set_xlabel("Behavioral SR")
    ax.set_ylabel("Latent SR")
    ax.set_title("(b) Behavioral vs Latent SR")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Nash Deviation vs Cone Rate ---
    ax = axes[2]
    for label, results, color in labels_map:
        if results:
            ndi = [r.nash_deviation for r in results]
            cone = [r.latent_cone_rate for r in results]
            ax.scatter(ndi, cone, label=label, color=color, alpha=0.7, s=50, edgecolors="white")
    ax.set_xlabel("Nash Deviation Index")
    ax.set_ylabel("Latent Cone Rate")
    ax.set_title("(c) Nash Deviation vs Cone Membership")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 6: Episode-Level Analysis of SR Metrics",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(save_path)


# ═══════════════════════════════════════════════════════════════
# RUNNER: Execute full pipeline and generate all figures
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Generating Matplotlib Visualizations for All Components")
    print("=" * 60)

    LATENT_DIM = 16
    N_ACTIONS = 5
    ALIGNMENT_PROFILES = {
        "agent_0": "well_aligned",
        "agent_1": "partially_aligned",
        "agent_2": "poorly_aligned",
    }

    # ── 1. AQI Evaluation ─────────────────────────────────────
    print("\n[1/6] Running AQI evaluation...")
    aqi_results, aqi_baseline = evaluate_alignment_profiles(
        ALIGNMENT_PROFILES, probe_path="alignment_probes.json"
    )
    plot_aqi_results(aqi_results)

    # ── 2. Train Agents ───────────────────────────────────────
    print("\n[2/6] Training PettingZoo agents (300 episodes)...")
    probe_env = simple_spread_v3.env(N=3, max_cycles=25, render_mode=None)
    probe_env.reset()
    obs_dim = probe_env.observation_space("agent_0").shape[0]
    probe_env.close()

    networks = train_agents(
        obs_dim, LATENT_DIM, N_ACTIONS,
        alignment_profiles=ALIGNMENT_PROFILES, n_episodes=300,
    )

    leader_name, _ = select_baseline_agent(networks)
    print(f"  Baseline agent: {leader_name}")

    # ── 3. Fit Steering Adapters (with loss curve tracking) ───
    print("\n[3/6] Fitting latent steering adapters...")
    from pettingzoo_sr_eval import collect_steering_pairs
    paired_latents = collect_steering_pairs(networks, leader_name, n_episodes=30)

    adapters = {}
    fit_stats = {}
    loss_curves = {}

    for agent_name, (source_lats, target_lats) in paired_latents.items():
        adapter = LatentSteeringAdapter(latent_dim=LATENT_DIM, alpha=0.7)
        source_d = source_lats.detach()
        target_d = target_lats.detach()

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        identity = torch.eye(LATENT_DIM)

        with torch.no_grad():
            initial_pred = adapter(source_d)
            initial_loss = loss_fn(initial_pred, target_d).item()
            before_dist = (source_d - target_d).norm(dim=-1).mean().item()

        curve = []
        for _ in range(200):
            predicted = adapter(source_d)
            fit_loss = loss_fn(predicted, target_d)
            reg_loss = ((adapter.mapper.weight - identity) ** 2).mean()
            loss = fit_loss + 1e-3 * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curve.append(loss.item())

        with torch.no_grad():
            final_pred = adapter(source_d)
            final_loss = loss_fn(final_pred, target_d).item()
            after_dist = (final_pred - target_d).norm(dim=-1).mean().item()

        adapter.is_fit = True
        adapters[agent_name] = adapter
        fit_stats[agent_name] = SteeringFitStats(initial_loss, final_loss, before_dist, after_dist)
        loss_curves[agent_name] = curve

    plot_steering_results(fit_stats, loss_curves)

    # ── 4. Calibrate SR Evaluator ─────────────────────────────
    print("\n[4/6] Calibrating SR evaluator on aligned episodes...")
    aligned_leader_lats, aligned_follower_lats = collect_aligned_latents(
        networks, leader_name, n_episodes=30,
    )

    evaluator = StackelbergRegretEvaluator(
        latent_dim=LATENT_DIM, n_actions=N_ACTIONS,
        gamma=0.99, sigma_scale=3.0, sr_threshold=0.15,
    )
    evaluator.fit_aligned_cone(aligned_leader_lats)
    evaluator.fit_projection_head(aligned_leader_lats, aligned_follower_lats, epochs=100)

    # ── 5. Run Evaluation Episodes ────────────────────────────
    print("\n[5/6] Running evaluation episodes (4 scenarios × 10 episodes)...")
    N_EVAL = 10
    results_aligned, results_drifted, results_steered, results_random = [], [], [], []
    candidate_followers = [n for n in networks if n != leader_name]
    drifted_follower = candidate_followers[0]

    random_networks = {}
    for name, net in networks.items():
        random_networks[name] = AgentNetwork(
            obs_dim, LATENT_DIM, N_ACTIONS, alignment_profile=net.alignment_profile,
        )

    for ep in range(N_EVAL):
        buf_a, iq_a, oq_a = run_episode(networks, leader_name)
        if buf_a.leader_steps and iq_a:
            results_aligned.append(evaluator.evaluate(buf_a, iq_a, oq_a))

        buf_b, iq_b, oq_b = run_episode(
            networks, leader_name, drifted_follower=drifted_follower, drift_scale=1.25)
        if buf_b.leader_steps and iq_b:
            results_drifted.append(evaluator.evaluate(buf_b, iq_b, oq_b))

        buf_c, iq_c, oq_c = run_episode(
            networks, leader_name, latent_steering=adapters,
            drifted_follower=drifted_follower, drift_scale=1.25)
        if buf_c.leader_steps and iq_c:
            results_steered.append(evaluator.evaluate(buf_c, iq_c, oq_c))

        buf_d, iq_d, oq_d = run_episode(random_networks, leader_name)
        if buf_d.leader_steps and iq_d:
            results_random.append(evaluator.evaluate(buf_d, iq_d, oq_d))

    # ── 6. Generate Remaining Plots ───────────────────────────
    print("\n[6/6] Generating all plots...")
    plot_scenario_comparison(results_aligned, results_drifted, results_steered, results_random)
    plot_drift_tracking(results_aligned, evaluator)
    plot_steering_impact(results_drifted, results_steered)
    plot_episode_level_analysis(results_aligned, results_drifted, results_steered, results_random)

    print("\n" + "=" * 60)
    print("  All figures saved!")
    print("    figures/fig1_aqi_results.png        — AQI Module Evaluation")
    print("    figures/fig2_latent_steering.png     — Steering Adapter Training")
    print("    figures/fig3_scenario_comparison.png — 4-Scenario SR Comparison")
    print("    figures/fig4_drift_tracking.png      — Drift Tracking Over Episodes")
    print("    figures/fig5_steering_impact.png     — Steering Impact Analysis")
    print("    figures/fig6_episode_analysis.png    — Episode-Level SR Scatter")
    print("=" * 60)


if __name__ == "__main__":
    main()
