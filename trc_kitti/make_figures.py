import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def box(ax, xy, width, height, text, color, text_color="white"):
    patch = Rectangle(xy, width, height, facecolor=color, edgecolor="none")
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
            color=text_color, fontsize=9, fontweight="bold")


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.3, color="#3b4652"))


def framework():
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 4.0); ax.axis("off")
    navy, teal, orange, gray = "#17324d", "#167d7f", "#c7602b", "#586474"
    box(ax, (0.2, 2.45), 1.65, 0.78, "Camera\n2D candidates", navy)
    box(ax, (2.25, 2.45), 1.55, 0.78, "Distance gate\n40 m", gray)
    box(ax, (4.2, 2.45), 1.65, 0.78, "LiDAR\n3D association", teal)
    box(ax, (6.25, 2.45), 1.7, 0.78, "LiDAR ROI\nrange proxy", orange)
    box(ax, (8.35, 2.45), 1.65, 0.78, "Range-gated\nobject output", navy)
    for a, b in [((1.85, 2.84), (2.25, 2.84)), ((3.8, 2.84), (4.2, 2.84)),
                 ((5.85, 2.84), (6.25, 2.84)), ((7.95, 2.84), (8.35, 2.84))]:
        arrow(ax, a, b)
    ax.text(3.02, 1.72, "far field", color=navy, fontsize=9, fontweight="bold", ha="center")
    ax.text(3.02, 1.35, "retain early camera candidate", color="#3b4652", fontsize=8, ha="center")
    arrow(ax, (3.02, 2.45), (3.02, 1.9))
    ax.text(6.05, 1.72, "mid / near field", color=teal, fontsize=9, fontweight="bold", ha="center")
    ax.text(6.05, 1.35, "replace uncertain depth with geometric state", color="#3b4652", fontsize=8, ha="center")
    arrow(ax, (6.05, 2.45), (6.05, 1.9))
    ax.plot([0.55, 9.65], [0.62, 0.62], color="#8b949e", linewidth=1.2)
    ax.scatter([0.55, 4.9, 9.65], [0.62] * 3, color=[navy, teal, orange], s=28, zorder=3)
    ax.text(0.55, 0.25, ">40 m", ha="center", fontsize=8)
    ax.text(4.9, 0.25, "20-40 m", ha="center", fontsize=8)
    ax.text(9.65, 0.25, "0-20 m", ha="center", fontsize=8)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"framework.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def performance(metrics_path):
    payload = json.loads(Path(metrics_path).read_text())
    rows = payload["results"]
    labels = ["Camera", "LiDAR", "Late fusion", "Range gate"]
    colors = ["#17324d", "#167d7f", "#8b949e", "#c7602b"]
    x = range(len(rows)); width = 0.23
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    for offset, key, name in [(-width, "near_recall", "Near (0-20 m)"),
                              (0, "mid_recall", "Mid (20-40 m)"),
                              (width, "far_recall", "Far (40-70 m)")]:
        axes[0].bar([i + offset for i in x], [r[key] for r in rows], width, label=name)
    axes[0].set_xticks(list(x), labels, rotation=12)
    axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("Recall")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", color="#d9dde2", linewidth=0.7)
    for i, row in enumerate(rows):
        axes[1].scatter(row["mean_latency_ms"], row["near_localization_error_m"],
                        s=75, color=colors[i], label=labels[i])
    axes[1].set_xlabel("Mean latency (ms)")
    axes[1].set_ylabel("Near-field localization error (m)")
    axes[1].grid(color="#d9dde2", linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"performance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def camera_convergence(results_path):
    with Path(results_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    map50 = [float(row["metrics/mAP50(B)"]) for row in rows]
    map95 = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    recall = [float(row["metrics/recall(B)"]) for row in rows]
    fitness = [0.1 * a + 0.9 * b for a, b in zip(map50, map95)]
    best_index = max(range(len(fitness)), key=fitness.__getitem__)

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.plot(epochs, map50, marker="o", markersize=3, label="mAP50")
    ax.plot(epochs, map95, marker="s", markersize=3, label="mAP50-95")
    ax.plot(epochs, recall, marker="^", markersize=3, label="Recall")
    ax.axvline(epochs[best_index], color="#c7602b", linestyle="--", linewidth=1.2,
               label=f"Selected epoch {epochs[best_index]}")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Calibration metric")
    ax.set_ylim(0.2, 0.62)
    ax.grid(color="#d9dde2", linewidth=0.7)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"camera_convergence.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    framework()
    candidate = ROOT / "output_locked" / "metrics_locked.json"
    if candidate.exists():
        performance(candidate)
    training_results = ROOT / "training" / "camera_locked_converged" / "results.csv"
    if training_results.exists():
        camera_convergence(training_results)
