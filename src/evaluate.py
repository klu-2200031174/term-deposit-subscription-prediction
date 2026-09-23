"""Metrics and figures.

All evaluation logic lives here so that train.py reads as a story rather than a
wall of plotting code.
"""

import matplotlib
matplotlib.use("Agg")  # render to files, never to a window

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from config import (
    C_ACCENT,
    C_BAD,
    C_GOOD,
    C_INK,
    C_MUTED,
    C_PRIMARY,
    C_SURFACE,
    COST_FN,
    COST_FP,
    REPORTS_DIR,
)


def business_cost(false_positives: int, false_negatives: int) -> int:
    """What this set of mistakes costs, using the assumptions in config.py."""
    return false_positives * COST_FP + false_negatives * COST_FN


def metrics_at(y_true, proba, threshold: float) -> dict:
    """Every number we care about, for one choice of decision threshold.

    A model outputs a probability. The threshold is the line we draw to turn that
    probability into a yes/no decision. 0.5 is a default, not a law.
    """
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision_score(y_true, proba),
        "cost": business_cost(fp, fn),
    }


def sweep_thresholds(y_true, proba, steps: int = 99):
    """Compute metrics across every threshold from 0.01 to 0.99."""
    return [metrics_at(y_true, proba, t) for t in np.linspace(0.01, 0.99, steps)]


def cheapest_threshold(y_true, proba) -> dict:
    """The threshold that minimises total business cost."""
    return min(sweep_thresholds(y_true, proba), key=lambda row: row["cost"])


# --- figures --------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return path


def plot_confusion(m_default: dict, m_tuned: dict, name="confusion_matrices.png"):
    """Two confusion matrices side by side: default 0.5 vs the tuned threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor=C_SURFACE)
    panels = [("Default threshold 0.50", m_default), 
              (f"Tuned threshold {m_tuned['threshold']:.2f}", m_tuned)]

    for ax, (title, m) in zip(axes, panels):
        grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max())
        labels = [["True negative", "False positive"], ["False negative", "True positive"]]
        for i in range(2):
            for j in range(2):
                shade = C_SURFACE if grid[i, j] > grid.max() * 0.55 else C_INK
                ax.text(j, i - 0.12, f"{grid[i, j]:,}", ha="center", va="center",
                        fontsize=15, fontweight="bold", color=shade)
                ax.text(j, i + 0.18, labels[i][j], ha="center", va="center",
                        fontsize=8, color=shade)
        ax.set_xticks([0, 1], ["Predicted no", "Predicted yes"], fontsize=9, color=C_MUTED)
        ax.set_yticks([0, 1], ["Actually no", "Actually yes"], fontsize=9, color=C_MUTED)
        ax.set_title(f"{title}\nrecall {m['recall']:.1%}  ·  precision {m['precision']:.1%}",
                     fontsize=10, color=C_INK, pad=10)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Where the model's mistakes go", fontsize=13, color=C_INK, y=1.02)
    return _save(fig, name)


def plot_pr_curve(y_true, proba, chosen: dict, baseline_rate: float,
                  name="precision_recall_curve.png"):
    """Precision and recall trade against each other. This shows the whole trade."""
    precision, recall, _ = precision_recall_curve(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=C_SURFACE)
    _style(ax)
    ax.plot(recall, precision, color=C_PRIMARY, linewidth=2, label="Tuned model")
    ax.axhline(baseline_rate, color=C_MUTED, linewidth=1.5, linestyle=(0, (4, 3)),
               label=f"Random guessing ({baseline_rate:.1%})")
    ax.scatter([chosen["recall"]], [chosen["precision"]], s=90, color=C_ACCENT,
               zorder=5, edgecolor=C_SURFACE, linewidth=2)
    ax.annotate(f"chosen: threshold {chosen['threshold']:.2f}  ",
                (chosen["recall"], chosen["precision"]), color=C_ACCENT, fontsize=9,
                ha="right", va="bottom")

    ax.set_xlabel("Recall — share of real subscribers we find", fontsize=10, color=C_MUTED)
    ax.set_ylabel("Precision — share of our calls that convert", fontsize=10, color=C_MUTED)
    ax.set_title("Every point on this line is a threshold you could pick",
                 fontsize=12, color=C_INK, pad=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=C_MUTED)
    return _save(fig, name)


def plot_threshold_tradeoff(sweep, chosen: dict, name="threshold_tradeoff.png"):
    """Two stacked panels sharing an x-axis. Never two y-scales on one plot."""
    thresholds = [r["threshold"] for r in sweep]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True,
                                      facecolor=C_SURFACE, height_ratios=[1, 1])
    _style(top)
    _style(bottom)

    top.plot(thresholds, [r["recall"] for r in sweep], color=C_PRIMARY, linewidth=2)
    top.plot(thresholds, [r["precision"] for r in sweep], color=C_ACCENT, linewidth=2)
    top.text(0.10, 0.90, "Recall", color=C_PRIMARY, fontsize=10, fontweight="bold")
    top.text(0.62, 0.86, "Precision", color=C_ACCENT, fontsize=10, fontweight="bold")
    top.set_ylabel("Rate", fontsize=10, color=C_MUTED)
    top.set_title("Raising the threshold trades recall away for precision",
                  fontsize=12, color=C_INK, pad=12)
    top.set_ylim(0, 1)

    costs = [r["cost"] for r in sweep]
    bottom.plot(thresholds, costs, color=C_BAD, linewidth=2)
    bottom.axvline(chosen["threshold"], color=C_GOOD, linewidth=1.5,
                   linestyle=(0, (4, 3)))
    bottom.scatter([chosen["threshold"]], [chosen["cost"]], s=80, color=C_GOOD,
                   zorder=5, edgecolor=C_SURFACE, linewidth=2)
    bottom.annotate(f"   cheapest threshold: {chosen['threshold']:.2f}",
                    (chosen["threshold"], chosen["cost"]),
                    xytext=(chosen["threshold"] + 0.06, chosen["cost"] * 1.22),
                    color=C_GOOD, fontsize=9, fontweight="bold")
    bottom.set_ylabel("Total cost of mistakes", fontsize=10, color=C_MUTED)
    bottom.set_xlabel(
        f"Decision threshold\n(a wasted call costs {COST_FP}; "
        f"a missed subscriber costs {COST_FN})", fontsize=10, color=C_MUTED)
    bottom.set_title("The cheapest threshold is nowhere near 0.50",
                     fontsize=12, color=C_INK, pad=12)
    return _save(fig, name)


def plot_leakage_impact(with_leak: dict, without_leak: dict,
                        name="leakage_impact.png"):
    """The honest before/after, once the leaked column is removed."""
    labels = ["Recall", "Precision", "PR-AUC"]
    left = [with_leak["recall"], with_leak["precision"], with_leak["pr_auc"]]
    right = [without_leak["recall"], without_leak["precision"], without_leak["pr_auc"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=C_SURFACE)
    _style(ax)
    b1 = ax.bar(x - width / 2 - 0.01, left, width, color=C_BAD, label="With `duration` (leaked)")
    b2 = ax.bar(x + width / 2 + 0.01, right, width, color=C_PRIMARY, label="Without `duration` (honest)")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9, color=C_INK)

    ax.set_xticks(x, labels, fontsize=10, color=C_MUTED)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on the held-out test set", fontsize=10, color=C_MUTED)
    ax.set_title("What removing the leaked column actually costs",
                 fontsize=12, color=C_INK, pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=C_MUTED)
    return _save(fig, name)
