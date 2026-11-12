"""OptiSelect - the five mandatory figures plus two EDA plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS = Path(__file__).parent / "results"
COLORS = {"All features": "#6b7280", "SelectKBest": "#2563eb", "RFE": "#ea580c"}


def _ctx(r, method=None):
    return f"{r['dataset']} | {r['clf']}" + (f" | {method}" if method else "")


def save(fig, name):
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def fig_acc_vs_k(r):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, g in r["sweep"].groupby("method"):
        ax.errorbar(g.k, g.cv_mean, yerr=g.cv_std, marker="o", capsize=3, label=m, color=COLORS[m])
    ax.set(xlabel="Number of selected features (k)", ylabel="CV accuracy (train split, 5-fold)",
           title=f"Fig 1: Accuracy vs number of features\n{_ctx(r)} | SelectKBest vs RFE")
    ax.set_xticks(sorted(r["sweep"].k.unique()))
    ax.grid(alpha=.3); ax.legend()
    return save(fig, f"fig1_acc_vs_k_{r['dataset']}_{r['clf']}".replace(" ", "_"))


def fig_baseline(r):
    d, m = r["details"], r["method"]
    fig, ax = plt.subplots(figsize=(6, 4))
    w = .35
    for i, (lab, key) in enumerate([("Accuracy", "test_acc"), ("Macro-F1", "macro_f1")]):
        vals = [d["All features"][key], d[m][key]]
        bars = ax.bar([j + (i - .5) * w for j in range(2)], vals, w, label=lab)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks([0, 1], [f"All ({r['n']})", f"{m} ({r['k']})"])
    ax.set(ylabel="Test score", ylim=(0, 1.1),
           title=f"Fig 2: Baseline vs optimized\n{_ctx(r, m)}")
    ax.legend(loc="lower right")
    return save(fig, f"fig2_baseline_{r['dataset']}_{r['clf']}_{m}".replace(" ", "_"))


def fig_scores(r):
    m = r["method"]
    s = r["details"][m]["scores"].sort_values("score")
    fig, ax = plt.subplots(figsize=(7, max(3, .3 * len(s) + 1)))
    ax.barh(s.feature, s.score, color=[COLORS[m] if x else "#cbd5e1" for x in s.selected])
    ax.set(xlabel="ANOVA F-score" if m == "SelectKBest" else "1 / RFE rank (higher = kept longer)",
           ylabel="Feature", title=f"Fig 3: Feature scores (selected = coloured)\n{_ctx(r, m)}")
    return save(fig, f"fig3_scores_{r['dataset']}_{r['clf']}_{m}".replace(" ", "_"))


def fig_cm(r):
    m = r["method"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(r["details"][m]["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=r["classes"], yticklabels=r["classes"])
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Fig 4: Confusion matrix (test set)\n{_ctx(r, m)}")
    return save(fig, f"fig4_cm_{r['dataset']}_{r['clf']}_{m}".replace(" ", "_"))


def fig_runtime(r):
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, v in r["details"].items():
        ax.scatter(v["train_time"], v["test_acc"], s=120, color=COLORS[lab])
        ax.annotate(f"{lab} (k={v['k']})", (v["train_time"], v["test_acc"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set(xlabel="Training time (s, incl. selection)", ylabel="Test accuracy",
           title=f"Fig 5: Accuracy vs runtime\n{_ctx(r)} | k={r['k']}")
    ax.grid(alpha=.3)
    return save(fig, f"fig5_runtime_{r['dataset']}_{r['clf']}".replace(" ", "_"))


def fig_class_dist(r):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = r["y"].value_counts().sort_index()
    ax.bar(r["classes"], counts.values, color="#2563eb")
    ax.set(ylabel="Samples", title=f"Class distribution - {r['dataset']}")
    return fig


def fig_corr(r):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(r["X"].corr(), cmap="coolwarm", center=0, ax=ax,
                annot=r["n"] <= 13, fmt=".2f", annot_kws={"size": 7})
    ax.set_title(f"Correlation heatmap - {r['dataset']}")
    return fig
