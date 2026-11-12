"""The five mandatory figures + extras. Every chart has title, axis labels, dataset, model and method."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

COLORS = {"All features": "#64748b", "SelectKBest": "#2864a5", "RFE": "#e07a1f", "Manual": "#2f9e6e"}


def fig_acc_vs_features(sweep, dataset, model):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for method, g in sweep.groupby("Method"):
        g = g.sort_values("Features")
        ax.errorbar(g["Features"], g["CV accuracy"], yerr=g["CV SD"], marker="o", capsize=3,
                    label=method, color=COLORS.get(method))
    ax.set(title=f"{dataset} | {model} | SelectKBest vs RFE\nAccuracy vs number of features",
           xlabel="Number of selected features", ylabel="5-fold CV accuracy (training split)")
    lo = max(0, sweep["CV accuracy"].min() - 0.08)
    ax.set_ylim(lo, 1.02); ax.grid(alpha=.25); ax.legend(title="Method")
    ax.set_xticks(sorted(sweep["Features"].unique()))
    fig.tight_layout(); return fig


def fig_baseline_vs_opt(baseline, opt, dataset, model, method):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(2); w = .36
    b = ax.bar(x - w/2, [baseline["accuracy"], baseline["macro_f1"]], w, label=f"All features ({baseline['k']})", color=COLORS[ALL_KEY])
    o = ax.bar(x + w/2, [opt["accuracy"], opt["macro_f1"]], w, label=f"{method} ({opt['k']})", color=COLORS.get(method))
    for bars in (b, o):
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, ["Test accuracy", "Test macro-F1"])
    ax.set(title=f"{dataset} | {model} | {method}\nBaseline vs optimized", ylabel="Score")
    ax.set_ylim(max(0, min(baseline["accuracy"], opt["accuracy"], baseline["macro_f1"], opt["macro_f1"]) - .15), 1.05)
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); return fig

ALL_KEY = "All features"


def fig_feature_scores(score_df, chosen, dataset, model, method):
    top = score_df.head(15).sort_values("F-score")
    cols = [COLORS.get(method, "#2864a5") if f in chosen else "#b6c2d4" for f in top["Feature"]]
    fig, ax = plt.subplots(figsize=(7.5, max(3, .33 * len(top) + 1.2)))
    ax.barh(top["Feature"], top["F-score"], color=cols)
    ax.set(title=f"{dataset} | {model} | {method}\nFeature scores (ANOVA F, training split) - selected in colour",
           xlabel="F-score", ylabel="Feature")
    ax.grid(axis="x", alpha=.2); fig.tight_layout(); return fig


def fig_confusion(y_true, y_pred, class_names, dataset, model, method, k):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [class_names[i] if i < len(class_names) else str(i) for i in labels]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax, cbar=False)
    ax.set(title=f"{dataset} | {model} | {method} (k={k})\nConfusion matrix (held-out test set)",
           xlabel="Predicted class", ylabel="Actual class")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); return fig


def fig_acc_vs_runtime(points, dataset, model):
    """points: list of (label, method, train_time, test_accuracy)"""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, method, t, acc in points:
        ax.scatter(t, acc, s=130, color=COLORS.get(method, "#444"), label=label, zorder=3)
        ax.annotate(label, (t, acc), xytext=(7, 6), textcoords="offset points", fontsize=8)
    ax.set(title=f"{dataset} | {model}\nTest accuracy vs training time",
           xlabel="Training time of final pipeline (seconds)", ylabel="Test accuracy")
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); return fig


def fig_class_distribution(y, class_names, dataset):
    vc = y.value_counts().sort_index()
    names = [class_names[i] if i < len(class_names) else str(i) for i in vc.index]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(names, vc.values, color="#2864a5"); ax.bar_label(bars, padding=2)
    ax.set(title=f"{dataset} - class distribution", xlabel="Class", ylabel="Samples")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right"); fig.tight_layout(); return fig


def fig_correlation(X, dataset):
    n = X.shape[1]
    fig, ax = plt.subplots(figsize=(min(11, 4 + .35 * n), min(9, 3.5 + .3 * n)))
    sns.heatmap(X.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax, annot=n <= 13, fmt=".2f",
                annot_kws={"size": 7})
    ax.set_title(f"{dataset} - feature correlation (training split)"); fig.tight_layout(); return fig
