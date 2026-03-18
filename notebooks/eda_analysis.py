"""
notebooks/eda_analysis.py
=========================
Run this for a full EDA report saved as PNG charts.
Usage:  python notebooks/eda_analysis.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_raw, engineer_features

SAVE_DIR = "notebooks/eda_plots"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = "data/cs-training.csv"

sns.set_style("whitegrid")
PALETTE = ["#3b82f6", "#ef4444"]


def run_eda():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Dataset not found at {CSV_PATH}. Skipping EDA.")
        return

    df = load_raw(CSV_PATH)
    df = engineer_features(df)

    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDefault rate: {df['target'].mean():.2%}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Credit Risk — Exploratory Data Analysis", fontsize=15, fontweight="bold", y=1.02)

    # 1. Default rate
    ax = axes[0, 0]
    counts = df["target"].value_counts()
    ax.bar(["No Default", "Default"], counts.values, color=PALETTE, edgecolor="none", width=0.5)
    ax.set_title("Default Rate Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)

    # 2. Age distribution
    ax = axes[0, 1]
    for label, color, grp in zip(["No Default", "Default"], PALETTE, [0, 1]):
        subset = df[df["target"] == grp]["age"]
        ax.hist(subset, bins=40, alpha=0.65, color=color, label=label, edgecolor="none")
    ax.set_title("Age Distribution by Default", fontweight="bold")
    ax.set_xlabel("Age")
    ax.legend()

    # 3. Revolving utilization
    ax = axes[0, 2]
    for label, color, grp in zip(["No Default", "Default"], PALETTE, [0, 1]):
        subset = df[df["target"] == grp]["revolving_utilization"].clip(0, 1)
        ax.hist(subset, bins=40, alpha=0.65, color=color, label=label, edgecolor="none")
    ax.set_title("Credit Utilization by Default", fontweight="bold")
    ax.set_xlabel("Utilization Rate")
    ax.legend()

    # 4. Monthly income (log scale)
    ax = axes[1, 0]
    for label, color, grp in zip(["No Default", "Default"], PALETTE, [0, 1]):
        subset = df[df["target"] == grp]["monthly_income"].clip(1, 30_000)
        ax.hist(np.log1p(subset), bins=50, alpha=0.65, color=color, label=label, edgecolor="none")
    ax.set_title("Monthly Income (log) by Default", fontweight="bold")
    ax.set_xlabel("log(Income + 1)")
    ax.legend()

    # 5. Total past due
    ax = axes[1, 1]
    df["total_past_due_clip"] = df["total_past_due"].clip(0, 10)
    avg_default = df.groupby("total_past_due_clip")["target"].mean()
    ax.bar(avg_default.index, avg_default.values * 100, color="#f59e0b", edgecolor="none")
    ax.set_title("Default Rate by Total Late Payments", fontweight="bold")
    ax.set_xlabel("Total Past Due (clipped at 10)")
    ax.set_ylabel("Default Rate (%)")

    # 6. Correlation heatmap
    ax = axes[1, 2]
    num_cols = ["revolving_utilization", "debt_ratio", "monthly_income",
                "total_past_due", "age", "target"]
    corr = df[num_cols].corr()
    labels = ["Util.", "Debt Ratio", "Income", "Late Pmts", "Age", "Default"]
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlBu_r",
        linewidths=0.5, center=0, cbar=False,
        xticklabels=labels, yticklabels=labels, annot_kws={"size": 9}
    )
    ax.set_title("Correlation Matrix", fontweight="bold")

    plt.tight_layout()
    out = f"{SAVE_DIR}/eda_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ EDA plot saved → {out}")


if __name__ == "__main__":
    run_eda()
