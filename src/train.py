"""
src/train.py
============
Train models, evaluate, pick the best, save artifacts.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE
import shap

from src.preprocess import get_X_y, FEATURE_COLS


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=20, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ),
    }


# ─────────────────────────────────────────────
# Train & Evaluate
# ─────────────────────────────────────────────

def train_and_evaluate(df, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[smote] Resampled: {dict(zip(*np.unique(y_res, return_counts=True)))}")

    models   = build_models()
    results  = {}

    print("\n" + "=" * 55)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 55)

    for name, model in models.items():
        print(f"\n▶ Training: {name}")
        model.fit(X_res, y_res)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob)

        print(f"  AUC-ROC        : {auc:.4f}")
        print(f"  Avg Precision  : {ap:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=["No Default", "Default"],
            digits=3
        ))

        results[name] = {
            "model": model, "auc": auc, "ap": ap,
            "y_prob": y_prob, "y_test": y_test
        }

    # ── Pick best ──
    best_name = max(results, key=lambda k: results[k]["auc"])
    best      = results[best_name]
    print(f"\n🏆  Best model: {best_name}  (AUC={best['auc']:.4f})")

    # ── Save artifacts ──
    with open(f"{output_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    with open(f"{output_dir}/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    with open(f"{output_dir}/model_name.txt", "w") as f:
        f.write(best_name)

    print(f"[save] Artifacts written to {output_dir}/")

    # ── Plots ──
    _plot_roc_pr(results, best_name, output_dir)
    _plot_shap(best["model"], best_name, X_test, output_dir)
    _plot_feature_importance(best["model"], best_name, output_dir)

    return best["model"], best_name


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def _plot_roc_pr(results, best_name, output_dir):
    """Save ROC and Precision-Recall curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#3b82f6", "#10b981", "#f59e0b"]

    for (name, res), color in zip(results.items(), colors):
        lw = 2.5 if name == best_name else 1.2
        alpha = 1.0 if name == best_name else 0.6

        RocCurveDisplay.from_predictions(
            res["y_test"], res["y_prob"],
            ax=axes[0], name=f"{name} ({res['auc']:.3f})",
            color=color, lw=lw, alpha=alpha
        )
        PrecisionRecallDisplay.from_predictions(
            res["y_test"], res["y_prob"],
            ax=axes[1], name=f"{name} ({res['ap']:.3f})",
            color=color, lw=lw, alpha=alpha
        )

    axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
    axes[1].set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#f8fafc")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] ROC/PR curves → {output_dir}/roc_pr_curves.png")


def _plot_shap(model, model_name, X_test, output_dir):
    """Generate SHAP bar summary plot."""
    try:
        import pandas as pd
        X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)

        if "Logistic" in model_name:
            scaler = model.named_steps["scaler"]
            clf    = model.named_steps["clf"]
            X_sc   = scaler.transform(X_sample)
            explainer   = shap.LinearExplainer(clf, X_sc)
            shap_values = explainer.shap_values(X_sc)
            X_plot = pd.DataFrame(X_sc, columns=X_sample.columns)
        else:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            X_plot      = X_sample.reset_index(drop=True)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_plot, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] SHAP summary → {output_dir}/shap_summary.png")
    except Exception as e:
        print(f"[plot] SHAP skipped: {e}")


def _plot_feature_importance(model, model_name, output_dir):
    """Bar chart of built-in feature importances (tree models only)."""
    try:
        import pandas as pd

        if "Logistic" in model_name:
            return

        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#3b82f6" if v > feat_imp.median() else "#94a3b8" for v in feat_imp]
        feat_imp.plot(kind="barh", ax=ax, color=colors, edgecolor="none")

        ax.set_title("Feature Importances", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("white")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Feature importance → {output_dir}/feature_importance.png")
    except Exception as e:
        print(f"[plot] Feature importance skipped: {e}")
