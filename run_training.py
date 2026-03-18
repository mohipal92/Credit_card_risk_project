"""
run_training.py
===============
Entry point — run this first to train the model.

Usage:
    python run_training.py
"""

import os
import sys

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_raw, sql_eda, engineer_features
from src.train import train_and_evaluate

CSV_PATH = "data/cs-training.csv"


def main():
    print("=" * 55)
    print("  CREDIT RISK SCORING — TRAINING PIPELINE")
    print("=" * 55)

    # 1. Check dataset
    if not os.path.exists(CSV_PATH):
        print(f"\n❌  Dataset not found at: {CSV_PATH}")
        print("\nTo fix this:")
        print("  1. Go to  https://www.kaggle.com/c/GiveMeSomeCredit/data")
        print("  2. Sign in to Kaggle (free)")
        print("  3. Download  cs-training.csv")
        print("  4. Place it inside the  data/  folder")
        print("\nThen re-run:  python run_training.py\n")
        sys.exit(1)

    # 2. Load raw data
    df_raw = load_raw(CSV_PATH)

    # 3. SQL-based EDA
    sql_eda(df_raw)

    # 4. Feature engineering
    df = engineer_features(df_raw)

    # 5. Train, evaluate, save
    model, best_name = train_and_evaluate(df)

    print("\n" + "=" * 55)
    print(f"  ✅  DONE — Best model: {best_name}")
    print("  Saved artifacts:")
    print("    models/best_model.pkl")
    print("    models/feature_cols.pkl")
    print("    models/shap_summary.png")
    print("    models/roc_pr_curves.png")
    print("    models/feature_importance.png")
    print("\n  Now run:  streamlit run app.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
