"""
src/preprocess.py
=================
Data loading, SQL-based EDA, and feature engineering.
"""

import os
import sqlite3
import pandas as pd
import numpy as np


FEATURE_COLS = [
    "revolving_utilization", "age", "past_due_30_59", "debt_ratio",
    "monthly_income", "open_credit_lines", "past_due_90_plus",
    "real_estate_loans", "past_due_60_89", "num_dependents",
    "income_per_dependent", "total_past_due", "credit_stability",
    "age_income_ratio", "high_utilization", "senior_borrower"
]

TARGET_COL = "target"


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load CSV and rename columns to friendly names."""
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = [
        "target", "revolving_utilization", "age",
        "past_due_30_59", "debt_ratio", "monthly_income",
        "open_credit_lines", "past_due_90_plus", "real_estate_loans",
        "past_due_60_89", "num_dependents"
    ]
    print(f"[load] Raw shape: {df.shape}")
    return df


def sql_eda(df: pd.DataFrame, db_path: str = "data/credit_risk.db") -> None:
    """Push data to SQLite and run exploratory queries."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("applicants", conn, if_exists="replace", index=False)

    print("\n" + "=" * 55)
    print("SQL-BASED EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    queries = {
        "Default rate": """
            SELECT target,
                   COUNT(*) AS count,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM applicants), 2) AS pct
            FROM applicants GROUP BY target
        """,
        "Avg stats by default status": """
            SELECT target,
                   ROUND(AVG(monthly_income), 0) AS avg_income,
                   ROUND(AVG(age), 1)             AS avg_age,
                   ROUND(AVG(debt_ratio), 3)      AS avg_debt_ratio,
                   ROUND(AVG(revolving_utilization), 3) AS avg_util
            FROM applicants GROUP BY target
        """,
        "High-utilization defaulters (>90% util)": """
            SELECT COUNT(*) AS count
            FROM applicants
            WHERE revolving_utilization > 0.9 AND target = 1
        """,
        "Defaulters by age group": """
            SELECT
                CASE
                    WHEN age < 30 THEN 'Under 30'
                    WHEN age BETWEEN 30 AND 50 THEN '30-50'
                    ELSE 'Over 50'
                END AS age_group,
                ROUND(100.0 * SUM(target) / COUNT(*), 1) AS default_rate_pct
            FROM applicants
            GROUP BY age_group
        """
    }

    for title, sql in queries.items():
        result = pd.read_sql(sql, conn)
        print(f"\n[{title}]")
        print(result.to_string(index=False))

    conn.close()
    print("\n[sql_eda] Done.\n")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and create domain-meaningful features."""
    df = df.copy()

    # ── Fill missing values ──
    df["monthly_income"].fillna(df["monthly_income"].median(), inplace=True)
    df["num_dependents"].fillna(0, inplace=True)

    # ── Clip extreme outliers ──
    df["revolving_utilization"] = df["revolving_utilization"].clip(0, 1)
    df["debt_ratio"]            = df["debt_ratio"].clip(0, 10)
    df["monthly_income"]        = df["monthly_income"].clip(0, 100_000)

    # ── Remove invalid ages ──
    df = df[df["age"] > 18].copy()

    # ── New features ──
    df["income_per_dependent"] = df["monthly_income"] / (df["num_dependents"] + 1)
    df["total_past_due"]       = (
        df["past_due_30_59"] + df["past_due_60_89"] + df["past_due_90_plus"]
    )
    df["credit_stability"]  = df["open_credit_lines"] / (df["total_past_due"] + 1)
    df["age_income_ratio"]  = df["age"] / (df["monthly_income"] + 1)
    df["high_utilization"]  = (df["revolving_utilization"] > 0.75).astype(int)
    df["senior_borrower"]   = (df["age"] > 60).astype(int)

    print(f"[engineer] Final shape: {df.shape}")
    return df


def get_X_y(df: pd.DataFrame):
    """Split into features and target."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y
