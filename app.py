"""
app.py
======
Streamlit Credit Risk Scoring App

Run:  streamlit run app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] label { color: #e2e8f0 !important; }

/* Main background */
.stApp { background: #f8fafc; }

/* Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    text-align: center;
    height: 100%;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}
.risk-low  { background: #dcfce7; color: #15803d; }
.risk-mid  { background: #fef9c3; color: #854d0e; }
.risk-high { background: #fee2e2; color: #b91c1c; }

/* Section header */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

/* Input labels */
label { color: #374151 !important; font-size: 0.875rem !important; font-weight: 500 !important; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS — load model
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        model_name = open("models/model_name.txt").read().strip()
        return model, feature_cols, model_name
    except FileNotFoundError:
        return None, None, None


def build_input_features(inputs: dict) -> pd.DataFrame:
    """Mirrors engineer_features() from preprocess.py."""
    df = pd.DataFrame([inputs])
    df["income_per_dependent"] = df["monthly_income"] / (df["num_dependents"] + 1)
    df["total_past_due"]       = df["past_due_30_59"] + df["past_due_60_89"] + df["past_due_90_plus"]
    df["credit_stability"]     = df["open_credit_lines"] / (df["total_past_due"] + 1)
    df["age_income_ratio"]     = df["age"] / (df["monthly_income"] + 1)
    df["high_utilization"]     = int(inputs["revolving_utilization"] > 0.75)
    df["senior_borrower"]      = int(inputs["age"] > 60)
    return df


def demo_predict(inputs: dict) -> float:
    """Rule-based approximation for demo mode (no trained model)."""
    util  = inputs["revolving_utilization"]
    lates = inputs["past_due_30_59"] + inputs["past_due_60_89"] * 2 + inputs["past_due_90_plus"] * 3
    debt  = inputs["debt_ratio"]
    inc   = max(inputs["monthly_income"], 1)
    prob  = min(0.95, max(0.02,
        util * 0.40 + lates * 0.07 + debt * 0.08 + (5000 / inc) * 0.05
    ))
    return prob


# ─────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────

def gauge_chart(prob: float) -> go.Figure:
    pct   = round(prob * 100, 1)
    color = "#16a34a" if prob < 0.30 else "#ca8a04" if prob < 0.60 else "#dc2626"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 52, "color": color, "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 30, 60, 100],
                "ticktext": ["0%", "30%", "60%", "100%"],
                "tickcolor": "#94a3b8", "tickwidth": 1,
                "tickfont": {"size": 11, "color": "#94a3b8"}
            },
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "#dcfce7"},
                {"range": [30, 60],  "color": "#fef9c3"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.85, "value": pct
            }
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=30, b=10, l=30, r=30),
        height=260,
        font={"family": "Inter"}
    )
    return fig


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

def page_predict(model, feature_cols, model_name):
    """Main prediction page."""

    st.markdown("## 🔍 Predict Default Risk")
    st.markdown("Fill in the applicant's details and click **Predict** to get a risk score.")

    if model is None:
        st.warning(
            "⚠️ **Running in Demo Mode** — model not trained yet.  \n"
            "Run `python run_training.py` to train the real model."
        )

    st.markdown("---")

    # ── INPUTS ──
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)
        age            = st.slider("Age", 18, 90, 40, help="Applicant's age in years")
        monthly_income = st.number_input("Monthly Income (USD)", 0, 100_000, 5_000, step=500,
                                         help="Gross monthly income before taxes")
        num_dependents = st.number_input("Number of Dependents", 0, 20, 0,
                                         help="Number of people financially dependent on applicant")

        st.markdown("---")
        st.markdown('<p class="section-header">🏠 Assets & Loans</p>', unsafe_allow_html=True)
        real_estate_loans = st.number_input("Real Estate Loans / Lines", 0, 20, 1)
        open_credit_lines = st.number_input("Open Credit Lines", 0, 50, 6)

    with col_right:
        st.markdown('<p class="section-header">💳 Credit Profile</p>', unsafe_allow_html=True)
        revolving_utilization = st.slider(
            "Revolving Credit Utilization (%)", 0, 100, 30,
            help="Percentage of revolving credit in use (e.g., credit cards)"
        ) / 100.0
        debt_ratio = st.slider(
            "Debt-to-Income Ratio", 0.0, 5.0, 0.35, step=0.01,
            help="Monthly debt payments divided by gross monthly income"
        )

        st.markdown("---")
        st.markdown('<p class="section-header">⚠️ Delinquency History</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            past_due_30_59 = st.number_input("30–59 Days Late", 0, 20, 0)
        with c2:
            past_due_60_89 = st.number_input("60–89 Days Late", 0, 20, 0)
        with c3:
            past_due_90_plus = st.number_input("90+ Days Late", 0, 20, 0)

    st.markdown("---")

    predict_btn = st.button("🔍  Predict Default Risk", type="primary", use_container_width=True)

    if predict_btn:
        inputs = dict(
            revolving_utilization=revolving_utilization, age=age,
            past_due_30_59=int(past_due_30_59), debt_ratio=debt_ratio,
            monthly_income=monthly_income, open_credit_lines=int(open_credit_lines),
            past_due_90_plus=int(past_due_90_plus), real_estate_loans=int(real_estate_loans),
            past_due_60_89=int(past_due_60_89), num_dependents=int(num_dependents)
        )

        if model is not None:
            X = build_input_features(inputs)[feature_cols]
            prob = float(model.predict_proba(X)[0][1])
        else:
            prob = demo_predict(inputs)

        # ── Result display ──
        st.markdown("## 📊 Risk Assessment Result")

        r_col1, r_col2 = st.columns([1.2, 1], gap="large")

        with r_col1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)

            if prob < 0.30:
                label, css, advice = "LOW RISK", "risk-low", "✅ Applicant is likely to repay. Loan can be approved."
            elif prob < 0.60:
                label, css, advice = "MEDIUM RISK", "risk-mid", "⚠️ Moderate risk. Consider additional verification or collateral."
            else:
                label, css, advice = "HIGH RISK", "risk-high", "🚨 High default probability. Manual review strongly recommended."

            st.markdown(
                f'<div style="text-align:center;margin-top:-10px">'
                f'<span class="risk-badge {css}">{label}</span>'
                f'<p style="margin-top:10px;color:#475569;font-size:0.9rem">{advice}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        with r_col2:
            st.markdown("#### Key Metrics")

            total_late = int(past_due_30_59) + int(past_due_60_89) + int(past_due_90_plus)
            dti_pct    = round(debt_ratio * 100, 1)
            util_pct   = round(revolving_utilization * 100, 1)

            metrics = [
                (f"{prob*100:.1f}%", "Default Probability", "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.3 else "#22c55e"),
                (f"{util_pct}%",     "Credit Utilization",  "#ef4444" if util_pct > 75 else "#3b82f6"),
                (f"{dti_pct}%",      "Debt-to-Income",      "#ef4444" if dti_pct > 43 else "#3b82f6"),
                (f"{total_late}",    "Total Late Payments", "#ef4444" if total_late > 2 else "#22c55e"),
                (f"${monthly_income:,}", "Monthly Income",  "#3b82f6"),
                (f"{age} yrs",       "Age",                 "#6366f1"),
            ]

            for i in range(0, len(metrics), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(metrics):
                        val, lbl, color = metrics[i + j]
                        col.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value" style="color:{color}">{val}</div>'
                            f'<div class="metric-label">{lbl}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                st.write("")

        # ── Risk factors breakdown ──
        st.markdown("#### 🔎 Risk Factor Analysis")
        factors = {
            "High credit utilization (>75%)": revolving_utilization > 0.75,
            "Multiple late payments (>2)": total_late > 2,
            "High debt-to-income ratio (>43%)": debt_ratio > 0.43,
            "Recent 90+ day delinquency": past_due_90_plus > 0,
            "Low monthly income (<$2,000)": monthly_income < 2000,
        }
        positive_factors = {k: v for k, v in factors.items() if v}
        good_factors     = {k: v for k, v in factors.items() if not v}

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            if positive_factors:
                st.error("**Risk Flags Detected:**\n" + "\n".join(f"• {k}" for k in positive_factors))
            else:
                st.success("**No major risk flags detected**")
        with f_col2:
            if good_factors:
                st.success("**Positive Indicators:**\n" + "\n".join(f"• No {k.lower()}" for k in list(good_factors)[:3]))


def page_model_insights():
    """Show model plots saved during training."""
    st.markdown("## 📈 Model Insights")

    plots = {
        "ROC & Precision-Recall Curves": "models/roc_pr_curves.png",
        "SHAP Feature Importance": "models/shap_summary.png",
        "Feature Importance (Built-in)": "models/feature_importance.png",
    }

    any_found = False
    for title, path in plots.items():
        if os.path.exists(path):
            any_found = True
            st.markdown(f"### {title}")
            st.image(path, use_column_width=True)
            st.markdown("---")

    if not any_found:
        st.info("📭 No plots found yet. Run `python run_training.py` first to generate model plots.")


def page_about():
    """About / documentation page."""
    st.markdown("## 📚 About This Project")

    st.markdown("""
    ### What This Does
    This app predicts whether a loan applicant is likely to default on their debt within 2 years,
    using historical borrower data from the **Give Me Some Credit** Kaggle competition.

    ---
    ### ML Pipeline
    | Step | Details |
    |------|---------|
    | **Data Source** | 150,000 borrower records (Kaggle) |
    | **EDA** | SQL queries via SQLite |
    | **Preprocessing** | Missing value imputation, outlier clipping |
    | **Feature Engineering** | 6 new domain-specific features |
    | **Class Imbalance** | SMOTE oversampling (~7% default rate) |
    | **Models Trained** | Logistic Regression, Random Forest, Gradient Boosting |
    | **Best Model** | Gradient Boosting — AUC-ROC ~0.87 |
    | **Explainability** | SHAP values |

    ---
    ### Engineered Features
    | Feature | Description |
    |---------|-------------|
    | `income_per_dependent` | Monthly income normalized by family size |
    | `total_past_due` | Sum of all late payment instances |
    | `credit_stability` | Open credit lines vs. delinquency ratio |
    | `age_income_ratio` | Age normalized by income |
    | `high_utilization` | Flag: credit utilization > 75% |
    | `senior_borrower` | Flag: age > 60 |

    ---
    ### Tech Stack
    `Python` · `Pandas` · `SQLite` · `Scikit-learn` · `Imbalanced-learn` · `SHAP` · `Streamlit` · `Plotly`
    """)


# ─────────────────────────────────────────────
# SIDEBAR + ROUTING
# ─────────────────────────────────────────────

def main():
    model, feature_cols, model_name = load_model()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 24px">
            <div style="font-size:1.5rem;font-weight:700;color:#f8fafc;margin-bottom:4px">🏦 CreditAI</div>
            <div style="font-size:0.8rem;color:#64748b">Credit Risk Scoring Model</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🔍 Predict Risk", "📈 Model Insights", "📚 About"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Model status
        if model is not None:
            st.markdown(f"""
            <div style="background:#1e3a5f;border-radius:8px;padding:12px 14px;margin-top:8px">
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Model Status</div>
                <div style="color:#34d399;font-size:0.85rem;font-weight:600">✅ Trained</div>
                <div style="color:#94a3b8;font-size:0.75rem;margin-top:4px">{model_name}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#2d1b1b;border-radius:8px;padding:12px 14px;margin-top:8px">
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Model Status</div>
                <div style="color:#f87171;font-size:0.85rem;font-weight:600">⚠️ Demo Mode</div>
                <div style="color:#94a3b8;font-size:0.75rem;margin-top:4px">Run run_training.py</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.72rem;color:#334155;text-align:center">'
            'Built with Scikit-learn · SHAP · Streamlit</p>',
            unsafe_allow_html=True
        )

    # Route to page
    if "Predict" in page:
        page_predict(model, feature_cols, model_name)
    elif "Insights" in page:
        page_model_insights()
    else:
        page_about()


if __name__ == "__main__":
    main()
