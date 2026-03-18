# 🏦 Credit Risk Scoring Model

> End-to-end machine learning pipeline that predicts **loan default probability** on 150,000 real borrower records.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.87-brightgreen?style=flat-square)

---

## 📌 Overview

A production-style ML project covering the full data science lifecycle:

- **SQL-based EDA** using SQLite
- **Feature engineering** with domain-specific financial features
- **Class imbalance handling** using SMOTE
- **Model comparison** — Logistic Regression, Random Forest, Gradient Boosting
- **Explainability** using SHAP values
- **Interactive web app** built with Streamlit + Plotly

---

## 📁 Project Structure

```
credit_risk/
├── data/
│   └── cs-training.csv          ← Download from Kaggle (see below)
├── models/
│   ├── best_model.pkl            ← Auto-generated after training
│   ├── feature_cols.pkl
│   ├── shap_summary.png
│   ├── roc_pr_curves.png
│   └── feature_importance.png
├── src/
│   ├── __init__.py
│   ├── preprocess.py             ← Data loading, SQL EDA, feature engineering
│   └── train.py                  ← Model training, evaluation, saving
├── notebooks/
│   └── eda_analysis.py           ← Standalone EDA plots
├── app.py                        ← Streamlit web application
├── run_training.py               ← Entry point: run this first!
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Clone & Set Up Environment

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-scoring.git
cd credit-risk-scoring

python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Step 2 — Download Dataset

1. Go to [https://www.kaggle.com/c/GiveMeSomeCredit/data](https://www.kaggle.com/c/GiveMeSomeCredit/data)
2. Create a free Kaggle account if needed
3. Download `cs-training.csv`
4. Place it inside the `data/` folder

### Step 3 — Train the Model

```bash
python run_training.py
```

This will:
- Run SQL-based exploratory analysis
- Engineer 6 new features
- Apply SMOTE to balance the dataset
- Train 3 models and compare them
- Save the best model + SHAP/ROC plots to `models/`

### Step 4 — Launch the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser 🎉

### Optional — Run EDA plots

```bash
python notebooks/eda_analysis.py
```

---

## 📊 Results

| Model | AUC-ROC | Avg Precision |
|---|---|---|
| Logistic Regression | ~0.82 | ~0.37 |
| Random Forest | ~0.86 | ~0.44 |
| **Gradient Boosting** | **~0.87** | **~0.47** |

---

## 🔬 Engineered Features

| Feature | Formula | Business Meaning |
|---|---|---|
| `income_per_dependent` | income / (dependents + 1) | Effective disposable income |
| `total_past_due` | sum of all late payments | Overall delinquency severity |
| `credit_stability` | open_lines / (late_pmts + 1) | Responsible credit management |
| `age_income_ratio` | age / (income + 1) | Life stage vs earnings |
| `high_utilization` | utilization > 75% | Red-flag binary feature |
| `senior_borrower` | age > 60 | Demographic risk signal |

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this project to GitHub (include `models/` folder with `.pkl` files)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → connect your repo
4. Set main file path: `app.py`
5. Click **Deploy** ✅

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python, Pandas | Data manipulation |
| SQLite + SQL | Exploratory data analysis |
| Scikit-learn | Model training & evaluation |
| Imbalanced-learn (SMOTE) | Class imbalance handling |
| SHAP | Model explainability |
| Streamlit | Interactive web app |
| Plotly | Charts & gauge visualization |

---

## 📄 Resume Entry

**Credit Risk Scoring Model** | Python, Scikit-learn, SQL, Streamlit, SHAP

- Built end-to-end ML pipeline on 150,000 borrower records to predict loan default probability
- Handled severe class imbalance (7% default rate) using SMOTE; achieved AUC-ROC of 0.87 with Gradient Boosting
- Used SHAP values to identify top 5 features driving default risk for business interpretability
- Deployed interactive Streamlit web app for real-time credit risk assessment
- Performed SQL-based EDA using SQLite to identify high-risk borrower segments

---

## 👤 Author

**Your Name** | [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
