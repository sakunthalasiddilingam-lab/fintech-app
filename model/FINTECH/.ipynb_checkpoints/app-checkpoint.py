import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =========================
# LOAD FILES
# =========================
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")
num_cols = joblib.load("model/num_cols.pkl")
cat_cols = joblib.load("model/cat_cols.pkl")

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# FUNCTIONS (KEEP THESE ABOVE NAVIGATION)
# =========================
def build_input(data):
    income = data["inflow"]

    return pd.DataFrame([{
        "Age": data["age"],
        "Work_Experience": data["exp"],
        "Projects_Last_1M": data["p1"],
        "Projects_Last_6M": data["p6"],
        "Upcoming_Projects": data["upcoming"],
        "Avg_Project_Value": data["avg_val"],
        "Monthly_Expenses": data["monthly_exp"],
        "Income_1M": income,
        "Existing_Loans": data["loans"],
        "EMI_Amount": data["emi"],
        "Freelancer_Category": data["category"],
        "Platform": data["platform"],
        "monthly_Cash_Inflow": data["inflow"],
        "monthly_Cash_Outflow": data["outflow"],
        "Savings": data["savings"],
        "EMI_Income_Ratio": data["emi"]/income if income>0 else 0,
        "Income_Stability": data["p6"]/6 if data["p6"]>0 else 0,
        "Experience_Score": data["exp"]/data["age"] if data["age"]>0 else 0,
        "Loan_Burden": data["loans"]*data["emi"],
        "Income_per_Project": income/data["p6"] if data["p6"]>0 else income
    }])

def generate_report(row, score):
    report = []

    inflow = row["monthly_Cash_Inflow"].values[0]
    outflow = row["monthly_Cash_Outflow"].values[0]
    emi = row["EMI_Amount"].values[0]
    savings = row["Savings"].values[0]

    total_out = outflow + emi
    surplus = inflow - total_out

    if inflow > 0:
        exp_ratio = row["Monthly_Expenses"].values[0] / inflow
        emi_ratio = emi / inflow
    else:
        exp_ratio = 0
        emi_ratio = 0

    if exp_ratio > 0.6:
        report.append("High expense-to-income ratio affecting repayment capacity.")

    if emi_ratio > 0.2:
        report.append("High EMI burden impacting financial flexibility.")

    if savings < 20000:
        report.append("Low savings buffer reduces financial stability.")

    if surplus < 15000:
        report.append("Low monthly surplus indicates tight cash flow.")

    if score < 580:
        report.append("High risk borrower profile.")

    return report

# =========================
# NAVIGATION (NOW SAFE)
# =========================
page = st.sidebar.radio("Navigation", ["Home", "Overview", "Prediction", "History"])

# =========================
# HOME
# =========================
if page == "Home":
    st.title("AI Credit Intelligence System")

    st.markdown("""
    - Credit score prediction  
    - Loan approval system  
    - Financial analysis  
    - Smart recommendations  
    """)

# =========================
# OVERVIEW
# =========================
elif page == "Overview":
    st.title("System Overview")

    st.markdown("""
    1. User enters data  
    2. Features are generated  
    3. ML model predicts score  
    4. Decision engine evaluates risk  
    """)

# =========================
# PREDICTION
# =========================
elif page == "Prediction":

    st.title("Credit Evaluation")

    age = st.number_input("Age", 18, 100, 30)
    exp = st.number_input("Experience", 0, max(0, age-18), 3)
    p1 = st.number_input("Projects Last 1M", 0, 100, 2)
    p6 = st.number_input("Projects Last 6M", 0, 500, 20)
    upcoming = st.number_input("Upcoming Projects", 0, 100, 3)

    avg_val = st.number_input("Avg Project Value", 0.0, 1e7, 10000.0)
    monthly_exp = st.number_input("Monthly Expenses", 0.0, 1e7, 40000.0)
    inflow = st.number_input("Monthly Income", 0.0, 1e7, 60000.0)
    outflow = st.number_input("Monthly Outflow", 0.0, 1e7, 45000.0)
    savings = st.number_input("Savings", 0.0, 1e7, 10000.0)

    loans = st.number_input("Loans", 0, 10, 1)
    emi = st.number_input("EMI", 0.0, 1e7, 10000.0)

    category = st.selectbox("Category", encoder.categories_[0])
    platform = st.selectbox("Platform", encoder.categories_[1])

    if st.button("Generate Report"):

        data = {
            "age": age,
            "exp": exp,
            "p1": p1,
            "p6": p6,
            "upcoming": upcoming,
            "avg_val": avg_val,
            "monthly_exp": monthly_exp,
            "inflow": inflow,
            "outflow": outflow,
            "savings": savings,
            "loans": loans,
            "emi": emi,
            "category": category,
            "platform": platform
        }

        row = build_input(data)

        for col in num_cols:
            if col not in row.columns:
                row[col] = 0

        for col in cat_cols:
            if col not in row.columns:
                row[col] = "Other"

        X = np.concatenate([
            scaler.transform(row[num_cols]),
            encoder.transform(row[cat_cols])
        ], axis=1)

        score = float(np.clip(model.predict(X)[0], 300, 850))

        if score < 580:
            decision = "REJECTED"
        elif score < 750:
            decision = "CONDITIONAL APPROVAL"
        else:
            decision = "APPROVED"

        report = generate_report(row, score)

        st.markdown("---")
        st.subheader("Final Credit Report")

        st.write(f"Score: {round(score,2)}")
        st.write(f"Decision: {decision}")

        st.write("Analysis:")
        for r in report:
            st.write("-", r)

        # history
        st.session_state.history.append({
            "time": datetime.now(),
            "score": score,
            "decision": decision
        })

        # download
        df_report = pd.DataFrame({
            "Score": [score],
            "Decision": [decision],
            "Analysis": [" | ".join(report)]
        })

        st.download_button(
            "Download Report",
            df_report.to_csv(index=False),
            "report.csv",
            "text/csv"
        )

# =========================
# HISTORY
# =========================
elif page == "History":

    st.title("Prediction History")

    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.write("No history yet")