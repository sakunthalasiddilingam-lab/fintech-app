import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AI Credit Intelligence System", layout="wide")

# =========================================================
# LOGIN CREDENTIALS
# =========================================================
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# =========================================================
# SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================================================
# LOAD MODEL FILES
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    num_cols = joblib.load("num_cols.pkl")
    cat_cols = joblib.load("cat_cols.pkl")
    return model, scaler, encoder, num_cols, cat_cols

model, scaler, encoder, num_cols, cat_cols = load_artifacts()

# =========================================================
# LOGIN PAGE
# =========================================================
def login_page():
    st.markdown(
        """
        <h1 style='text-align:center;'>🔐 Login Page</h1>
        <h3 style='text-align:center;'>AI Credit Intelligence System</h3>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

# =========================================================
# LOGOUT
# =========================================================
def logout():
    st.session_state.logged_in = False
    st.rerun()

# =========================================================
# OCR HELPERS
# =========================================================
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        text += page.get_text()

    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

def extract_financial_info(text):
    income = 0
    expenses = 0

    income_patterns = [
        r"salary[:\s]*₹?([\d,]+)",
        r"income[:\s]*₹?([\d,]+)",
        r"credited[:\s]*₹?([\d,]+)",
        r"deposit[:\s]*₹?([\d,]+)"
    ]

    expense_patterns = [
        r"debited[:\s]*₹?([\d,]+)",
        r"withdrawal[:\s]*₹?([\d,]+)",
        r"expense[:\s]*₹?([\d,]+)",
        r"payment[:\s]*₹?([\d,]+)"
    ]

    for pattern in income_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        income += sum(float(x.replace(",", "")) for x in matches if x)

    for pattern in expense_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        expenses += sum(float(x.replace(",", "")) for x in matches if x)

    return income, expenses

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_credit_score(input_df):
    X_num = input_df[num_cols].copy()
    X_cat = input_df[cat_cols].copy()

    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = encoder.transform(X_cat)

    X = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
    score = model.predict(X)[0]

    score = max(300, min(850, score))
    return round(score, 2)

# =========================================================
# SCORE INTERPRETATION
# =========================================================
def get_score_status(score):
    if score >= 750:
        return "Excellent", "Loan Approval Highly Likely"
    elif score >= 700:
        return "Good", "Loan Approval Likely"
    elif score >= 650:
        return "Fair", "Loan Approval Possible"
    else:
        return "Poor", "Loan Approval Difficult"

# =========================================================
# MAIN APP
# =========================================================
def main_app():
    st.sidebar.success("Logged in")
    if st.sidebar.button("Logout"):
        logout()

    page = st.sidebar.radio("Navigation", ["Home", "Overview", "Prediction", "History"])

    if page == "Home":
        st.title("AI Credit Intelligence System")
        st.markdown("""
        - Credit score prediction  
        - Loan approval system  
        - Financial analysis  
        - Smart recommendations  
        - OCR bank statement upload  
        """)

    elif page == "Overview":
        st.header("Overview")
        st.write("This system predicts user credit score using financial and professional details.")
        st.write("It also supports OCR-based bank statement analysis.")

    elif page == "Prediction":
        st.header("Credit Score Prediction")

        st.subheader("Enter User Details")

        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        work_experience = st.number_input("Work Experience", min_value=0, max_value=50, value=2)
        projects_last_1m = st.number_input("Projects Last 1 Month", min_value=0, value=2)
        projects_last_6m = st.number_input("Projects Last 6 Months", min_value=0, value=10)
        upcoming_projects = st.number_input("Upcoming Projects", min_value=0, value=1)
        avg_project_value = st.number_input("Average Project Value", min_value=0.0, value=10000.0)
        cash_inflow = st.number_input("Monthly Cash Inflow", min_value=0.0, value=50000.0)
        monthly_expenses = st.number_input("Monthly Expenses", min_value=0.0, value=20000.0)
        savings = st.number_input("Savings", min_value=0.0, value=100000.0)
        existing_loans = st.number_input("Existing Loans", min_value=0, value=0)

        emi_amount = 0.0
        if existing_loans > 0:
            emi_amount = st.number_input("EMI Amount", min_value=0.0, value=5000.0)

        freelancer_category = st.selectbox(
            "Freelancer Category",
            [
                "Content Writer", "Graphic Designer", "Web Developer", "App Developer",
                "Digital Marketer", "Video Editor", "Photographer", "UI UX Designer",
                "Data Scientist", "AI Engineer"
            ]
        )

        platform = st.selectbox(
            "Platform",
            ["Upwork", "Fiverr", "Freelancer", "Toptal", "LinkedIn", "Direct Clients"]
        )

        uploaded_file = st.file_uploader("Upload Bank Statement (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

        extracted_income = 0
        extracted_expense = 0

        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    extracted_text = extract_text_from_pdf(uploaded_file)
                else:
                    extracted_text = extract_text_from_image(uploaded_file)

                extracted_income, extracted_expense = extract_financial_info(extracted_text)

                st.subheader("OCR Extracted Details")
                st.write(f"Detected Income: ₹{extracted_income:,.2f}")
                st.write(f"Detected Expenses: ₹{extracted_expense:,.2f}")

            except Exception as e:
                st.error(f"Error while processing statement: {e}")

        if st.button("Predict Credit Score"):
            final_income = extracted_income if extracted_income > 0 else cash_inflow
            final_expenses = extracted_expense if extracted_expense > 0 else monthly_expenses

            input_data = {
                "Age": [age],
                "Work_Experience": [work_experience],
                "Projects_Last_1M": [projects_last_1m],
                "Projects_Last_6M": [projects_last_6m],
                "Upcoming_Projects": [upcoming_projects],
                "Avg_Project_Value": [avg_project_value],
                "Cash_Inflow": [final_income],
                "Monthly_Expenses": [final_expenses],
                "Savings": [savings],
                "Existing_Loans": [existing_loans],
                "EMI_Amount": [emi_amount],
                "Freelancer_Category": [freelancer_category],
                "Platform": [platform]
            }

            input_df = pd.DataFrame(input_data)

            try:
                score = predict_credit_score(input_df)
                category, approval_status = get_score_status(score)

                st.subheader("Prediction Result")
                st.success(f"Predicted Credit Score: {score}")
                st.info(f"Score Category: {category}")
                st.warning(f"Loan Status: {approval_status}")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    elif page == "History":
        st.header("History")
        st.write("You can add prediction history storage here later.")

# =========================================================
# APP FLOW
# =========================================================
if st.session_state.logged_in:
    main_app()
else:
    login_page()