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

# =========================
# PAGE CONFIG
# =========================
#st.set_page_config(page_title="AI Credit Intelligence System", layout="wide")

# =========================
# LOGIN SYSTEM
# =========================
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# =========================
# TESSERACT PATH (WINDOWS)
# Uncomment and change path if needed
# =========================


#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# LOAD FILES
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# FUNCTIONS
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
        "EMI_Income_Ratio": data["emi"] / income if income > 0 else 0,
        "Income_Stability": data["p6"] / 6 if data["p6"] > 0 else 0,
        "Experience_Score": data["exp"] / data["age"] if data["age"] > 0 else 0,
        "Loan_Burden": data["loans"] * data["emi"],
        "Income_per_Project": income / data["p6"] if data["p6"] > 0 else income
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

    if not report:
        report.append("Financial profile looks reasonably stable.")

    return report

def clean_amount(text):
    if text is None:
        return None
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("₹", "")
    text = text.replace("Rs.", "")
    text = text.replace("Rs", "")
    text = re.sub(r"[^\d\.\-]", "", text)
    if text == "":
        return None
    try:
        return float(text)
    except:
        return None

def preprocess_image_for_ocr(pil_img):
    img = pil_img.convert("L")  # grayscale
    # simple thresholding
    img = img.point(lambda x: 0 if x < 150 else 255, "1")
    return img

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(image, lang="eng")
    return text

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # First try normal PDF text extraction
        page_text = page.get_text("text", sort=True)

        # If too little text, fallback to OCR on rendered image
        if len(page_text.strip()) < 30:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = preprocess_image_for_ocr(img)
            page_text = pytesseract.image_to_string(img, lang="eng")

        text += "\n" + page_text

    doc.close()
    return text

def extract_text_from_statement(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(uploaded_file)

    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)

    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return df.to_string(index=False)

    else:
        return ""

def parse_statement_text(text):
    """
    Basic OCR text parser.
    Tries to estimate:
    - inflow
    - outflow
    - savings (closing balance)
    """
    inflow = 0.0
    outflow = 0.0
    savings = 0.0

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    amount_pattern = r"[-+]?\d[\d,]*\.?\d*"

    for line in lines:
        lower_line = line.lower()

        amounts = re.findall(amount_pattern, line)
        parsed_amounts = [clean_amount(a) for a in amounts]
        parsed_amounts = [a for a in parsed_amounts if a is not None and a > 0]

        if not parsed_amounts:
            continue

        # Closing balance / available balance / balance
        if ("closing balance" in lower_line or
            "available balance" in lower_line or
            "avl bal" in lower_line or
            "balance" in lower_line):
            savings = max(savings, max(parsed_amounts))

        # Credit-like lines
        elif ("credit" in lower_line or "cr" in lower_line or "deposit" in lower_line or
              "salary" in lower_line or "received" in lower_line):
            inflow += max(parsed_amounts)

        # Debit-like lines
        elif ("debit" in lower_line or "dr" in lower_line or "withdrawal" in lower_line or
              "upi" in lower_line or "purchase" in lower_line or "bill" in lower_line or
              "emi" in lower_line or "payment" in lower_line or "atm" in lower_line):
            outflow += max(parsed_amounts)

    # fallback if OCR didn't identify keywords well
    if inflow == 0 and outflow == 0:
        all_amounts = re.findall(amount_pattern, text)
        vals = [clean_amount(a) for a in all_amounts]
        vals = [v for v in vals if v is not None and v > 0]

        if len(vals) >= 2:
            # rough fallback
            inflow = sum(vals[: len(vals)//2]) * 0.3
            outflow = sum(vals[: len(vals)//2]) * 0.2
            savings = max(vals)

    monthly_exp = outflow

    return {
        "inflow": round(float(inflow), 2),
        "outflow": round(float(outflow), 2),
        "monthly_exp": round(float(monthly_exp), 2),
        "savings": round(float(savings), 2)
    }

def predict_credit_score(data):
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
    return row, score, decision, report
# =========================
# LOGIN CONTROL
# =========================
if not st.session_state.logged_in:
    login_page()
    st.stop()

# =========================
# SIDEBAR + LOGOUT
# =========================
st.sidebar.success("Logged in")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
# =========================
# NAVIGATION
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
    - OCR bank statement upload
    """)

# =========================
# OVERVIEW
# =========================
elif page == "Overview":
    st.title("System Overview")
    st.markdown("""
    1. User enters data manually or uploads bank statement  
    2. OCR reads scanned statement/image/PDF  
    3. Financial values are estimated  
    4. ML model predicts score  
    5. Decision engine evaluates risk  
    """)

# =========================
# PREDICTION
# =========================
elif page == "Prediction":
    st.title("Credit Evaluation")

    mode = st.radio("Choose Input Mode", ["Manual Entry", "Upload Bank Statement (OCR)"])

    age = st.number_input("Age", 18, 100, 30)
    exp = st.number_input("Experience", 0, max(0, age - 18), 3)
    p1 = st.number_input("Projects Last 1M", 0, 100, 2)
    p6 = st.number_input("Projects Last 6M", 0, 500, 20)
    upcoming = st.number_input("Upcoming Projects", 0, 100, 3)

    avg_val = st.number_input("Avg Project Value", 0.0, 1e7, 10000.0)
    loans = st.number_input("Loans", 0, 10, 1)
    emi = st.number_input("EMI", 0.0, 1e7, 10000.0)

    category = st.selectbox("Category", encoder.categories_[0])
    platform = st.selectbox("Platform", encoder.categories_[1])

    if mode == "Manual Entry":
        monthly_exp = st.number_input("Monthly Expenses", 0.0, 1e7, 40000.0)
        inflow = st.number_input("Monthly Income", 0.0, 1e7, 60000.0)
        outflow = st.number_input("Monthly Outflow", 0.0, 1e7, 45000.0)
        savings = st.number_input("Savings", 0.0, 1e7, 10000.0)

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

            row, score, decision, report = predict_credit_score(data)

            st.markdown("---")
            st.subheader("Final Credit Report")
            st.write(f"Score: {round(score, 2)}")
            st.write(f"Decision: {decision}")

            st.write("Analysis:")
            for r in report:
                st.write("-", r)

            st.session_state.history.append({
                "time": datetime.now(),
                "score": score,
                "decision": decision,
                "mode": "Manual"
            })

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

    else:
        uploaded_file = st.file_uploader(
            "Upload bank statement",
            type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx", "xls"]
        )

        if uploaded_file is not None:
            if st.button("Read Statement and Predict"):
                try:
                    text = extract_text_from_statement(uploaded_file)

                    if not text.strip():
                        st.error("Could not extract readable text from the statement.")
                    else:
                        extracted = parse_statement_text(text)

                        st.subheader("OCR Extracted Summary")
                        st.write(f"Monthly Inflow: {extracted['inflow']}")
                        st.write(f"Monthly Outflow: {extracted['outflow']}")
                        st.write(f"Monthly Expenses: {extracted['monthly_exp']}")
                        st.write(f"Savings / Balance: {extracted['savings']}")

                        with st.expander("Show extracted text"):
                            st.text(text[:10000])

                        data = {
                            "age": age,
                            "exp": exp,
                            "p1": p1,
                            "p6": p6,
                            "upcoming": upcoming,
                            "avg_val": avg_val,
                            "monthly_exp": extracted["monthly_exp"],
                            "inflow": extracted["inflow"],
                            "outflow": extracted["outflow"],
                            "savings": extracted["savings"],
                            "loans": loans,
                            "emi": emi,
                            "category": category,
                            "platform": platform
                        }

                        row, score, decision, report = predict_credit_score(data)

                        st.markdown("---")
                        st.subheader("Final Credit Report")
                        st.write(f"Score: {round(score, 2)}")
                        st.write(f"Decision: {decision}")

                        st.write("Analysis:")
                        for r in report:
                            st.write("-", r)

                        st.session_state.history.append({
                            "time": datetime.now(),
                            "score": score,
                            "decision": decision,
                            "mode": "OCR Statement"
                        })

                        df_report = pd.DataFrame({
                            "Score": [score],
                            "Decision": [decision],
                            "Analysis": [" | ".join(report)],
                            "OCR_Inflow": [extracted["inflow"]],
                            "OCR_Outflow": [extracted["outflow"]],
                            "OCR_Savings": [extracted["savings"]]
                        })

                        st.download_button(
                            "Download Report",
                            df_report.to_csv(index=False),
                            "ocr_statement_report.csv",
                            "text/csv"
                        )

                except Exception as e:
                    st.error(f"Error while processing statement: {e}")

# =========================
# HISTORY
# =========================
elif page == "History":
    st.title("Prediction History")

    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.write("No history yet")