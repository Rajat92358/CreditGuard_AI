
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import io

from model import train_model, FEATURES, predict_applicant
from scoring import (
    calculate_dti,
    calculate_risk_score,
    determine_decision,
    explain_drivers,
    generate_recommendation,
)
from file_parser import (
    prepare_dataframe,
    validate_dataframe,
    parse_pdf_application,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL UI
# =========================================================

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#081a3a 0%,#0f2f67 100%);
    }
    section[data-testid="stSidebar"] * { color:#f8fafc; }
    .block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }

    h1 { color:#0b1f44; font-weight:850; }
    h2 { color:#102a56; font-weight:800; }
    h3 { color:#173f7a; font-weight:750; }

    .hero {
        background: linear-gradient(135deg,#071a3d 0%,#0d47a1 58%,#12a8e8 100%);
        border-radius:24px; padding:34px 38px; color:white;
        box-shadow:0 12px 35px rgba(7,26,61,.18); margin-bottom:22px;
    }
    .hero-title { font-size:40px; font-weight:850; letter-spacing:-1px; }
    .hero-subtitle { color:#dff5ff; font-size:17px; margin-top:6px; }
    .pill {
        display:inline-block; padding:7px 12px; border-radius:999px;
        background:rgba(255,255,255,.14); color:white; font-size:12px;
        margin-top:16px; border:1px solid rgba(255,255,255,.2);
    }
    .metric-card {
        background:white; border:1px solid #e2e8f0; border-radius:17px;
        padding:20px; min-height:118px;
        box-shadow:0 5px 18px rgba(15,23,42,.06);
    }
    .metric-title { color:#64748b; font-size:13px; font-weight:700; }
    .metric-value { color:#0b1f44; font-size:28px; font-weight:850; margin-top:7px; }
    .feature-card {
        background:white; border:1px solid #e2e8f0; border-radius:18px;
        padding:23px; min-height:185px; box-shadow:0 4px 16px rgba(15,23,42,.05);
    }
    .feature-icon { font-size:30px; }
    .feature-title { color:#102a56; font-size:19px; font-weight:800; margin-top:9px; }
    .feature-text { color:#64748b; line-height:1.55; font-size:14px; }
    .result-card {
        background:white; border:1px solid #e2e8f0; border-radius:20px;
        padding:25px; box-shadow:0 6px 22px rgba(15,23,42,.07);
    }
    .driver-positive { background:#ecfdf5; border-left:5px solid #10b981; padding:13px 15px; border-radius:10px; margin-bottom:9px; color:#14532d; }
    .driver-neutral { background:#fffbeb; border-left:5px solid #f59e0b; padding:13px 15px; border-radius:10px; margin-bottom:9px; color:#78350f; }
    .driver-negative { background:#fef2f2; border-left:5px solid #ef4444; padding:13px 15px; border-radius:10px; margin-bottom:9px; color:#7f1d1d; }
    .info-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:13px; padding:16px; color:#163b73; }
    .section-label { color:#2563eb; font-size:12px; font-weight:800; letter-spacing:1px; text-transform:uppercase; }
    .decision-banner { border-radius:16px; padding:18px 20px; margin:10px 0 18px; font-weight:800; font-size:20px; }
    .footer { text-align:center; color:#64748b; padding:30px; font-size:13px; }
    [data-testid="stMetric"] {
        background:white; border:1px solid #e2e8f0; padding:12px 15px;
        border-radius:14px; box-shadow:0 3px 12px rgba(15,23,42,.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODEL
# =========================================================


@st.cache_resource
def load_model():
    return train_model()


model, training_data, model_accuracy, model_auc = load_model()

# =========================================================
# SESSION STATE
# =========================================================

if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []

if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================================================
# HELPERS
# =========================================================


def money(value):
    return f"₹{float(value):,.0f}"


def calculate_emi(principal, annual_rate, months):
    """Standard reducing-balance EMI estimate."""
    principal = float(principal)
    months = int(months)
    if principal <= 0 or months <= 0:
        return 0.0
    monthly_rate = float(annual_rate) / 12 / 100
    if monthly_rate == 0:
        return principal / months
    return principal * monthly_rate * (1 + monthly_rate) ** months / (
        (1 + monthly_rate) ** months - 1
    )


def assess_applicant(applicant, source="Manual"):
    probability = predict_applicant(model, applicant)
    eligibility_score = calculate_risk_score(probability)
    dti = calculate_dti(applicant["income"], applicant["monthly_debt"])
    decision, risk_category = determine_decision(
        eligibility_score, applicant["credit_score"], dti
    )
    positive, neutral, negative = explain_drivers(
        applicant["income"],
        applicant["employment_years"],
        applicant["credit_score"],
        applicant["loan_amount"],
        applicant["existing_loans"],
        applicant["monthly_debt"],
    )
    result = applicant.copy()
    result.update({
        "eligibility_score": eligibility_score,
        "model_probability": round(probability * 100, 2),
        "dti": round(dti, 2),
        "risk_category": risk_category,
        "decision": decision,
        "positive_drivers": " | ".join(positive),
        "neutral_drivers": " | ".join(neutral),
        "negative_drivers": " | ".join(negative),
        "recommendation": generate_recommendation(decision),
        "source": source,
        "assessment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return result


def show_score_gauge(score):
    if score >= 70:
        gauge_color = "#10b981"
    elif score >= 50:
        gauge_color = "#f59e0b"
    else:
        gauge_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 42, "color": "#0b1f44"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 50], "color": "#fee2e2"},
                {"range": [50, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#d1fae5"},
            ],
        },
    ))
    fig.update_layout(height=285, margin=dict(l=20, r=20, t=20, b=5))
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


def result_to_text(result):
    return f"""
CREDITGUARD AI - APPLICANT ASSESSMENT REPORT

Assessment Time: {result['assessment_time']}
Source: {result['source']}

FINANCIAL PROFILE
Monthly Income: {money(result['income'])}
Employment Experience: {result['employment_years']:.1f} years
Credit Score: {result['credit_score']:.0f}
Existing Loans: {result['existing_loans']:.0f}
Monthly Debt: {money(result['monthly_debt'])}
Requested Loan: {money(result['loan_amount'])}
Loan Term: {result['loan_term']:.0f} months

ASSESSMENT
Eligibility Score: {result['eligibility_score']}/100
Model Probability: {result['model_probability']:.2f}%
DTI: {result['dti']:.2f}%
Risk Category: {result['risk_category']}
Decision: {result['decision']}

RECOMMENDATION
{result['recommendation']}

POSITIVE DRIVERS
{result['positive_drivers'] or 'None'}

NEUTRAL FACTORS
{result['neutral_drivers'] or 'None'}

RISK FACTORS
{result['negative_drivers'] or 'None'}

DISCLAIMER
Academic decision-support prototype using synthetic data.
This result is not a final lending decision.
Human review and verified information would be required in a real lending environment.
"""


def display_result(result):
    st.markdown("---")
    st.markdown("## 🎯 CreditGuard Assessment")

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### Eligibility Score")
        show_score_gauge(result["eligibility_score"])
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        decision = result["decision"]
        if decision == "APPROVE":
            st.success("## ✅ APPROVE")
        elif decision == "REFER":
            st.warning("## ⚠️ REFER")
        else:
            st.error("## ❌ REJECT")
        st.markdown(f"### {result['risk_category']}")
        st.write(result["recommendation"])
        st.markdown(
            f"**Source:** {result['source']}  \n"
            f"**Assessment:** {result['assessment_time']}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📌 Financial Snapshot")
    cols = st.columns(5)
    snapshot = [
        ("Credit Score", f"{result['credit_score']:.0f}"),
        ("DTI", f"{result['dti']:.1f}%"),
        ("Monthly Income", money(result["income"])),
        ("Loan Amount", money(result["loan_amount"])),
        ("Existing Loans", f"{result['existing_loans']:.0f}"),
    ]
    for col, (title, value) in zip(cols, snapshot):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 🧠 Why did the model reach this result?")
    c1, c2, c3 = st.columns(3)
    groups = [
        ("🟢 Positive Drivers",
         result["positive_drivers"], "driver-positive", "✓"),
        ("🟡 Neutral Factors",
         result["neutral_drivers"], "driver-neutral", "•"),
        ("🔴 Risk Factors", result["negative_drivers"], "driver-negative", "!"),
    ]
    for col, (title, values, css, symbol) in zip([c1, c2, c3], groups):
        with col:
            st.markdown(f"#### {title}")
            if values:
                for item in values.split(" | "):
                    st.markdown(
                        f'<div class="{css}">{symbol} {item}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("None identified.")

    st.markdown("### 💰 Affordability Snapshot")
    rate = st.slider(
        "Illustrative annual interest rate for EMI estimate",
        5.0, 20.0, 10.0, 0.5,
        key=f"rate_{result['assessment_time']}",
    )
    emi = calculate_emi(result["loan_amount"], rate, result["loan_term"])
    total_outflow = emi * result["loan_term"]
    emi_plus_debt = emi + result["monthly_debt"]
    affordability_dti = (emi_plus_debt / max(result["income"], 1)) * 100

    a, b, c, d = st.columns(4)
    a.metric("Estimated EMI", money(emi))
    b.metric("Total Repayment", money(total_outflow))
    c.metric("Debt + EMI", money(emi_plus_debt))
    d.metric("Post-EMI DTI", f"{affordability_dti:.1f}%")

    st.download_button(
        "📥 Download Assessment Report",
        data=result_to_text(result).encode("utf-8"),
        file_name="creditguard_assessment_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

# =========================================================
# SIDEBAR
# =========================================================


logo_path = Path("assets/creditguard_logo.png")
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)
else:
    st.sidebar.markdown(
        "<div style='text-align:center;font-size:44px'>🛡️</div>"
        "<div style='text-align:center;font-size:22px;font-weight:800'>CreditGuard AI</div>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    "<div style='text-align:center;color:#b9d8ff;font-size:11px;letter-spacing:1px'>"
    "AI CREDIT INTELLIGENCE</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Overview",
        "👤 Applicant Assessment",
        "🧪 What-If Simulator",
        "📂 Bulk Assessment",
        "📄 PDF Assessment",
        "📊 Risk Dashboard",
        "🕘 Assessment History",
        "🤖 Model Intelligence",
        "ℹ️ About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='background:#102a56;padding:14px;border-radius:12px;font-size:12px;"
    "color:#dbeafe'><b>Academic Prototype</b><br><br>"
    "Synthetic data • ML scoring • Human review required</div>",
    unsafe_allow_html=True,
)

# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🛡️ CreditGuard AI</div>
            <div class="hero-subtitle">Smarter Pre-Screening. Better Credit Decisions.</div>
            <div class="pill">AI-POWERED LOAN ELIGIBILITY & CREDIT-RISK PRE-SCREENING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "A machine-learning decision-support prototype that evaluates applicant "
        "financial indicators and produces an explainable pre-screening result."
    )

    st.markdown("### 📈 System Overview")
    cols = st.columns(5)
    cards = [
        ("Training Records", f"{len(training_data):,}"),
        ("Model Accuracy", f"{model_accuracy * 100:.1f}%"),
        ("ROC-AUC", f"{model_auc:.3f}"),
        ("Risk Features", str(len(FEATURES))),
        ("Assessments", str(len(st.session_state.assessment_history))),
    ]
    for col, (title, value) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### ⚡ Explore CreditGuard")
    cols = st.columns(4)
    features = [
        ("👤", "Applicant Assessment",
         "Assess one applicant with an instant score and explanation."),
        ("🧪", "What-If Simulator",
         "Test how changes in income, debt or credit score affect eligibility."),
        ("📂", "Bulk Assessment",
         "Assess hundreds or thousands of synthetic applicants from CSV."),
        ("📊", "Risk Dashboard",
         "Explore portfolio-level decisions, distributions and relationships."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        '<div class="info-box"><b>Decision Pipeline</b><br><br>'
        'Applicant Data → Validation → Logistic Regression → Eligibility Score → '
        'Credit/DTI Rules → Approve / Refer / Reject → Explainable Drivers</div>',
        unsafe_allow_html=True,
    )

# =========================================================
# MANUAL ASSESSMENT
# =========================================================

elif page == "👤 Applicant Assessment":
    st.title("👤 Applicant Assessment")
    st.caption(
        "Enter a synthetic applicant profile and run an explainable pre-screening assessment.")

    with st.form("manual_assessment_form"):
        st.markdown("### 💰 Applicant Financial Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.number_input(
                "Monthly Income (₹)", min_value=1.0, value=50000.0, step=1000.0)
        with c2:
            employment_years = st.number_input(
                "Employment Experience (years)", 0.0, 50.0, 5.0, 0.5)
        with c3:
            credit_score = st.number_input(
                "Credit / CIBIL Score", 300, 850, 700, 1)

        st.markdown("### 🏦 Loan Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amount = st.number_input(
                "Requested Loan Amount (₹)", min_value=1.0, value=500000.0, step=10000.0)
        with c2:
            loan_term = st.selectbox("Loan Term", [
                                     12, 24, 36, 48, 60, 72, 84], index=4, format_func=lambda x: f"{x} months")
        with c3:
            existing_loans = st.number_input("Existing Loans", 0, 20, 1, 1)

        monthly_debt = st.number_input(
            "Existing Monthly Debt (₹)", min_value=0.0, value=10000.0, step=1000.0)

        st.markdown("### 🔎 Live Input Checks")
        dti_preview = calculate_dti(income, monthly_debt)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current DTI", f"{dti_preview:.1f}%")
        c2.metric("Loan / Monthly Income",
                  f"{loan_amount / max(income, 1):.1f}×")
        c3.metric("Credit Band", "Strong" if credit_score >=
                  750 else "Moderate" if credit_score >= 650 else "Low")

        submitted = st.form_submit_button(
            "🔍 Run Credit Assessment", use_container_width=True)

    if submitted:
        if dti_preview > 100:
            st.error(
                "Monthly debt cannot exceed 100% of monthly income in this prototype.")
        else:
            applicant = {
                "income": income,
                "employment_years": employment_years,
                "credit_score": credit_score,
                "loan_amount": loan_amount,
                "existing_loans": existing_loans,
                "monthly_debt": monthly_debt,
                "loan_term": loan_term,
            }
            result = assess_applicant(applicant, "Manual")
            st.session_state.assessment_history.append(result)
            st.session_state.last_result = result
            display_result(result)

# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

elif page == "🧪 What-If Simulator":
    st.title("🧪 What-If Credit Simulator")
    st.write("Change one or more financial variables and see how the simulated eligibility result responds.")

    if st.session_state.last_result:
        default = st.session_state.last_result
    else:
        default = {
            "income": 60000, "employment_years": 5, "credit_score": 700,
            "loan_amount": 600000, "existing_loans": 1, "monthly_debt": 10000, "loan_term": 60
        }

    c1, c2 = st.columns(2)
    with c1:
        income_w = st.slider("Monthly Income (₹)", 18000,
                             250000, int(default["income"]), 1000)
        credit_w = st.slider("Credit Score", 300, 850,
                             int(default["credit_score"]), 1)
        employment_w = st.slider("Employment Years", 0.0, 30.0, float(
            default["employment_years"]), 0.5)
        existing_w = st.slider("Existing Loans", 0, 6,
                               int(default["existing_loans"]), 1)
    with c2:
        loan_w = st.slider("Loan Amount (₹)", 50000, 3000000,
                           int(default["loan_amount"]), 10000)
        debt_w = st.slider("Monthly Debt (₹)", 0, 150000, min(
            int(default["monthly_debt"]), 150000), 1000)
        term_w = st.selectbox("Loan Term", [12, 24, 36, 48, 60, 72, 84], index=[12, 24, 36, 48, 60, 72, 84].index(
            int(default["loan_term"])) if int(default["loan_term"]) in [12, 24, 36, 48, 60, 72, 84] else 4)

    scenario = {
        "income": income_w,
        "employment_years": employment_w,
        "credit_score": credit_w,
        "loan_amount": loan_w,
        "existing_loans": existing_w,
        "monthly_debt": debt_w,
        "loan_term": term_w,
    }
    scenario_result = assess_applicant(scenario, "What-If Simulator")

    st.markdown("### Scenario Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eligibility Score",
              f"{scenario_result['eligibility_score']:.1f}/100")
    c2.metric("Decision", scenario_result["decision"])
    c3.metric("DTI", f"{scenario_result['dti']:.1f}%")
    c4.metric("Risk", scenario_result["risk_category"])

    show_score_gauge(scenario_result["eligibility_score"])

    if st.session_state.last_result:
        baseline = st.session_state.last_result["eligibility_score"]
        delta = scenario_result["eligibility_score"] - baseline
        st.metric("Change vs Last Assessment", f"{delta:+.1f} points")

        compare = pd.DataFrame({
            "Scenario": ["Previous Assessment", "Current What-If"],
            "Eligibility Score": [baseline, scenario_result["eligibility_score"]],
        })
        fig = px.bar(compare, x="Scenario", y="Eligibility Score", range_y=[0, 100],
                     title="Before vs What-If Eligibility Score", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

    st.info("This simulator demonstrates model sensitivity. It is not a recommendation to change a real applicant's information.")

# =========================================================
# BULK ASSESSMENT
# =========================================================

elif page == "📂 Bulk Assessment":
    st.title("📂 Bulk Applicant Assessment")
    st.write("Upload a CSV containing multiple applicants.")

    uploaded_csv = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_csv is not None:
        try:
            df = prepare_dataframe(pd.read_csv(uploaded_csv))
            st.success(f"✓ {len(df):,} applicants loaded")

            required_columns = ["income", "employment_years", "credit_score", "loan_amount",
                                "existing_loans", "monthly_debt", "loan_term"]
            missing = validate_dataframe(df, required_columns)

            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                st.success("✓ All required fields detected")
                st.dataframe(
                    df.head(10), use_container_width=True, hide_index=True)

                if st.button("🚀 Assess All Applicants", use_container_width=True):
                    results = []
                    progress = st.progress(0)

                    for i, row in df.iterrows():
                        try:
                            applicant = {k: float(row[k])
                                         for k in required_columns}
                            result = assess_applicant(applicant, "CSV")
                            if "applicant_id" in df.columns:
                                result["applicant_id"] = row["applicant_id"]
                            results.append(result)
                        except Exception as e:
                            st.warning(f"Row {i+1} skipped: {e}")
                        progress.progress((i + 1) / len(df))

                    if results:
                        results_df = pd.DataFrame(results)
                        st.session_state.bulk_results = results_df
                        st.success(
                            f"✓ Assessment completed for {len(results_df):,} applicants")

                        counts = results_df["decision"].value_counts()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total", len(results_df))
                        c2.metric("Approve", int(counts.get("APPROVE", 0)))
                        c3.metric("Refer", int(counts.get("REFER", 0)))
                        c4.metric("Reject", int(counts.get("REJECT", 0)))

                        st.markdown("### 📊 Bulk Risk Overview")
                        chart = px.pie(
                            results_df, names="decision", hole=.5,
                            title="Decision Distribution"
                        )
                        st.plotly_chart(chart, use_container_width=True)

                        st.dataframe(
                            results_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "📥 Download Results CSV",
                            results_df.to_csv(index=False).encode("utf-8"),
                            "creditguard_results.csv",
                            "text/csv",
                            use_container_width=True,
                        )
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")

# =========================================================
# PDF ASSESSMENT
# =========================================================

elif page == "📄 PDF Assessment":
    st.title("📄 PDF Applicant Assessment")
    st.write(
        "Upload a text-based applicant PDF. Always verify extracted values before assessment.")

    uploaded_pdf = st.file_uploader("Choose PDF file", type=["pdf"])

    if uploaded_pdf is not None:
        try:
            extracted_text, extracted_data = parse_pdf_application(
                uploaded_pdf)
            st.success("✓ PDF processed successfully")

            with st.expander("View Extracted Text"):
                st.text(extracted_text)

            st.markdown("### 🔍 Verify Extracted Information")
            c1, c2 = st.columns(2)
            with c1:
                pdf_income = st.number_input(
                    "Monthly Income (₹)", min_value=0.0, value=float(extracted_data["income"] or 0))
                pdf_employment = st.number_input("Employment Years", min_value=0.0, value=float(
                    extracted_data["employment_years"] or 0))
                pdf_credit = st.number_input("Credit Score", min_value=300, max_value=850, value=int(
                    extracted_data["credit_score"] or 650))
                pdf_loan = st.number_input("Loan Amount (₹)", min_value=0.0, value=float(
                    extracted_data["loan_amount"] or 0))
            with c2:
                pdf_existing = st.number_input("Existing Loans", min_value=0, value=int(
                    extracted_data["existing_loans"] or 0))
                pdf_debt = st.number_input("Monthly Debt (₹)", min_value=0.0, value=float(
                    extracted_data["monthly_debt"] or 0))
                pdf_term = st.selectbox("Loan Term", [
                                        12, 24, 36, 48, 60, 72, 84], index=4, format_func=lambda x: f"{x} months")

            if st.button("🔍 Assess Verified PDF Applicant", use_container_width=True):
                if pdf_income <= 0 or pdf_loan <= 0:
                    st.error(
                        "Please verify and enter valid income and loan amount.")
                else:
                    applicant = {
                        "income": pdf_income, "employment_years": pdf_employment,
                        "credit_score": pdf_credit, "loan_amount": pdf_loan,
                        "existing_loans": pdf_existing, "monthly_debt": pdf_debt,
                        "loan_term": pdf_term
                    }
                    result = assess_applicant(applicant, "PDF")
                    st.session_state.assessment_history.append(result)
                    st.session_state.last_result = result
                    display_result(result)
        except Exception as e:
            st.error(f"Could not process PDF: {e}")
            st.info(
                "This version supports text-based PDFs. Scanned PDFs may require OCR.")

# =========================================================
# RISK DASHBOARD
# =========================================================

elif page == "📊 Risk Dashboard":
    st.title("📊 Risk Intelligence Dashboard")

    if st.session_state.bulk_results is not None:
        dashboard_df = st.session_state.bulk_results.copy()
    elif st.session_state.assessment_history:
        dashboard_df = pd.DataFrame(st.session_state.assessment_history)
    else:
        dashboard_df = None

    if dashboard_df is None or dashboard_df.empty:
        st.info(
            "Complete an assessment or bulk assessment first to populate the dashboard.")
    else:
        st.markdown("### 🎛️ Portfolio Filters")
        c1, c2, c3 = st.columns(3)
        with c1:
            decisions = st.multiselect("Decision", ["APPROVE", "REFER", "REJECT"], default=[
                                       "APPROVE", "REFER", "REJECT"])
        with c2:
            min_score = st.slider("Minimum Eligibility Score", 0, 100, 0)
        with c3:
            max_dti = st.slider("Maximum DTI (%)", 0, 100, 100)

        filtered = dashboard_df[
            dashboard_df["decision"].isin(decisions)
            & (dashboard_df["eligibility_score"] >= min_score)
            & (dashboard_df["dti"] <= max_dti)
        ].copy()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Applicants", len(filtered))
        c2.metric("Avg Score", f"{filtered['eligibility_score'].mean():.1f}" if len(
            filtered) else "—")
        c3.metric("Avg Credit", f"{filtered['credit_score'].mean():.0f}" if len(
            filtered) else "—")
        c4.metric("Avg DTI", f"{filtered['dti'].mean():.1f}%" if len(
            filtered) else "—")
        c5.metric("Avg Income", money(
            filtered["income"].mean()) if len(filtered) else "—")

        if len(filtered):
            c1, c2 = st.columns(2)
            with c1:
                counts = filtered["decision"].value_counts().reset_index()
                counts.columns = ["Decision", "Count"]
                fig = px.pie(counts, names="Decision", values="Count",
                             hole=.5, title="Decision Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(filtered, x="eligibility_score", nbins=20,
                                   title="Eligibility Score Distribution")
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(filtered, x="credit_score", y="eligibility_score",
                                 color="decision", hover_data=["income", "loan_amount", "dti"],
                                 title="Credit Score vs Eligibility Score")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(filtered, x="income", y="loan_amount",
                                 color="risk_category", hover_data=["credit_score", "dti"],
                                 title="Income vs Loan Amount")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📋 Filtered Portfolio")
            st.dataframe(filtered, use_container_width=True, hide_index=True)

# =========================================================
# HISTORY
# =========================================================

elif page == "🕘 Assessment History":
    st.title("🕘 Assessment History")

    if not st.session_state.assessment_history:
        st.info("No individual assessments have been completed in this session yet.")
    else:
        history_df = pd.DataFrame(st.session_state.assessment_history)
        display_cols = [
            c for c in ["assessment_time", "source", "eligibility_score", "risk_category",
                        "decision", "credit_score", "dti", "income", "loan_amount"]
            if c in history_df.columns
        ]
        st.dataframe(history_df[display_cols],
                     use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download History",
            history_df.to_csv(index=False).encode("utf-8"),
            "creditguard_assessment_history.csv",
            "text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear Session History"):
            st.session_state.assessment_history = []
            st.session_state.last_result = None
            st.rerun()

# =========================================================
# MODEL INTELLIGENCE
# =========================================================

elif page == "🤖 Model Intelligence":
    st.title("🤖 Model Intelligence")
    st.write("CreditGuard AI uses Logistic Regression to estimate the probability of a positive simulated outcome.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{model_accuracy * 100:.2f}%")
    c2.metric("ROC-AUC", f"{model_auc:.3f}")
    c3.metric("Training Records", f"{len(training_data):,}")

    st.markdown("### 🧩 Model Features")
    feature_names = {
        "income": "Monthly Income", "employment_years": "Employment Experience",
        "credit_score": "Credit Score", "loan_amount": "Requested Loan Amount",
        "existing_loans": "Existing Loans", "monthly_debt": "Monthly Debt",
        "loan_term": "Loan Term"
    }
    feature_df = pd.DataFrame({
        "Variable": FEATURES,
        "Business Meaning": [feature_names[x] for x in FEATURES],
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.markdown("### 🔄 Decision Pipeline")
    st.markdown(
        '<div class="info-box"><b>Applicant Data</b> → <b>Validation</b> → '
        '<b>Logistic Regression</b> → <b>Eligibility Score</b> → '
        '<b>Credit Score + DTI Rules</b> → <b>Decision</b> → <b>Explanation</b></div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "Model performance is based on synthetic academic data and should not be interpreted as real-world lending performance."
    )

# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About CreditGuard AI")
    st.markdown(
        """
        ## CreditGuard AI

        CreditGuard AI is an academic loan eligibility and credit-risk
        pre-screening prototype.

        ### Technology Stack
        - Python
        - Streamlit
        - Pandas / NumPy
        - Scikit-learn
        - Plotly
        - PyPDF
        - Logistic Regression

        ### Core Capabilities
        ✓ Manual assessment  
        ✓ What-if simulation  
        ✓ Bulk CSV assessment  
        ✓ PDF information extraction  
        ✓ Eligibility scoring  
        ✓ Risk categorisation  
        ✓ Approve / Refer / Reject  
        ✓ Explainable score drivers  
        ✓ Interactive portfolio dashboard  
        ✓ Assessment history  
        ✓ Downloadable results  

        ### Responsible AI
        This application is an academic prototype using synthetic data.
        It is not intended to make actual lending decisions. Real lending
        would require verified information, regulatory compliance, model
        validation, fairness testing, security, privacy controls and human review.
        """
    )
    st.info("Never upload Aadhaar numbers, PAN numbers, bank passwords, bank account credentials or other sensitive personal information.")

st.markdown(
    '<div class="footer">🛡️ CreditGuard AI | Academic Decision-Support Prototype<br>'
    'Synthetic data • Machine Learning • Explainable Risk Assessment</div>',
    unsafe_allow_html=True,
)
