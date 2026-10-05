import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from model import train_model, FEATURES, predict_applicant

from scoring import (
    calculate_dti,
    calculate_risk_score,
    determine_decision,
    explain_drivers,
    generate_recommendation
)

from file_parser import (
    prepare_dataframe,
    validate_dataframe,
    parse_pdf_application
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL UI STYLE
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background: #f5f7fb;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    /* Main content */

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Headers */

    h1 {
        color: #0f172a;
        font-weight: 800;
    }

    h2 {
        color: #172554;
        font-weight: 700;
    }

    h3 {
        color: #1e3a8a;
        font-weight: 700;
    }

    /* Metric cards */

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Hero */

    .hero {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 100%
        );
        padding: 40px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.15);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #dbeafe;
        margin-bottom: 0;
    }

    /* Feature cards */

    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 25px;
        min-height: 210px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .feature-icon {
        font-size: 30px;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 750;
        color: #172554;
        margin-top: 10px;
    }

    .feature-text {
        color: #64748b;
        line-height: 1.6;
    }

    /* Result card */

    .result-card {
        background: white;
        border-radius: 20px;
        padding: 28px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.07);
    }

    /* Driver cards */

    .driver-positive {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .driver-neutral {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .driver-negative {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    /* Info box */

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 16px;
        color: #1e3a8a;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model, training_data, accuracy, auc = train_model()

    return model, training_data, accuracy, auc


model, training_data, model_accuracy, model_auc = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "assessment_history" not in st.session_state:

    st.session_state.assessment_history = []


if "bulk_results" not in st.session_state:

    st.session_state.bulk_results = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:10px 0 20px 0;
    ">
        <div style="font-size:42px;">🛡️</div>
        <div style="
            font-size:24px;
            font-weight:800;
        ">
            CreditGuard
        </div>
        <div style="
            font-size:12px;
            color:#94a3b8;
        ">
            AI CREDIT INTELLIGENCE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Overview",
        "👤 Applicant Assessment",
        "📂 Bulk Assessment",
        "📄 PDF Assessment",
        "📊 Risk Dashboard",
        "🤖 Model Intelligence",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        background:#1e293b;
        padding:15px;
        border-radius:12px;
        font-size:12px;
        color:#cbd5e1;
    ">
    <b>Academic Prototype</b><br><br>
    Uses synthetic data for demonstration.
    Not intended for actual lending decisions.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ASSESS APPLICANT
# =========================================================

def assess_applicant(
    applicant,
    source="Manual"
):

    probability = predict_applicant(
        model,
        applicant
    )

    eligibility_score = calculate_risk_score(
        probability
    )

    dti = calculate_dti(
        applicant["income"],
        applicant["monthly_debt"]
    )

    decision, risk_category = determine_decision(
        eligibility_score,
        applicant["credit_score"],
        dti
    )

    positive, neutral, negative = explain_drivers(
        applicant["income"],
        applicant["employment_years"],
        applicant["credit_score"],
        applicant["loan_amount"],
        applicant["existing_loans"],
        applicant["monthly_debt"]
    )

    recommendation = generate_recommendation(
        decision
    )

    result = applicant.copy()

    result.update(
        {
            "eligibility_score":
                eligibility_score,

            "model_probability":
                round(
                    probability * 100,
                    2
                ),

            "dti":
                round(
                    dti,
                    2
                ),

            "risk_category":
                risk_category,

            "decision":
                decision,

            "positive_drivers":
                " | ".join(positive),

            "neutral_drivers":
                " | ".join(neutral),

            "negative_drivers":
                " | ".join(negative),

            "recommendation":
                recommendation,

            "source":
                source,

            "assessment_time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }
    )

    return result


# =========================================================
# SCORE GAUGE
# =========================================================

def show_score_gauge(score):

    if score >= 70:

        gauge_color = "#10b981"

    elif score >= 50:

        gauge_color = "#f59e0b"

    else:

        gauge_color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "/100",
                "font": {
                    "size": 38,
                    "color": "#0f172a"
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": gauge_color
                },
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "#fee2e2"
                    },
                    {
                        "range": [50, 70],
                        "color": "#fef3c7"
                    },
                    {
                        "range": [70, 100],
                        "color": "#d1fae5"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=10
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# DISPLAY RESULT
# =========================================================

def display_result(result):

    st.markdown("---")

    st.markdown(
        "## 🎯 CreditGuard Assessment"
    )

    col1, col2 = st.columns(
        [1.3, 1]
    )

    with col1:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Eligibility Score"
        )

        show_score_gauge(
            result["eligibility_score"]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        decision = result["decision"]

        if decision == "APPROVE":

            st.success(
                "## ✅ APPROVE"
            )

        elif decision == "REFER":

            st.warning(
                "## ⚠️ REFER"
            )

        else:

            st.error(
                "## ❌ REJECT"
            )

        st.markdown(
            f"### {result['risk_category']}"
        )

        st.write(
            result["recommendation"]
        )

        st.markdown(
            f"""
            **Assessment Source:** {result['source']}  
            **Assessment Time:** {result['assessment_time']}
            """
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # KEY METRICS
    # -----------------------------------------------------

    st.markdown(
        "### Financial Snapshot"
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (
            "Credit Score",
            f"{result['credit_score']:.0f}"
        ),
        (
            "Debt-to-Income",
            f"{result['dti']:.1f}%"
        ),
        (
            "Monthly Income",
            f"₹{result['income']:,.0f}"
        ),
        (
            "Loan Amount",
            f"₹{result['loan_amount']:,.0f}"
        )
    ]

    for column, metric in zip(
        [col1, col2, col3, col4],
        metrics
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        {metric[0]}
                    </div>
                    <div class="metric-value">
                        {metric[1]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # DRIVERS
    # -----------------------------------------------------

    st.markdown(
        "### 🧠 Why did the model reach this result?"
    )

    col1, col2, col3 = st.columns(3)

    positive = result["positive_drivers"]

    neutral = result["neutral_drivers"]

    negative = result["negative_drivers"]

    with col1:

        st.markdown(
            "#### 🟢 Positive Drivers"
        )

        if positive:

            for item in positive.split(" | "):

                st.markdown(
                    f"""
                    <div class="driver-positive">
                    ✓ {item}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No major positive drivers."
            )

    with col2:

        st.markdown(
            "#### 🟡 Neutral Factors"
        )

        if neutral:

            for item in neutral.split(" | "):

                st.markdown(
                    f"""
                    <div class="driver-neutral">
                    • {item}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No major neutral factors."
            )

    with col3:

        st.markdown(
            "#### 🔴 Risk Factors"
        )

        if negative:

            for item in negative.split(" | "):

                st.markdown(
                    f"""
                    <div class="driver-negative">
                    ! {item}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.success(
                "No major risk factors identified."
            )


# =========================================================
# HOME / OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🛡️ CreditGuard AI
        </div>

        <div class="hero-subtitle">
        Intelligent Loan Eligibility & Credit-Risk
        Pre-Screening
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        A machine-learning powered academic prototype
        that combines financial indicators, predictive
        modelling and explainable rules to support
        initial loan pre-screening.
        """
    )

    st.markdown("### 📈 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        (
            "Training Records",
            f"{len(training_data):,}"
        ),
        (
            "Model Accuracy",
            f"{model_accuracy * 100:.1f}%"
        ),
        (
            "ROC-AUC",
            f"{model_auc:.3f}"
        ),
        (
            "Risk Features",
            str(len(FEATURES))
        )
    ]

    for column, card in zip(
        [col1, col2, col3, col4],
        cards
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        {card[0]}
                    </div>
                    <div class="metric-value">
                        {card[1]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### ⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">👤</div>

            <div class="feature-title">
            Manual Assessment
            </div>

            <div class="feature-text">
            Enter one applicant's financial information
            and instantly generate a pre-screening result.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">📂</div>

            <div class="feature-title">
            Bulk Assessment
            </div>

            <div class="feature-text">
            Upload hundreds or thousands of applicants
            through a CSV file and analyse them together.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">

            <div class="feature-icon">📄</div>

            <div class="feature-title">
            PDF Assessment
            </div>

            <div class="feature-text">
            Extract applicant information from a
            text-based PDF before running the assessment.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">

        <b>How CreditGuard AI works</b><br><br>

        Applicant Data → Validation → ML Model →
        Eligibility Score → Risk Rules →
        Approve / Refer / Reject → Explainable Drivers

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">

        🛡️ CreditGuard AI | Academic Decision-Support Prototype<br>

        Synthetic data • Machine Learning • Explainable Risk Assessment

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MANUAL ASSESSMENT
# =========================================================

elif page == "👤 Applicant Assessment":

    st.title("👤 Applicant Assessment")

    st.write(
        "Complete the applicant's financial profile below."
    )

    with st.form(
        "manual_assessment_form"
    ):

        st.markdown(
            "### 💰 Financial Profile"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            income = st.number_input(
                "Monthly Income (₹)",
                min_value=1.0,
                value=50000.0,
                step=1000.0
            )

        with col2:

            employment_years = st.number_input(
                "Employment Experience",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5
            )

        with col3:

            credit_score = st.number_input(
                "Credit / CIBIL Score",
                min_value=300,
                max_value=850,
                value=700,
                step=1
            )

        st.markdown(
            "### 🏦 Loan Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            loan_amount = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=1.0,
                value=500000.0,
                step=10000.0
            )

        with col2:

            loan_term = st.selectbox(
                "Loan Term",
                [12, 24, 36, 48, 60, 72, 84],
                index=4,
                format_func=lambda x:
                    f"{x} months"
            )

        with col3:

            existing_loans = st.number_input(
                "Existing Loans",
                min_value=0,
                max_value=20,
                value=1,
                step=1
            )

        st.markdown(
            "### 📊 Existing Obligations"
        )

        monthly_debt = st.number_input(
            "Existing Monthly Debt (₹)",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

        st.markdown("")

        submitted = st.form_submit_button(
            "🔍 Run Credit Assessment",
            use_container_width=True
        )

    if submitted:

        applicant = {
            "income": income,
            "employment_years": employment_years,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "existing_loans": existing_loans,
            "monthly_debt": monthly_debt,
            "loan_term": loan_term
        }

        result = assess_applicant(
            applicant,
            "Manual"
        )

        st.session_state.assessment_history.append(
            result
        )

        display_result(
            result
        )


# =========================================================
# CSV ASSESSMENT
# =========================================================

elif page == "📂 Bulk Assessment":

    st.title("📂 Bulk Applicant Assessment")

    st.write(
        "Upload a CSV containing multiple applicants."
    )

    uploaded_csv = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_csv is not None:

        try:

            df = pd.read_csv(
                uploaded_csv
            )

            df = prepare_dataframe(
                df
            )

            st.success(
                f"✓ {len(df):,} applicants loaded"
            )

            st.markdown(
                "### Data Preview"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

            required_columns = [
                "income",
                "employment_years",
                "credit_score",
                "loan_amount",
                "existing_loans",
                "monthly_debt",
                "loan_term"
            ]

            missing = validate_dataframe(
                df,
                required_columns
            )

            if missing:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing)
                )

            else:

                st.success(
                    "✓ All required fields detected"
                )

                if st.button(
                    "🚀 Assess All Applicants",
                    use_container_width=True
                ):

                    results = []

                    progress = st.progress(
                        0
                    )

                    for i, row in df.iterrows():

                        try:

                            applicant = {
                                "income":
                                    float(
                                        row["income"]
                                    ),

                                "employment_years":
                                    float(
                                        row[
                                            "employment_years"
                                        ]
                                    ),

                                "credit_score":
                                    float(
                                        row[
                                            "credit_score"
                                        ]
                                    ),

                                "loan_amount":
                                    float(
                                        row[
                                            "loan_amount"
                                        ]
                                    ),

                                "existing_loans":
                                    float(
                                        row[
                                            "existing_loans"
                                        ]
                                    ),

                                "monthly_debt":
                                    float(
                                        row[
                                            "monthly_debt"
                                        ]
                                    ),

                                "loan_term":
                                    float(
                                        row[
                                            "loan_term"
                                        ]
                                    )
                            }

                            result = assess_applicant(
                                applicant,
                                "CSV"
                            )

                            if "applicant_id" in df.columns:

                                result["applicant_id"] = (
                                    row[
                                        "applicant_id"
                                    ]
                                )

                            results.append(
                                result
                            )

                        except Exception as e:

                            st.warning(
                                f"Row {i + 1} skipped: {e}"
                            )

                        progress.progress(
                            (i + 1) / len(df)
                        )

                    if results:

                        results_df = pd.DataFrame(
                            results
                        )

                        st.session_state.bulk_results = (
                            results_df
                        )

                        st.success(
                            f"✓ Assessment completed for "
                            f"{len(results_df):,} applicants"
                        )

                        st.markdown(
                            "### Portfolio Summary"
                        )

                        counts = (
                            results_df[
                                "decision"
                            ].value_counts()
                        )

                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )

                        with col1:

                            st.metric(
                                "Total",
                                len(results_df)
                            )

                        with col2:

                            st.metric(
                                "Approve",
                                int(
                                    counts.get(
                                        "APPROVE",
                                        0
                                    )
                                )
                            )

                        with col3:

                            st.metric(
                                "Refer",
                                int(
                                    counts.get(
                                        "REFER",
                                        0
                                    )
                                )
                            )

                        with col4:

                            st.metric(
                                "Reject",
                                int(
                                    counts.get(
                                        "REJECT",
                                        0
                                    )
                                )
                            )

                        st.markdown(
                            "### Results"
                        )

                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        csv_data = (
                            results_df
                            .to_csv(
                                index=False
                            )
                            .encode("utf-8")
                        )

                        st.download_button(
                            "📥 Download Results",
                            data=csv_data,
                            file_name=(
                                "creditguard_results.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:

            st.error(
                f"Could not read the CSV: {e}"
            )


# =========================================================
# PDF ASSESSMENT
# =========================================================

elif page == "📄 PDF Assessment":

    st.title("📄 PDF Applicant Assessment")

    st.write(
        "Upload a text-based applicant PDF."
    )

    uploaded_pdf = st.file_uploader(
        "Choose PDF file",
        type=["pdf"]
    )

    if uploaded_pdf is not None:

        try:

            extracted_text, extracted_data = (
                parse_pdf_application(
                    uploaded_pdf
                )
            )

            st.success(
                "✓ PDF processed successfully"
            )

            with st.expander(
                "View Extracted Text"
            ):

                st.text(
                    extracted_text
                )

            st.markdown(
                "### 🔍 Review Extracted Information"
            )

            col1, col2 = st.columns(2)

            with col1:

                pdf_income = st.number_input(
                    "Monthly Income (₹)",
                    min_value=0.0,
                    value=float(
                        extracted_data[
                            "income"
                        ] or 0
                    )
                )

                pdf_employment = st.number_input(
                    "Employment Years",
                    min_value=0.0,
                    value=float(
                        extracted_data[
                            "employment_years"
                        ] or 0
                    )
                )

                pdf_credit = st.number_input(
                    "Credit Score",
                    min_value=300,
                    max_value=850,
                    value=int(
                        extracted_data[
                            "credit_score"
                        ] or 650
                    )
                )

                pdf_loan = st.number_input(
                    "Loan Amount (₹)",
                    min_value=0.0,
                    value=float(
                        extracted_data[
                            "loan_amount"
                        ] or 0
                    )
                )

            with col2:

                pdf_existing = st.number_input(
                    "Existing Loans",
                    min_value=0,
                    value=int(
                        extracted_data[
                            "existing_loans"
                        ] or 0
                    )
                )

                pdf_debt = st.number_input(
                    "Monthly Debt (₹)",
                    min_value=0.0,
                    value=float(
                        extracted_data[
                            "monthly_debt"
                        ] or 0
                    )
                )

                pdf_term = st.selectbox(
                    "Loan Term",
                    [12, 24, 36, 48, 60, 72, 84],
                    format_func=lambda x:
                        f"{x} months"
                )

            if st.button(
                "🔍 Assess PDF Applicant",
                use_container_width=True
            ):

                if pdf_income <= 0:

                    st.error(
                        "Please enter a valid income."
                    )

                elif pdf_loan <= 0:

                    st.error(
                        "Please enter a valid loan amount."
                    )

                else:

                    applicant = {
                        "income":
                            pdf_income,

                        "employment_years":
                            pdf_employment,

                        "credit_score":
                            pdf_credit,

                        "loan_amount":
                            pdf_loan,

                        "existing_loans":
                            pdf_existing,

                        "monthly_debt":
                            pdf_debt,

                        "loan_term":
                            pdf_term
                    }

                    result = assess_applicant(
                        applicant,
                        "PDF"
                    )

                    st.session_state.assessment_history.append(
                        result
                    )

                    display_result(
                        result
                    )

        except Exception as e:

            st.error(
                f"Could not process PDF: {e}"
            )

            st.info(
                "This version supports text-based PDFs. "
                "Scanned PDFs may require OCR."
            )


# =========================================================
# RISK DASHBOARD
# =========================================================

elif page == "📊 Risk Dashboard":

    st.title("📊 Risk Intelligence Dashboard")

    if (
        st.session_state.bulk_results is None
        and not st.session_state.assessment_history
    ):

        st.info(
            "Complete an assessment first to populate "
            "the dashboard."
        )

    else:

        if st.session_state.bulk_results is not None:

            dashboard_df = (
                st.session_state.bulk_results.copy()
            )

        else:

            dashboard_df = pd.DataFrame(
                st.session_state.assessment_history
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Applicants",
                len(dashboard_df)
            )

        with col2:

            st.metric(
                "Avg. Eligibility Score",
                f"{dashboard_df['eligibility_score'].mean():.1f}"
            )

        with col3:

            st.metric(
                "Avg. Credit Score",
                f"{dashboard_df['credit_score'].mean():.0f}"
            )

        with col4:

            st.metric(
                "Avg. DTI",
                f"{dashboard_df['dti'].mean():.1f}%"
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            decision_counts = (
                dashboard_df[
                    "decision"
                ]
                .value_counts()
                .reset_index()
            )

            decision_counts.columns = [
                "Decision",
                "Count"
            ]

            fig = px.pie(
                decision_counts,
                names="Decision",
                values="Count",
                title="Decision Distribution",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.histogram(
                dashboard_df,
                x="eligibility_score",
                nbins=20,
                title="Eligibility Score Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.scatter(
                dashboard_df,
                x="credit_score",
                y="eligibility_score",
                color="decision",
                hover_data=[
                    "income",
                    "loan_amount",
                    "dti"
                ],
                title="Credit Score vs Eligibility Score"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.scatter(
                dashboard_df,
                x="income",
                y="loan_amount",
                color="risk_category",
                hover_data=[
                    "credit_score",
                    "dti"
                ],
                title="Income vs Loan Amount"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# MODEL INTELLIGENCE
# =========================================================

elif page == "🤖 Model Intelligence":

    st.title("🤖 Model Intelligence")

    st.write(
        """
        CreditGuard AI uses Logistic Regression to estimate
        the probability of a positive simulated outcome.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model Accuracy",
            f"{model_accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "ROC-AUC",
            f"{model_auc:.3f}"
        )

    st.markdown("---")

    st.markdown(
        "### Model Features"
    )

    feature_names = {
        "income":
            "Monthly Income",

        "employment_years":
            "Employment Experience",

        "credit_score":
            "Credit Score",

        "loan_amount":
            "Requested Loan Amount",

        "existing_loans":
            "Existing Loans",

        "monthly_debt":
            "Monthly Debt",

        "loan_term":
            "Loan Term"
    }

    feature_df = pd.DataFrame(
        {
            "Variable":
                FEATURES,

            "Business Meaning":
                [
                    feature_names[x]
                    for x in FEATURES
                ]
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🔄 Decision Pipeline

        **Applicant Data**

        ↓

        **Data Validation**

        ↓

        **Logistic Regression Model**

        ↓

        **Eligibility Score**

        ↓

        **Credit Score + DTI Rules**

        ↓

        **APPROVE / REFER / REJECT**

        ↓

        **Explainable Score Drivers**
        """
    )

    st.warning(
        "Model performance is based on synthetic academic "
        "data and should not be interpreted as real-world "
        "lending performance."
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About CreditGuard AI")

    st.markdown(
        """
        ## CreditGuard AI

        CreditGuard AI is an academic loan eligibility
        and credit-risk pre-screening prototype.

        ### Technology Stack

        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - Plotly
        - PyPDF
        - Logistic Regression

        ### Core Capabilities

        ✓ Manual applicant assessment

        ✓ Bulk CSV assessment

        ✓ PDF information extraction

        ✓ Eligibility scoring

        ✓ Risk categorisation

        ✓ Approve / Refer / Reject

        ✓ Explainable score drivers

        ✓ Interactive dashboards

        ✓ Downloadable assessment results

        ### Responsible AI

        This application is an academic prototype.

        It uses synthetic data and simplified financial
        variables.

        It is not intended to make actual lending decisions.

        A real lending system would require verified
        applicant information, regulatory compliance,
        model validation, fairness testing, security,
        privacy controls and human review.

        Never upload Aadhaar numbers, PAN numbers,
        bank passwords, bank account credentials or
        other sensitive personal information.
        """
    )

    st.markdown("---")

    st.caption(
        "CreditGuard AI | Academic Decision-Support Prototype"
    )
