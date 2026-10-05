import re
import pandas as pd

from pypdf import PdfReader


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ---------------------------------------------------------
# FIND NUMBERS FROM TEXT
# ---------------------------------------------------------

def find_number(text, patterns):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(1)

            value = (
                value
                .replace(",", "")
                .replace("₹", "")
                .strip()
            )

            try:
                return float(value)

            except ValueError:
                pass

    return None


# ---------------------------------------------------------
# PARSE PDF APPLICATION
# ---------------------------------------------------------

def parse_pdf_application(uploaded_file):

    text = extract_pdf_text(uploaded_file)

    data = {

        "income": find_number(
            text,
            [
                r"monthly income\s*[:\-]?\s*[₹]?\s*([\d,]+)",
                r"income\s*[:\-]?\s*[₹]?\s*([\d,]+)"
            ]
        ),

        "employment_years": find_number(
            text,
            [
                r"employment years\s*[:\-]?\s*([\d.]+)",
                r"employment duration\s*[:\-]?\s*([\d.]+)",
                r"years of employment\s*[:\-]?\s*([\d.]+)"
            ]
        ),

        "credit_score": find_number(
            text,
            [
                r"credit score\s*[:\-]?\s*(\d+)",
                r"cibil score\s*[:\-]?\s*(\d+)"
            ]
        ),

        "loan_amount": find_number(
            text,
            [
                r"loan amount\s*[:\-]?\s*[₹]?\s*([\d,]+)",
                r"requested loan\s*[:\-]?\s*[₹]?\s*([\d,]+)"
            ]
        ),

        "existing_loans": find_number(
            text,
            [
                r"existing loans\s*[:\-]?\s*(\d+)",
                r"number of existing loans\s*[:\-]?\s*(\d+)"
            ]
        ),

        "monthly_debt": find_number(
            text,
            [
                r"monthly debt\s*[:\-]?\s*[₹]?\s*([\d,]+)",
                r"existing monthly debt\s*[:\-]?\s*[₹]?\s*([\d,]+)"
            ]
        ),

        "loan_term": find_number(
            text,
            [
                r"loan term\s*[:\-]?\s*(\d+)",
                r"tenure\s*[:\-]?\s*(\d+)"
            ]
        )
    }

    return text, data


# ---------------------------------------------------------
# PREPARE CSV DATA
# ---------------------------------------------------------

def prepare_dataframe(df):

    df = df.copy()

    # Standardize column names
    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    # Alternative column names
    aliases = {

        "monthly_income": "income",

        "income_monthly": "income",

        "employment_duration": "employment_years",

        "years_employed": "employment_years",

        "cibil_score": "credit_score",

        "requested_loan": "loan_amount",

        "loan_requested": "loan_amount",

        "number_of_existing_loans": "existing_loans",

        "existing_debt": "monthly_debt",

        "monthly_existing_debt": "monthly_debt",

        "tenure": "loan_term"
    }

    df = df.rename(
        columns=aliases
    )

    return df


# ---------------------------------------------------------
# VALIDATE CSV DATA
# ---------------------------------------------------------

def validate_dataframe(df, required_columns):

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    return missing
