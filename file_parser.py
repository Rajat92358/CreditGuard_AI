import re
import pandas as pd
from pypdf import PdfReader


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def find_number(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
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


def parse_pdf_application(uploaded_file):
    text = extract_pdf_text(uploaded_file)

    # PDF readers can extract the ₹ symbol as a square/unknown character.
    # Therefore the money patterns allow any non-digit characters before the number.
    data = {
        "income": find_number(
            text,
            [
                r"monthly income\s*[:\-]?\s*[^\d\n]*([\d,]+)",
                r"income\s*[:\-]?\s*[^\d\n]*([\d,]+)"
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
                r"loan amount\s*[:\-]?\s*[^\d\n]*([\d,]+)",
                r"requested loan\s*[:\-]?\s*[^\d\n]*([\d,]+)"
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
                r"monthly debt\s*[:\-]?\s*[^\d\n]*([\d,]+)",
                r"existing monthly debt\s*[:\-]?\s*[^\d\n]*([\d,]+)"
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


def prepare_dataframe(df):
    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

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

    df = df.rename(columns=aliases)

    return df


def validate_dataframe(df, required_columns):
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    return missing
