import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


FEATURES = [
    "income",
    "employment_years",
    "credit_score",
    "loan_amount",
    "existing_loans",
    "monthly_debt",
    "loan_term"
]


def generate_synthetic_data(n=1500, random_state=42):

    rng = np.random.default_rng(random_state)

    income = rng.integers(
        18000,
        250000,
        n
    )

    employment_years = np.clip(
        rng.normal(5, 4, n),
        0,
        30
    ).round(1)

    credit_score = np.clip(
        rng.normal(690, 85, n),
        300,
        850
    ).round()

    loan_amount = np.clip(
        income * rng.uniform(3, 12, n),
        50000,
        3000000
    ).round()

    existing_loans = rng.poisson(
        1.2,
        n
    )

    existing_loans = np.clip(
        existing_loans,
        0,
        6
    )

    monthly_debt = np.clip(
        income * rng.uniform(0.05, 0.65, n),
        0,
        None
    ).round()

    loan_term = rng.choice(
        [12, 24, 36, 48, 60, 72, 84],
        n
    )

    dti = monthly_debt / income

    # Synthetic relationship used ONLY
    # to create academic demonstration data.

    logit = (
        0.018 * (credit_score - 650)
        + 0.35 * employment_years
        + 0.000008 * income
        - 2.8 * dti
        - 0.35 * existing_loans
        - 0.00000025 * loan_amount
        - 0.003 * loan_term
    )

    probability = 1 / (
        1 + np.exp(-logit)
    )

    approved = rng.binomial(
        1,
        probability
    )

    return pd.DataFrame({

        "income": income,

        "employment_years":
            employment_years,

        "credit_score":
            credit_score,

        "loan_amount":
            loan_amount,

        "existing_loans":
            existing_loans,

        "monthly_debt":
            monthly_debt,

        "loan_term":
            loan_term,

        "approved":
            approved
    })


def train_model():

    df = generate_synthetic_data()

    X = df[FEATURES]

    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    return (
        model,
        df,
        accuracy,
        auc
    )


def predict_applicant(
    model,
    applicant
):

    input_data = pd.DataFrame(
        [applicant],
        columns=FEATURES
    )

    probability = model.predict_proba(
        input_data
    )[0][1]

    return float(probability)
