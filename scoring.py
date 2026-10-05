def calculate_dti(
    income,
    monthly_debt
):

    if income <= 0:
        return 100.0

    return (
        monthly_debt / income
    ) * 100


def calculate_risk_score(
    probability
):

    return round(
        probability * 100,
        1
    )


def determine_decision(
    score,
    credit_score,
    dti
):

    # Strong hard-risk flags

    if credit_score < 550:

        return (
            "REJECT",
            "High Risk"
        )

    if dti > 60:

        return (
            "REJECT",
            "High Risk"
        )

    # Main thresholds

    if score >= 70:

        return (
            "APPROVE",
            "Low Risk"
        )

    elif score >= 50:

        return (
            "REFER",
            "Medium Risk"
        )

    else:

        return (
            "REJECT",
            "High Risk"
        )


def explain_drivers(
    income,
    employment_years,
    credit_score,
    loan_amount,
    existing_loans,
    monthly_debt
):

    dti = calculate_dti(
        income,
        monthly_debt
    )

    positive = []

    neutral = []

    negative = []

    # Credit score

    if credit_score >= 750:

        positive.append(
            f"Strong credit score ({credit_score})"
        )

    elif credit_score >= 650:

        neutral.append(
            f"Moderate credit score ({credit_score})"
        )

    else:

        negative.append(
            f"Low credit score ({credit_score})"
        )

    # Employment

    if employment_years >= 5:

        positive.append(
            f"Stable employment history "
            f"({employment_years:.1f} years)"
        )

    elif employment_years >= 2:

        neutral.append(
            f"Moderate employment stability "
            f"({employment_years:.1f} years)"
        )

    else:

        negative.append(
            f"Short employment history "
            f"({employment_years:.1f} years)"
        )

    # DTI

    if dti <= 30:

        positive.append(
            f"Low debt-to-income ratio "
            f"({dti:.1f}%)"
        )

    elif dti <= 45:

        neutral.append(
            f"Moderate debt-to-income ratio "
            f"({dti:.1f}%)"
        )

    else:

        negative.append(
            f"High debt-to-income ratio "
            f"({dti:.1f}%)"
        )

    # Existing loans

    if existing_loans == 0:

        positive.append(
            "No existing loans"
        )

    elif existing_loans <= 2:

        neutral.append(
            f"{existing_loans} existing loan(s)"
        )

    else:

        negative.append(
            f"High number of existing loans "
            f"({existing_loans})"
        )

    # Income

    if income >= 75000:

        positive.append(
            f"Strong monthly income "
            f"(₹{income:,.0f})"
        )

    elif income >= 40000:

        neutral.append(
            f"Moderate monthly income "
            f"(₹{income:,.0f})"
        )

    else:

        negative.append(
            f"Lower monthly income "
            f"(₹{income:,.0f})"
        )

    # Loan amount

    loan_to_income = (
        loan_amount /
        max(income, 1)
    )

    if loan_to_income <= 8:

        positive.append(
            "Loan amount is moderate "
            "relative to income"
        )

    elif loan_to_income <= 12:

        neutral.append(
            "Loan amount is moderately "
            "high relative to income"
        )

    else:

        negative.append(
            "Loan amount is high "
            "relative to income"
        )

    return (
        positive,
        neutral,
        negative
    )


def generate_recommendation(
    decision
):

    if decision == "APPROVE":

        return (
            "The applicant falls within the lower-risk "
            "range of this academic prototype. Standard "
            "verification would still be required in a "
            "real lending environment."
        )

    if decision == "REFER":

        return (
            "The applicant falls within a medium-risk "
            "range. Additional review is recommended "
            "before proceeding."
        )

    return (
        "The applicant shows elevated simulated risk "
        "based on the available variables. This is a "
        "pre-screening result and not a final lending decision."
    )
