from dataclasses import dataclass


@dataclass
class RiskInput:
    monthly_income: float
    total_monthly_emi: float
    credit_utilization_ratio: float  # 0..1
    late_payments_last_12m: int
    requested_loan_amount: float


@dataclass
class RiskOutput:
    probability_of_default: float  # 0..1 -> here interpreted as chance of default
    risk_bucket: str  # "Low" | "Medium" | "High"
    derived_features: dict


def _categorize(prob_default: float, medium_threshold: float, high_threshold: float) -> str:
    if prob_default >= high_threshold:
        return "High"
    if prob_default >= medium_threshold:
        return "Medium"
    return "Low"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_risk(
    risk_input: RiskInput,
    medium_threshold: float = 0.4,
    high_threshold: float = 0.65,
) -> RiskOutput:
    """Simple, transparent risk scoring function.

    This is a placeholder, rule-based model designed for clarity. Replace with an ML
    model later (e.g., logistic regression/XGBoost) while keeping the same interface.
    """

    income = max(0.0, float(risk_input.monthly_income))
    monthly_emi = max(0.0, float(risk_input.total_monthly_emi))
    utilization = _clip01(float(risk_input.credit_utilization_ratio))
    late_payments = max(0, int(risk_input.late_payments_last_12m))

    dti = monthly_emi / income if income > 0 else 1.0  # debt-to-income

    # Base risk components (heuristics)
    risk_from_dti = min(1.0, dti)  # higher dti -> higher risk
    risk_from_util = utilization  # higher utilization -> higher risk
    risk_from_lates = min(1.0, late_payments / 4.0)  # 4+ late payments => max risk

    # Weighted combination
    # Tunable weights to reflect domain intuition
    w_dti, w_util, w_lates = 0.45, 0.35, 0.20
    prob_default = (
        w_dti * risk_from_dti + w_util * risk_from_util + w_lates * risk_from_lates
    )

    # Normalize softly to 0..1
    prob_default = _clip01(prob_default)
    bucket = _categorize(prob_default, medium_threshold, high_threshold)

    derived = {
        "debt_to_income_ratio": round(dti, 4),
        "credit_utilization_ratio": round(utilization, 4),
        "late_payments_last_12m": late_payments,
    }

    return RiskOutput(
        probability_of_default=prob_default,
        risk_bucket=bucket,
        derived_features=derived,
    )


