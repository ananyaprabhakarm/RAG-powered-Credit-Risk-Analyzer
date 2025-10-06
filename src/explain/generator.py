from __future__ import annotations

from typing import Dict, List, Tuple

from src.config import OPENAI_API_KEY, OPENAI_MODEL


def _templated_explanation(
    risk_bucket: str,
    probability_of_default: float,
    features: Dict,
    requested_loan_amount: float,
    retrieved_snippets: List[Dict],
) -> Tuple[str, List[Dict]]:
    lines: List[str] = []
    lines.append(
        f"Applicant is {risk_bucket} risk (default probability {probability_of_default:.2f})."
    )
    if "debt_to_income_ratio" in features:
        dti_pct = features["debt_to_income_ratio"] * 100.0
        lines.append(f"Debt-to-income ratio: {dti_pct:.1f}%.")
    if "credit_utilization_ratio" in features:
        util_pct = features["credit_utilization_ratio"] * 100.0
        lines.append(f"Credit utilization: {util_pct:.1f}%.")
    if "late_payments_last_12m" in features:
        lines.append(f"Late payments (12m): {features['late_payments_last_12m']}.")

    if retrieved_snippets:
        lines.append("Relevant guidelines and similar cases considered:")
        for r in retrieved_snippets[:3]:
            lines.append(f"- {r['source']}: {r['preview']}…")

    if risk_bucket == "High":
        lines.append("Recommendation: request collateral, reduce loan amount, or add guarantor.")
    elif risk_bucket == "Medium":
        lines.append("Recommendation: consider reduced amount or stricter verification checks.")
    else:
        lines.append("Recommendation: proceed subject to standard KYC and income verification.")

    return "\n".join(lines), [
        {"source": r["source"], "preview": r["preview"]} for r in retrieved_snippets[:5]
    ]


def _llm_explanation(
    risk_bucket: str,
    probability_of_default: float,
    features: Dict,
    requested_loan_amount: float,
    retrieved_snippets: List[Dict],
) -> Tuple[str, List[Dict]]:
    try:
        from openai import OpenAI
    except Exception:
        return _templated_explanation(
            risk_bucket, probability_of_default, features, requested_loan_amount, retrieved_snippets
        )

    if not OPENAI_API_KEY:
        return _templated_explanation(
            risk_bucket, probability_of_default, features, requested_loan_amount, retrieved_snippets
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    snippets_str = "\n\n".join([
        f"Source: {r['source']}\nExcerpt: {r['preview']}" for r in retrieved_snippets[:5]
    ])
    prompt = f"""
You are a credit risk analyst. Explain the decision clearly and concisely, citing relevant guidelines from the provided snippets when appropriate. Make concrete, actionable recommendations.

Risk bucket: {risk_bucket}
Default probability: {probability_of_default:.2f}
Requested loan amount: {requested_loan_amount}
Features: {features}

Snippets:\n{snippets_str}

Write 6-10 sentences.
"""

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise, regulation-aware credit analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = completion.choices[0].message.content.strip()
        cites = [{"source": r["source"], "preview": r["preview"]} for r in retrieved_snippets[:5]]
        return text, cites
    except Exception:
        return _templated_explanation(
            risk_bucket, probability_of_default, features, requested_loan_amount, retrieved_snippets
        )


def generate_explanation(
    risk_bucket: str,
    probability_of_default: float,
    features: Dict,
    requested_loan_amount: float,
    retrieved_snippets: List[Dict],
) -> Tuple[str, List[Dict]]:
    if OPENAI_API_KEY:
        return _llm_explanation(
            risk_bucket, probability_of_default, features, requested_loan_amount, retrieved_snippets
        )
    return _templated_explanation(
        risk_bucket, probability_of_default, features, requested_loan_amount, retrieved_snippets
    )


