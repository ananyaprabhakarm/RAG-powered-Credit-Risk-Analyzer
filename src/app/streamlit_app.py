import os
import json
import streamlit as st

from src.config import (
    RISK_MEDIUM_THRESHOLD,
    RISK_HIGH_THRESHOLD,
)
from src.risk_model.model import RiskInput, score_risk
from src.rag.retriever import get_retriever
from src.explain.generator import generate_explanation


st.set_page_config(page_title="Credit Risk Analyzer", page_icon="💳", layout="centered")

st.title("Credit Risk Analyzer")
st.caption("Predictive ML + RAG over regulations and cases")

with st.form("borrower_form"):
    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
        total_monthly_emi = st.number_input("Total Monthly EMIs (₹)", min_value=0.0, value=15000.0, step=500.0)
        requested_loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0.0, value=500000.0, step=10000.0)
    with col2:
        credit_utilization_ratio = st.slider("Credit Utilization (%)", min_value=0, max_value=100, value=75) / 100.0
        late_payments_last_12m = st.number_input("Late Payments (last 12 months)", min_value=0, value=3, step=1)

    submitted = st.form_submit_button("Analyze")

if submitted:
    inp = RiskInput(
        monthly_income=monthly_income,
        total_monthly_emi=total_monthly_emi,
        credit_utilization_ratio=credit_utilization_ratio,
        late_payments_last_12m=int(late_payments_last_12m),
        requested_loan_amount=requested_loan_amount,
    )
    result = score_risk(inp, medium_threshold=RISK_MEDIUM_THRESHOLD, high_threshold=RISK_HIGH_THRESHOLD)

    st.subheader("Credit Risk Score")
    st.metric(
        label="Risk Bucket",
        value=result.risk_bucket,
        delta=f"Default Prob: {result.probability_of_default:.2f}",
    )

    with st.expander("Derived features"):
        st.json(result.derived_features)

    # Retrieve relevant knowledge
    retriever = get_retriever()
    retrieved = retriever.search(
        query="credit risk assessment for borrowers with high utilization and late payments",
        top_k=5,
    )

    # Generate explanation (templated or LLM-based)
    explanation, cites = generate_explanation(
        risk_bucket=result.risk_bucket,
        probability_of_default=result.probability_of_default,
        features=result.derived_features,
        requested_loan_amount=requested_loan_amount,
        retrieved_snippets=retrieved,
    )

    st.subheader("Explanation")
    st.write(explanation)

    if cites:
        st.subheader("Citations")
        for c in cites:
            st.markdown(f"- {c['source']} — {c['preview']}")


