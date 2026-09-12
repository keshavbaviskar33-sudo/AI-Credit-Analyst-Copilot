import os
import csv
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_LOG_PATH = os.path.join(BASE_DIR, 'review_log.csv')

# Load trained artifacts once, at startup
model = joblib.load(os.path.join(BASE_DIR, 'model_v3.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
shap_background = joblib.load(os.path.join(BASE_DIR, 'shap_background.pkl'))

ratio_cols = ['current_ratio', 'debt_to_equity', 'net_profit_margin',
              'ebitda_margin', 'retained_earnings_to_assets']

# Built once against a real training-data sample (not the row being explained) —
# a single-row background collapses every waterfall to near-zero contributions.
explainer = shap.LinearExplainer(model, shap_background)


def compute_ratios(x1, x4, x6, x10, x11, x14, x15, x16, x17):
    """Takes raw financial line items, returns the 5 ratios as a dict."""
    return {
        'current_ratio': x1 / x14,
        'debt_to_equity': x11 / (x10 - x17),
        'net_profit_margin': x6 / x16,
        'ebitda_margin': x4 / x16,
        'retained_earnings_to_assets': x15 / x10,
    }


def predict_risk(ratios_dict):
    """Takes a ratios dict, returns (prediction, probability, scaled features)."""
    X = pd.DataFrame([ratios_dict])[ratio_cols]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    return prediction, probability, X_scaled


def log_review(ratios_dict, prediction, probability, decision, comment):
    row = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **ratios_dict,
        'prediction': prediction,
        'probability': probability,
        'decision': decision,
        'comment': comment,
    }
    file_exists = os.path.isfile(REVIEW_LOG_PATH)
    with open(REVIEW_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --- Streamlit UI starts here ---
st.title("AI Credit Analyst Copilot")
st.write("Enter a company's raw financial figures to get a risk assessment.")

st.subheader("Financial Inputs")
x1 = st.number_input("Current Assets (X1)", value=500.0)
x4 = st.number_input("EBITDA (X4)", value=80.0)
x6 = st.number_input("Net Income (X6)", value=30.0)
x10 = st.number_input("Total Assets (X10)", value=700.0)
x11 = st.number_input("Total Long-term Debt (X11)", value=180.0)
x14 = st.number_input("Total Current Liabilities (X14)", value=160.0)
x15 = st.number_input("Retained Earnings (X15)", value=200.0)
x16 = st.number_input("Total Revenue (X16)", value=1000.0)
x17 = st.number_input("Total Liabilities (X17)", value=400.0)

if st.button("Assess Risk"):
    ratios = compute_ratios(x1, x4, x6, x10, x11, x14, x15, x16, x17)
    prediction, probability, X_scaled = predict_risk(ratios)

    st.subheader("Computed Ratios")
    st.dataframe(pd.DataFrame([ratios]))

    st.subheader("Risk Assessment")
    if prediction == 1:
        st.error(f"⚠️ Elevated Risk — {probability:.1%} predicted failure probability")
    else:
        st.success(f"✅ Lower Risk — {probability:.1%} predicted failure probability")

    # --- SHAP explanation ---
    st.subheader("Why This Assessment?")
    shap_values = explainer.shap_values(X_scaled)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=list(ratios.values()),  # raw values for display, per our earlier decision
            feature_names=ratio_cols
        ),
        show=False
    )
    st.pyplot(fig)

    # --- Human-in-the-loop review ---
    st.session_state['assessed'] = True
    st.session_state['last_ratios'] = ratios
    st.session_state['last_prediction'] = int(prediction)
    st.session_state['last_probability'] = float(probability)

if st.session_state.get('assessed'):
    st.subheader("Analyst Review")
    decision = st.radio("Your decision:", ["Approve", "Modify", "Reject"], key="decision")
    comment = st.text_area("Comment (optional):", key="comment")
    if st.button("Submit Review"):
        log_review(
            st.session_state['last_ratios'],
            st.session_state['last_prediction'],
            st.session_state['last_probability'],
            decision,
            comment,
        )
        st.write(f"Recorded: **{decision}** — \"{comment}\"")
        st.success(f"Saved to {os.path.basename(REVIEW_LOG_PATH)} for audit trail.")
