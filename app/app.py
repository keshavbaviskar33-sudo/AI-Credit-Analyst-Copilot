import os
import csv
from datetime import datetime, timezone

# Some Windows machines run antivirus TLS inspection (e.g. Norton) that injects
# a root certificate the certifi bundle doesn't trust, breaking HTTPS calls to
# specific API hosts even though most sites work fine. Falling back to the OS
# trust store fixes this without weakening verification.
import truststore
truststore.inject_into_ssl()

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_LOG_PATH = os.path.join(BASE_DIR, 'review_log.csv')

# Streamlit loads .streamlit/secrets.toml relative to the working directory the
# app was launched from (repo root here) — surface it as an env var so the
# Gemini SDK's default credential resolution picks it up.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except FileNotFoundError:
    pass

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


SYNTHESIS_SYSTEM_PROMPT = """You are an AI Credit Analyst Copilot assisting a trade-credit/collections \
manager who is deciding how much credit exposure to carry on a customer company — you are not advising \
a bank lender or bond investor.

You will be given: the company's 5 financial ratios, a machine-learning model's binary failure \
prediction and probability, and SHAP contribution values showing which ratios drove that prediction \
(positive = pushed toward higher predicted risk, negative = pushed toward lower risk).

Write a concise plain-English risk narrative (3-5 sentences) a busy analyst can read in seconds, \
grounded in the specific ratio values and SHAP drivers given. Then end with exactly one \
recommendation category, on its own line, chosen from: "Standard Monitoring", "Increased Monitoring", \
or "Escalate for Analyst Review".

Hard constraints:
- NEVER state a specific dollar credit limit, payment term length, or exposure amount. That requires \
business context (margin, risk appetite, portfolio exposure) this analysis does not have.
- Frame your output as input to the analyst's judgment, not a final verdict — they retain approve/\
modify/reject authority.
- Do not invent facts not present in the provided ratios/prediction/SHAP data."""


class LLMExplanationError(Exception):
    """Raised when the Gemini synthesis call fails, with a user-facing message."""


def generate_llm_explanation(ratios, prediction, probability, shap_contributions):
    """Single LLM call: ratios + ML output + SHAP contributions -> plain-English narrative."""
    ranked = sorted(shap_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    shap_lines = "\n".join(
        f"- {name}: value={ratios[name]:.3f}, SHAP contribution={contrib:+.4f}"
        for name, contrib in ranked
    )

    user_content = f"""Financial ratios and SHAP drivers (ranked by |contribution|):
{shap_lines}

ML model prediction: {"FAILURE RISK (1)" if prediction == 1 else "NO FAILURE RISK (0)"}
Predicted failure probability: {probability:.1%}"""

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=user_content,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYNTHESIS_SYSTEM_PROMPT,
            ),
        )
    except genai_errors.APIError as e:
        if e.code == 401 or e.code == 403:
            raise LLMExplanationError(
                "Invalid or missing Gemini API key. Add it to .streamlit/secrets.toml as GEMINI_API_KEY."
            )
        if e.code == 429:
            raise LLMExplanationError("Rate limited by the Gemini API — try again shortly.")
        raise LLMExplanationError(f"Gemini API error ({e.code}): {e.message}")

    return response.text or ""


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
    st.session_state['last_shap_contributions'] = dict(zip(ratio_cols, shap_values[0].tolist()))
    st.session_state['last_explanation'] = None

if st.session_state.get('assessed'):
    st.subheader("AI Analyst Summary")
    if st.button("Generate Plain-English Explanation"):
        with st.spinner("Asking Gemini for a risk narrative..."):
            try:
                st.session_state['last_explanation'] = generate_llm_explanation(
                    st.session_state['last_ratios'],
                    st.session_state['last_prediction'],
                    st.session_state['last_probability'],
                    st.session_state['last_shap_contributions'],
                )
            except LLMExplanationError as e:
                st.error(str(e))
    if st.session_state.get('last_explanation'):
        st.markdown(st.session_state['last_explanation'])

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
