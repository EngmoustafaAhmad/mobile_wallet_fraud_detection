
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mobile Wallet Fraud Detection",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "fraud_detection_pipeline.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_bundle = load_model()
    model = model_bundle["model"]
    FEATURES = model_bundle["features"]
    THRESHOLD = model_bundle["threshold"]

except FileNotFoundError:
    st.error("Model artifact not found. Please export the trained model first.")
    st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# --------------------------------------------------
# 3. Application interface
# --------------------------------------------------

st.title("Mobile Wallet Fraud Detection")

st.write(
    "Analyze a mobile wallet transaction using a trained "
    "Machine Learning model."
)

st.info(
    "Enter the transaction details below to estimate its fraud risk."
)


# --------------------------------------------------
# 4. Transaction input form
# --------------------------------------------------

with st.form("transaction_form"):

    st.subheader("Transaction Information")

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction Step",
            min_value=1,
            max_value=1000,
            value=1,
            step=1,
            help="Time step associated with the transaction."
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            options=[
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.2f"
        )

    with col2:
        oldbalance_org = st.number_input(
            "Origin Account Balance Before Transaction",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            format="%.2f"
        )

        oldbalance_dest = st.number_input(
            "Destination Account Balance Before Transaction",
            min_value=0.0,
            value=2000.0,
            step=100.0,
            format="%.2f"
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        use_container_width=True
    )


# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

if submitted:

    if amount <= 0:
        st.warning("Transaction amount must be greater than zero.")

    else:
        transaction = pd.DataFrame(
            [{
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalance_org,
                "oldbalanceDest": oldbalance_dest
            }],
            columns=FEATURES
        )

        prediction = model.predict(transaction)[0]

        risk_score = model.predict_proba(transaction)[0][1]

        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fraud Risk Score",
                f"{risk_score:.2%}"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.2%}"
            )

        st.progress(float(risk_score))

        if risk_score >= THRESHOLD:
            st.error(
                "Potentially Fraudulent Transaction",
                icon="⚠️"
            )

            st.write(
                "The model classified this transaction as potentially fraudulent."
            )

        else:
            st.success(
                "Likely Legitimate Transaction",
                icon="✅"
            )

            st.write(
                "The model classified this transaction as legitimate."
            )

        st.caption(
            "The fraud risk score is a model output, not a calibrated "
            "probability or confirmation of fraud. Final decisions "
            "should involve additional verification."
        )