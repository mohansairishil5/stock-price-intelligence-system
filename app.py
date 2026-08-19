import streamlit as st
from src.predict import predict_stock_price

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Stock Price Intelligence System",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# Header
# ==========================================================
st.title("📈 Stock Price Intelligence System")

st.markdown("""
### AI-Powered Stock Market Forecasting Platform

Predict the **next closing price** of a stock using a trained Machine Learning model.

**Current Case Study:** Apple Inc. (AAPL)
""")

st.divider()

# ==========================================================
# User Input
# ==========================================================
ticker = st.text_input(
    "Enter Stock Symbol",
    value="AAPL"
).upper()

# ==========================================================
# Prediction
# ==========================================================
if st.button("Predict Stock Price", use_container_width=True):

    report = predict_stock_price(ticker)

    st.success("Prediction Completed Successfully!")

    # ======================================================
    # Prediction Information
    # ======================================================

    st.subheader("📋 Prediction Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Company Name**")
        st.write(report["prediction_information"]["company_name"])

        st.write("**Stock Symbol**")
        st.write(report["prediction_information"]["stock_symbol"])

    with col2:
        st.write("**Prediction Date**")
        st.write(report["prediction_information"]["prediction_date"])

        st.write("**Prediction Time**")
        st.write(report["prediction_information"]["prediction_time"])

    st.divider()

    # ======================================================
    # Market Information
    # ======================================================

    st.subheader("📈 Market Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"${report['market_information']['current_closing_price']:.2f}"
    )

    c2.metric(
        "Predicted Price",
        f"${report['market_information']['predicted_next_closing_price']:.2f}"
    )

    c3.metric(
        "Price Change",
        f"${report['market_information']['expected_price_change']:.2f}"
    )

    c4.metric(
        "Percentage Change",
        f"{report['market_information']['expected_percentage_change']:.2f}%"
    )

    st.divider()

    # ======================================================
    # Prediction Summary
    # ======================================================

    st.subheader("📊 Prediction Summary")

    current_price = report["market_information"]["current_closing_price"]
    predicted_price = report["market_information"]["predicted_next_closing_price"]
    price_change = report["market_information"]["expected_price_change"]
    percentage_change = report["market_information"]["expected_percentage_change"]

    if price_change > 0:
        outlook = "🟢 Upward Trend Expected"
        recommendation = "Hold / Buy (Educational Purpose Only)"
    elif price_change < 0:
        outlook = "🔴 Downward Trend Expected"
        recommendation = "Monitor the Stock (Educational Purpose Only)"
    else:
        outlook = "🟡 Stable Movement Expected"
        recommendation = "No Significant Price Change Expected"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Current Closing Price",
            f"${current_price:.2f}"
        )

        st.metric(
            "Predicted Closing Price",
            f"${predicted_price:.2f}"
        )

    with col2:
        st.metric(
            "Expected Price Change",
            f"${price_change:.2f}"
        )

        st.metric(
            "Expected Percentage Change",
            f"{percentage_change:.2f}%"
        )

    st.success(f"**Market Outlook:** {outlook}")

    st.info(f"**Recommendation:** {recommendation}")

    st.divider()

    # ======================================================
    # Model Information
    # ======================================================

    st.subheader("🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Name**")
        st.write(report["model_information"]["model_name"])

        st.write("**Model Version**")
        st.write(report["model_information"]["model_version"])

    with col2:
        st.write("**Prediction Target**")
        st.write(report["model_information"]["prediction_target"])

        st.write("**Number of Features**")
        st.write(report["model_information"]["number_of_features"])

    st.divider()

    # ======================================================
    # Model Performance
    # ======================================================

    st.subheader("📊 Model Performance")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        label="R² Score",
        value="0.9509"
    )

    m2.metric(
        label="MAE",
        value="3.8766"
    )

    m3.metric(
        label="RMSE",
        value="5.4467"
    )

    st.info(
        """
### Model Evaluation Summary

**Algorithm:** Linear Regression

- **R² Score (0.9509):** The model explains **95.09%** of the variation in stock prices.

- **MAE (3.8766):** Average prediction error is approximately **$3.88**.

- **RMSE (5.4467):** Larger prediction errors are penalized more heavily, giving an overall error of approximately **$5.45**.
"""
    )

    st.divider()

    # ======================================================
    # System Information
    # ======================================================

    st.subheader("🖥️ System Information")

    st.write(
        f"**Project Name:** {report['system_information']['project_name']}"
    )

    st.write(
        f"**Developer:** {report['system_information']['developer']}"
    )

    st.write(
        f"**Version:** {report['system_information']['version']}"
    )

    st.divider()

    st.warning(
        "⚠️ This prediction is generated using historical stock market data "
        "and Machine Learning. It is intended for educational purposes only "
        "and should not be considered financial advice."
    )