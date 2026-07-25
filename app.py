import streamlit as st

from src.predict import predict_stock_price


st.title("📈 Apple Stock Prediction AI")


ticker = st.text_input(
    "Enter Stock Symbol",
    "AAPL"
)


if st.button("Predict"):

    prediction = predict_stock_price(ticker)

    st.success(
        f"Predicted Next Closing Price: ${prediction:.2f}"
    )