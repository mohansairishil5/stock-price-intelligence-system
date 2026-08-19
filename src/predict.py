import os
from datetime import datetime

import joblib
import pandas as pd
import yfinance as yf

from src.feature_engineering import create_features


def predict_stock_price(ticker):
    """
    Predict the next closing price of a stock and
    return a professional prediction report.
    """

    # ==========================================================
    # Load Trained Model
    # ==========================================================
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "linear_regression_selected.pkl"
    )

    model = joblib.load(model_path)

    # ==========================================================
    # Download Latest Stock Data
    # ==========================================================
    stock_data = yf.download(
        ticker,
        period="6mo",
        progress=False
    )

    # Handle MultiIndex columns returned by yfinance
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    # ==========================================================
    # Feature Engineering
    # ==========================================================
    stock_data = create_features(stock_data)

    # ==========================================================
    # Features Used During Training
    # ==========================================================
    selected_features = [
        "Close",
        "High",
        "Low",
        "Open",
        "EMA_26",
        "BB_Middle",
        "Lag_1",
        "Lag_2",
        "MA_5",
        "MA_10"
    ]

    # ==========================================================
    # Prepare Latest Observation
    # ==========================================================
    latest_data = stock_data[selected_features].tail(1)

    # ==========================================================
    # Make Prediction
    # ==========================================================
    predicted_price = float(model.predict(latest_data)[0])

    current_price = float(stock_data["Close"].iloc[-1])

    expected_change = predicted_price - current_price

    expected_percentage_change = (
        expected_change / current_price
    ) * 100

    # ==========================================================
    # Company Names
    # ==========================================================
    company_names = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc."
    }

    company_name = company_names.get(
        ticker.upper(),
        ticker.upper()
    )

    # ==========================================================
    # Historical Data (for Streamlit Chart)
    # ==========================================================
    historical_data = stock_data.reset_index()

    historical_data["Date"] = historical_data["Date"].dt.strftime("%Y-%m-%d")

    # ==========================================================
    # Prediction Report
    # ==========================================================
    prediction_report = {

        "prediction_information": {

            "company_name": company_name,
            "stock_symbol": ticker.upper(),
            "prediction_date": datetime.now().strftime("%d-%m-%Y"),
            "prediction_time": datetime.now().strftime("%I:%M %p")

        },

        "market_information": {

            "current_closing_price": round(current_price, 2),
            "predicted_next_closing_price": round(predicted_price, 2),
            "expected_price_change": round(expected_change, 2),
            "expected_percentage_change": round(
                expected_percentage_change,
                2
            )

        },

        "model_information": {

            "model_name": "Linear Regression",
            "model_version": "1.0",
            "prediction_target": "Next Closing Price",
            "number_of_features": len(selected_features)

        },

        "historical_data": {

            "dates": historical_data["Date"].tolist(),
            "closing_prices": historical_data["Close"].round(2).tolist()

        },

        "performance_metrics": {

            "r2_score": 0.9509,
            "mae": 3.8766,
            "rmse": 5.4467

        },

        "system_information": {

            "project_name": "Stock Price Intelligence System",
            "developer": "Mohan Sai Rishil",
            "version": "1.0"

        }

    }

    return prediction_report