import joblib
import pandas as pd
import yfinance as yf
import os

from src.feature_engineering import create_features


def predict_stock_price(ticker):

    # Load trained model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "linear_regression_selected.pkl"
    )

    model = joblib.load(model_path)


    # Download latest stock data
    stock_data = yf.download(
        ticker,
        period="6mo"
    )


    # Handle yfinance MultiIndex columns
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)


    # Create features
    stock_data = create_features(stock_data)


    # Features used during training
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


    # Latest data row
    latest_data = stock_data[selected_features].tail(1)


    # Prediction
    prediction = model.predict(latest_data)


    return prediction[0]