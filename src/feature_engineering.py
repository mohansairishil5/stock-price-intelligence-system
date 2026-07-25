import pandas as pd
import numpy as np

def create_features(stock_data):

    print("Original Shape:", stock_data.shape)

    # Lag Features
    stock_data["Lag_1"] = stock_data["Close"].shift(1)
    stock_data["Lag_2"] = stock_data["Close"].shift(2)
    stock_data["Lag_3"] = stock_data["Close"].shift(3)

    print("After Lag:", stock_data.shape)

    # Moving Averages
    stock_data["MA_5"] = stock_data["Close"].rolling(5).mean()
    stock_data["MA_10"] = stock_data["Close"].rolling(10).mean()

    print("After MA:", stock_data.shape)

    # EMA
    stock_data["EMA_26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()

    print("After EMA:", stock_data.shape)

    # Bollinger Middle
    stock_data["BB_Middle"] = stock_data["Close"].rolling(20).mean()

    print("After BB:", stock_data.shape)

    # Daily Return
    stock_data["Daily_Return"] = stock_data["Close"].pct_change()

    # Volatility
    stock_data["Volatility_5"] = (
        stock_data["Daily_Return"].rolling(5).std()
    )

    print("Before dropna:", stock_data.shape)

    stock_data = stock_data.dropna()

    print("After dropna:", stock_data.shape)

    return stock_data