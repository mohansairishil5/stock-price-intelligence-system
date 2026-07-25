import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def run_forecast():
    # 1. Paths configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "linear_regression_model.pkl")
    data_path = os.path.join(base_dir, "data", "apple_stock_processed.csv")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # 2. Load the model and processed data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    print(f"Loaded dataset with {len(df)} rows.")
    print(f"Latest Date in Dataset: {df['Date'].iloc[-1].strftime('%Y-%m-%d')} | Close: ${df['Close'].iloc[-1]:.2f}")
    print("=" * 55)

    # Columns expected by the trained model
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Lag_1', 'Lag_2', 'Lag_3', 'MA_5', 'MA_10', 'Daily_Return', 'Volatility_5']

    temp_df = df.copy()

    forecast_dates = []
    forecast_prices = []

    # 3. Recursive Forecasting Loop for the next 10 Business Days
    for i in range(10):
        # Extract features from the latest day
        last_row = temp_df.iloc[-1]
        features_input = pd.DataFrame([last_row[feature_cols]])
        
        # Predict tomorrow's Close price
        pred_close = model.predict(features_input)[0]
        
        # Identify next business day
        next_date = last_row['Date'] + pd.tseries.offsets.BDay(1)
        
        # Estimate the base features for the new day
        new_close = pred_close
        new_open = last_row['Close']
        new_high = max(new_open, new_close) * 1.002
        new_low = min(new_open, new_close) * 0.998
        new_volume = last_row['Volume']
        
        # Add the raw values to temp_df
        new_row_df = pd.DataFrame([{
            'Date': next_date,
            'Close': new_close,
            'High': new_high,
            'Low': new_low,
            'Open': new_open,
            'Volume': new_volume
        }])
        temp_df = pd.concat([temp_df, new_row_df], ignore_index=True)
        
        # Recalculate indicators for the new row
        temp_df.loc[temp_df.index[-1], 'Lag_1'] = temp_df['Close'].iloc[-2]
        temp_df.loc[temp_df.index[-1], 'Lag_2'] = temp_df['Close'].iloc[-3]
        temp_df.loc[temp_df.index[-1], 'Lag_3'] = temp_df['Close'].iloc[-4]
        
        temp_df.loc[temp_df.index[-1], 'MA_5'] = temp_df['Close'].iloc[-5:].mean()
        temp_df.loc[temp_df.index[-1], 'MA_10'] = temp_df['Close'].iloc[-10:].mean()
        
        temp_df.loc[temp_df.index[-1], 'Daily_Return'] = (temp_df['Close'].iloc[-1] - temp_df['Close'].iloc[-2]) / temp_df['Close'].iloc[-2]
        
        last_5_returns = temp_df['Close'].iloc[-6:].pct_change().dropna()
        temp_df.loc[temp_df.index[-1], 'Volatility_5'] = last_5_returns.std()
        
        forecast_dates.append(next_date)
        forecast_prices.append(pred_close)

    # 4. Display Results
    print("10-DAY RECURSIVE STOCK FORECAST")
    print("=" * 55)
    for idx, (date, price) in enumerate(zip(forecast_dates, forecast_prices), 1):
        print(f"Day {idx:2d} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
    print("=" * 55)

if __name__ == "__main__":
    run_forecast()
