import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn

# 1. Setup MLflow Tracking
mlflow.set_tracking_uri(mlflow.set_tracking_uri("sqlite:///C:/Users/mohan/OneDrive/Desktop/ML-WORK/mlflow.db"))

mlflow.set_experiment("Apple_Stock_Prediction")

# --- GENERATE DUMMY STOCK DATA (Replace this block with your actual data loading) ---
dates = pd.date_range(start="2025-01-01", periods=100)
df = pd.DataFrame({
    'Open': np.random.uniform(150, 180, 100),
    'High': np.random.uniform(155, 185, 100),
    'Low': np.random.uniform(145, 175, 100),
    'Close': np.random.uniform(148, 182, 100)
}, index=dates)
X = df[['Open', 'High', 'Low']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ----------------------------------------------------------------------------------

# 2. Start the MLflow Run Container
with mlflow.start_run():
    
    # Train your chosen model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Generate predictions & evaluate performance metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # 3. Log parameters and metrics explicitly to the dashboard
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    
    # 4. Save the trained model artifact inside MLflow
    mlflow.sklearn.log_model(model, "apple_stock_model")
    
    print(f"Tracking complete! Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
