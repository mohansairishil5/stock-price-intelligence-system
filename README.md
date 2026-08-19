# 📈 Stock Price Intelligence System

An end-to-end Machine Learning application that predicts the **next closing price** of a stock using historical market data. The project demonstrates a complete ML workflow—from data collection and feature engineering to model training, evaluation, and deployment with Streamlit.

---

# 🚀 Project Overview

The Stock Price Intelligence System is designed to forecast the next day's stock closing price using historical stock market data.

This project follows an industry-style Machine Learning pipeline, including:

- Data Collection
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Prediction System
- Interactive Streamlit Dashboard

---

# 🎯 Business Problem

Investors and analysts monitor historical market trends to estimate future stock prices.

This project applies Machine Learning techniques to learn historical patterns and generate predictions for the next closing price, helping demonstrate how predictive analytics can support financial decision-making.

> **Note:** This project is intended for educational purposes only and does not provide financial advice.

---

# 🎯 Objectives

- Predict the next closing price of a stock
- Build a complete Machine Learning pipeline
- Engineer meaningful financial features
- Evaluate model performance using regression metrics
- Deploy an interactive prediction dashboard using Streamlit

---

# 📂 Project Structure

```text
Stock-Price-Intelligence-System/

│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│
├── models/
│   ├── linear_regression_model.pkl
│   └── linear_regression_selected.pkl
│
├── notebooks/
│
├── src/
│   ├── feature_engineering.py
│   ├── predict.py
│   └── train.py
│
└── assets/
```

---

# 📊 Dataset

**Source:** Yahoo Finance

**Library Used:**

- yfinance

The model downloads the latest historical stock data directly from Yahoo Finance.

---

# ⚙️ Feature Engineering

The following features were created to improve prediction performance.

| Feature | Description |
|----------|-------------|
| Lag_1 | Previous day's closing price |
| Lag_2 | Closing price two days earlier |
| Lag_3 | Closing price three days earlier |
| MA_5 | 5-day Moving Average |
| MA_10 | 10-day Moving Average |
| EMA_26 | 26-day Exponential Moving Average |
| BB_Middle | Bollinger Band Middle |
| Daily_Return | Daily Percentage Return |
| Volatility_5 | 5-Day Rolling Volatility |

---

# 🤖 Machine Learning Model

Model Used:

- Linear Regression

Prediction Target:

- Next Closing Price

---

# 📈 Model Performance

| Metric | Value |
|---------|------:|
| R² Score | **0.9509** |
| MAE | **3.8766** |
| RMSE | **5.4467** |

These evaluation metrics indicate that the model captures a large proportion of the variation in stock prices while maintaining relatively low prediction error.

---

# 🖥️ Streamlit Dashboard

The application provides:

- Prediction Information
- Prediction Summary
- Model Information
- Model Performance
- System Information
- Stock Price Prediction

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- pandas
- numpy
- scikit-learn
- joblib
- yfinance
- streamlit

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Stock-Price-Intelligence-System.git
```

Move into the project directory

```bash
cd Stock-Price-Intelligence-System
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📷 Application Preview

> Add screenshots of the Streamlit dashboard here.

Example:

```
assets/dashboard.png
```

---

# 🔮 Future Enhancements

- Interactive stock price visualization
- Multiple Machine Learning models
- XGBoost implementation
- LSTM implementation
- Model comparison dashboard
- Download prediction reports
- Advanced financial analytics

---

# 👨‍💻 Developer

**Mohan Sai Rishil**

Machine Learning | Data Science | Artificial Intelligence

GitHub:
https://github.com/mohansairishil5

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.