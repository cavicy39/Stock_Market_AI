import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 🎯 Streamlit UI
st.title("📈 Stock & Crypto AI Predictor")

# 📌 User Inputs
ticker = st.text_input("Enter Stock/Crypto Symbol (e.g., AAPL, BTC-USD)", "AAPL")
days_to_predict = st.slider("Days to Predict", 1, 30, 7)

if st.button("Predict Price"):
    st.write(f"🔄 Fetching Data for {ticker}...")

    # ✅ Step 1: Get Historical Data
    df = yf.download(ticker, period="5y", interval="1d")

    if df.empty:
        st.error("⚠️ Invalid Ticker Symbol or No Data Available!")
    else:
        st.write("✅ Data Fetched Successfully!")

        # ✅ Step 2: Feature Engineering
        df["Returns"] = df["Close"].pct_change()
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["Volatility"] = df["Returns"].rolling(window=10).std()

        # ✅ Additional Features (Momentum & RSI)
        df["Momentum"] = df["Close"] - df["Close"].shift(4)
        df["RSI"] = 100 - (100 / (1 + df["Returns"].rolling(window=14).mean() / df["Returns"].rolling(window=14).std()))

        df.dropna(inplace=True)  # Remove NaN values

        # ✅ Step 3: Train XGBoost Model
        features = ["SMA_10", "SMA_50", "Volatility", "Momentum", "RSI"]
        target = "Close"

        X = df[features].copy()
        y = df[target].copy()

        # ✅ Ensure all column names are properly formatted
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = X.columns.get_level_values(0)
        X.columns = X.columns.astype(str).str.strip()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # ✅ Train XGBoost Model
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X_train, y_train)

        # ✅ Step 4: Prepare Future Data for Prediction (Fixing Length Mismatch)
        future_dates = pd.date_range(start=df.index[-1], periods=days_to_predict + 1, freq="D")[1:]
        future_df = pd.DataFrame(index=future_dates)

        # ✅ Use realistic future trends instead of static last values
        future_df["SMA_10"] = df["SMA_10"].iloc[-10:].mean() + np.random.normal(0, 0.5, size=len(future_dates))
        future_df["SMA_50"] = df["SMA_50"].iloc[-50:].mean() + np.random.normal(0, 0.3, size=len(future_dates))
        future_df["Volatility"] = df["Volatility"].iloc[-10:].mean() + np.random.normal(0, 0.02, size=len(future_dates))
        future_df["Momentum"] = df["Momentum"].iloc[-10:].mean() + np.random.normal(0, 0.5, size=len(future_dates))
        future_df["RSI"] = df["RSI"].iloc[-10:].mean() + np.random.normal(0, 1, size=len(future_dates))

        # ✅ Ensure feature names match exactly with training data
        if isinstance(future_df.columns, pd.MultiIndex):
            future_df.columns = future_df.columns.get_level_values(0)
        future_df.columns = future_df.columns.astype(str).str.strip()

        # ✅ Step 5: Make Predictions
        future_predictions = model.predict(future_df)

        # ✅ Step 6: Show Results
        result_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})
        st.write("📊 **Predicted Prices:**", result_df)
        st.line_chart(result_df.set_index("Date"))

