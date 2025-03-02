import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🎯 Streamlit UI
st.title("📈 AI Stock & Crypto Price Predictor")

# 📌 User Inputs
ticker = st.text_input("Enter Stock/Crypto Symbol (e.g., AAPL, BTC-USD)", "AAPL")
days_to_predict = st.slider("Days to Predict", 1, 30, 7)

if st.button("Predict Price"):
    st.write(f"🔄 Fetching Data for {ticker}...")

    # ✅ Step 1: Get Historical Data
    df = yf.download(ticker, period="10y", interval="1d", auto_adjust=False)

    if df.empty:
        st.error("⚠️ Invalid Ticker Symbol or No Data Available!")
    else:
        st.write("✅ Data Fetched Successfully!")

        # ✅ Step 2: Feature Engineering
        df["Returns"] = df["Close"].pct_change()
        df["SMA10"] = df["Close"].rolling(window=10).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["Volatility"] = df["Returns"].rolling(window=10).std()
        df["Momentum"] = df["Close"] - df["Close"].shift(4)
        df["RSI"] = 100 - (100 / (1 + df["Returns"].rolling(window=14).mean() / df["Returns"].rolling(window=14).std()))
        df["BollingerUpper"] = df["SMA10"] + 2 * df["Volatility"]
        df["BollingerLower"] = df["SMA10"] - 2 * df["Volatility"]
        df["MACD"] = df["SMA10"] - df["SMA50"]

        df.dropna(inplace=True)  # Remove NaN values

        # ✅ Step 3: Train XGBoost Model with Standardization
        feature_names = ["SMA10", "SMA50", "Volatility", "Momentum", "RSI", "BollingerUpper", "BollingerLower", "MACD"]
        target = "Close"

        X = df[feature_names].copy()
        y = df[target].copy()

        # ✅ FIX: Convert MultiIndex to Single Index & Ensure Feature Name Consistency
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = X.columns.to_flat_index()
        
        X.columns = [str(col).strip().replace(" ", "").replace("_", "") for col in X.columns]  # Standardize names

        # ✅ Scale Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        # ✅ Train XGBoost Model
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8
        )
        model.fit(X_train, y_train)

        st.write("✅ Debug: Model Training Completed!")

        # ✅ Step 4: Prepare Future Data for Prediction
        future_dates = pd.date_range(start=df.index[-1], periods=days_to_predict + 1, freq="D")[1:]
        future_df = pd.DataFrame(index=future_dates)

        # ✅ Fix Feature Naming Issue by Ensuring Consistency
        for feature in feature_names:
            clean_feature = str(feature).strip().replace(" ", "").replace("_", "")  # Remove extra characters
            future_df[clean_feature] = df[feature].iloc[-10:].mean() + np.random.normal(0, 0.5, size=len(future_dates))

        st.write(f"✅ Debug: Future Data Sample - {future_df.head()}")

        # ✅ Fix the column mismatch error by making future_df match X.columns exactly
        future_df.columns = X.columns  # This ensures **perfect alignment** with training features

        # ✅ Scale Future Data & Predict
        future_scaled = scaler.transform(future_df)
        future_predictions = model.predict(future_scaled)

        st.write("✅ Debug: Model Prediction Completed!")

        # ✅ Display Predictions
        result_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})
        st.write("📊 **Predicted Prices:**", result_df)
        st.line_chart(result_df.set_index("Date"))

