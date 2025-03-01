import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import ta  # Ensure this is installed with 'pip install ta'

# 🎯 Streamlit Web UI
st.title("📈 Stock & Crypto AI Predictor")

# 📊 STOCK PREDICTION SECTION
st.header("📉 Stock Market AI Predictor")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOGL)")

if st.button("Predict Stock Price"):
    if ticker:
        try:
            # ✅ Get Stock Data
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            
            if data.empty:
                st.error("No data found! Try another stock ticker.")
            else:
                # 📊 Add Technical Indicators
                data['MA10'] = data['Close'].rolling(window=10).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
                data['MACD'] = ta.trend.MACD(data['Close']).macd()
                
                data = data.dropna()
                
                # 🎯 Define Features & Target
                X = data[['MA10', 'MA50', 'RSI', 'MACD']]
                y = data['Close']
                
                # 🚀 Train Model
                model = LinearRegression()
                model.fit(X, y)
                
                # 🔮 Predict Next Price
                latest_data = X.iloc[-1:].values.reshape(1, -1)
                predicted_price = model.predict(latest_data)[0]
                
                # ✅ Display Prediction
                st.success(f"📊 Predicted Stock Price for {ticker}: ${predicted_price:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid stock ticker!")

# 💰 CRYPTOCURRENCY PREDICTION SECTION
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import ta  # Ensure you installed this with 'pip install ta'

st.title("💰 Crypto Market AI Predictor")

crypto_ticker = st.text_input("Enter Crypto Pair (e.g., BTC-USD, ETH-USD)")

if st.button("Predict Crypto Price"):
    if crypto_ticker:
        try:
            # ✅ Fetch Crypto Data from Yahoo Finance
            crypto = yf.Ticker(crypto_ticker)
            crypto_data = crypto.history(period="1y")

            if crypto_data.empty:
                st.error("No data found! Try another crypto pair like BTC-USD.")
            else:
                # 📊 Add Moving Averages
                crypto_data['MA10'] = crypto_data['Close'].rolling(window=10).mean()
                crypto_data['MA50'] = crypto_data['Close'].rolling(window=50).mean()

                # Drop NaN values
                crypto_data = crypto_data.dropna()

                # 🎯 Define Features & Target
                X_crypto = crypto_data[['MA10', 'MA50']]
                y_crypto = crypto_data['Close']

                # 🚀 Define & Train Model
                model = LinearRegression()  # <-- This was missing!
                model.fit(X_crypto, y_crypto)

                # 🔮 Predict Next Crypto Price
                latest_crypto_data = X_crypto.iloc[-1:].values.reshape(1, -1)
                predicted_crypto_price = model.predict(latest_crypto_data)[0]

                # ✅ Display Prediction
                st.success(f"📈 Predicted Crypto Price for {crypto_ticker}: ${predicted_crypto_price:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid crypto pair!")

