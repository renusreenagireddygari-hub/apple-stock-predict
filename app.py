import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
import streamlit as st

st.write("FILES:", os.listdir())

# Load trained model
future_df = pd.read_csv("data/forecast.csv", index_col=0)
# App Title
st.title("Apple Stock Price Prediction App")
st.header("📊 Apple Stock Prediction Dashboard")
# Download stock data
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

# Display data
st.subheader("Recent Stock Data")
st.write(df.tail())

# Plot closing price
st.subheader("Closing Price Chart")
fig = plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Close Price")
st.pyplot(fig)

# -------------------------------
# DATA PREPARATION
# -------------------------------

# Show predictions
st.subheader("30-Day Future Stock Forecast")
st.write(future_df)
st.subheader("📊 Actual vs Forecast Comparison")

# Last 100 actual data
last_100 = df['Close'].tail(100)

# Combine actual + future
combined = pd.concat([last_100, future_df['Predicted Price']])

# Plot
fig3 = plt.figure(figsize=(12,6))

plt.plot(last_100.index, last_100.values, label='Actual Price', color='blue')
plt.plot(future_df.index, future_df['Predicted Price'], label='Forecast', color='red')

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs Predicted Stock Price")
plt.legend()

plt.xticks(rotation=45)
st.pyplot(fig3)
st.write(type(future_df['Predicted Price'].iloc[-1]))
st.write(type(df['Close'].iloc[-1]))
last_pred = future_df['Predicted Price'].iloc[-1].item()
last_actual = df['Close'].iloc[-1].item()
st.write("Last Predicted:", last_pred)
st.write("Last Actual:", last_actual)
if last_pred > last_actual:
    st.success("📈 Buy Signal")
else:
    st.error("📉 Sell Signal")

# -------------------------------
# PLOT FORECAST
# -------------------------------
st.subheader("Forecast Chart")

fig2 = plt.figure(figsize=(10,5))
plt.plot(future_df['Predicted Price'], marker='o')
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.title("Next 30 Days Forecast")
plt.xticks(rotation=45)
st.pyplot(fig2)
future_df.to_csv("forecast.csv")
