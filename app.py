import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load trained model
model = load_model("gru_model.keras")

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
data = df['Close'].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Take last 60 days
last_60_days = scaled_data[-60:]

# -------------------------------
# 30-DAY PREDICTION
# -------------------------------
future_predictions = []
current_input = last_60_days.copy()

for i in range(30):
    current_input_reshaped = current_input.reshape(1, 60, 1)
    
    pred = model.predict(current_input_reshaped, verbose=0)
    future_predictions.append(pred[0][0])
    
    # update input (remove first, add new prediction)
    current_input = np.vstack((current_input[1:], pred))

# Convert predictions to array
future_predictions = np.array(future_predictions).reshape(-1,1)

# Convert back to original prices
future_predictions = scaler.inverse_transform(future_predictions)

# -------------------------------
# CREATE OUTPUT TABLE
# -------------------------------
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=31)[1:]

future_df = pd.DataFrame(
    future_predictions,
    columns=['Predicted Price'],
    index=future_dates
)

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