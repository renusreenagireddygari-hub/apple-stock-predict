import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------- LOAD MODEL --------------------
model = load_model("gru_model.keras")

# -------------------- TITLE --------------------
st.title("📊 Apple Stock Price Prediction Dashboard")

# -------------------- DOWNLOAD DATA --------------------
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

st.subheader("📄 Recent Stock Data")
st.write(df.tail())

# -------------------- ACTUAL PRICE CHART --------------------
st.subheader("📈 Closing Price Chart")

fig = plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Close Price")
st.pyplot(fig)

# -------------------- DATA PREPARATION --------------------
data = df['Close'].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

last_60_days = scaled_data[-60:]

# -------------------- FUTURE PREDICTION --------------------
st.subheader("🔮 30-Day Future Stock Forecast")

future_predictions = []
current_input = last_60_days.copy()

for i in range(30):
    current_input_reshaped = current_input.reshape(1, 60, 1)
    pred = model.predict(current_input_reshaped, verbose=0)
    future_predictions.append(pred[0][0])

    # sliding window
    current_input = np.vstack((current_input[1:], pred))

# convert & inverse scale
future_predictions = np.array(future_predictions).reshape(-1,1)
future_predictions = scaler.inverse_transform(future_predictions)

# -------------------- FUTURE DATAFRAME --------------------
future_df = pd.DataFrame(future_predictions, columns=['Predicted Price'])

# ✅ Add real future dates
future_dates = pd.date_range(start=df.index[-1], periods=31, freq='B')[1:]
future_df.index = future_dates

st.write(future_df)

# -------------------- FORECAST CHART --------------------
st.subheader("📊 Forecast Chart")

fig2 = plt.figure(figsize=(10,5))
plt.plot(future_df.index, future_df['Predicted Price'], marker='o', color='red')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Next 30 Days Prediction")
st.pyplot(fig2)

# -------------------- COMPARISON CHART --------------------
st.subheader("📊 Actual vs Forecast Comparison")

last_100 = df['Close'].tail(100)

fig3 = plt.figure(figsize=(12,6))
plt.plot(last_100.index, last_100.values, label='Actual Price', color='blue')
plt.plot(future_df.index, future_df['Predicted Price'], label='Forecast', color='red')

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs Predicted Stock Price")
plt.legend()
plt.xticks(rotation=45)

st.pyplot(fig3)

st.info("Blue = Historical Prices | Red = Model Forecast")

# -------------------- BUY / SELL SIGNAL --------------------
last_pred = float(np.squeeze(future_df['Predicted Price'].iloc[-1]))
last_actual = float(np.squeeze(df['Close'].iloc[-1]))

st.write("Last Predicted:", last_pred)
st.write("Last Actual:", last_actual)

# -------------------- DOWNLOAD BUTTON --------------------
csv = future_df.to_csv().encode('utf-8')
st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")