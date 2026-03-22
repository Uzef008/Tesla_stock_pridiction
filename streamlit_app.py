# Tesla Stock Prediction (ONNX - FINAL STABLE)

import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------------
# Load ONNX Model
# -------------------------------
session = ort.InferenceSession("lstm_model.onnx")

input_name = session.get_inputs()[0].name

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("TSLA.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

data = df[['Close']]

# -------------------------------
# Scale Data
# -------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

seq_length = 60

# -------------------------------
# UI
# -------------------------------
st.title("📈 Tesla Stock Price Prediction (ONNX)")

st.subheader("📊 Last 100 Days")
st.line_chart(data.tail(100))

days = st.selectbox("Predict days:", [1, 2, 3, 5, 10, 12, 15, 20])

# -------------------------------
# Prediction Function
# -------------------------------
def predict_future(session, last_sequence, days):
    future_predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        pred = session.run(None, {input_name: current_seq.reshape(1, 60, 1).astype(np.float32)})
        pred_value = pred[0][0][0]

        future_predictions.append(pred_value)
        current_seq = np.append(current_seq[1:], [[pred_value]], axis=0)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# -------------------------------
# Button
# -------------------------------
if st.button("🚀 Predict"):
    last_sequence = scaled_data[-60:]
    result = predict_future(session, last_sequence, days)

    st.success("Prediction Done!")

    for i, val in enumerate(result, 1):
        st.write(f"Day {i}: {val[0]:.2f}")

    fig, ax = plt.subplots()
    ax.plot(result)
    ax.set_title("Future Predictions")
    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👨‍💻 Developed by Uzef (ONNX Deployment)")
