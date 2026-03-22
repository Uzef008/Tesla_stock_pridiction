# 🚀 Tesla Stock Price Prediction using Deep Learning

## 📌 Overview
This project predicts Tesla stock closing prices using Deep Learning models such as **SimpleRNN** and **LSTM**. Since stock prices are time-series data, these models help capture temporal dependencies and trends.

The application is deployed using **Streamlit** and the model is converted to **ONNX** format for better compatibility.

---

## 🎯 Objectives
- Predict Tesla stock closing price
- Forecast future prices for:
  - 1 day
  - 5 days
  - 10 days
- Compare performance of RNN and LSTM models
- Deploy an interactive web app

---

## 📊 Dataset
The dataset contains historical Tesla stock data with the following features:
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

👉 This project focuses on the **Close price** for prediction.

---

## ⚙️ Technologies Used
- Python
- Pandas & NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- ONNX (for deployment)
- Streamlit (for web app)

---

## 🧠 Model Details

### 🔹 SimpleRNN
- Captures short-term dependencies
- Faster but less powerful

### 🔹 LSTM
- Handles long-term dependencies
- Better for sequential/time-series data

---

## 🔄 Workflow

1. Data Cleaning  
2. Data Preprocessing (Scaling & Sequence Creation)  
3. Data Visualization  
4. Model Building (RNN & LSTM)  
5. Model Evaluation (MSE)  
6. Prediction  
7. Deployment using Streamlit  

---

## 📈 Results

| Model | Performance |
|------|-----------|
| RNN | Lower MSE (better in this case) |
| LSTM | Slightly higher MSE |

👉 Note: Stock prediction is highly volatile, so models capture trends rather than exact values.

---

## 🌐 Live Demo
👉 (Add your Streamlit link here after deployment)

---

## 💻 How to Run Locally

```bash
git clone https://github.com/your-username/tesla-stock-prediction.git
cd tesla-stock-prediction
pip install -r requirements.txt
streamlit run app.py
