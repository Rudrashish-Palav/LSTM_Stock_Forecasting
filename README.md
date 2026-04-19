# Stock Price Forecasting using LSTM

This repository contains a machine learning project that predicts Microsoft's stock price using a Long Short-Term Memory (LSTM) model.

## Model Overview

The prediction is performed using a **sliding window** approach.

### Sliding Window Configuration:
- **Window Size**: 60 days
- **Prediction Horizon**: 1 day

This means the model takes the stock's closing prices from the previous **60 days** as input to forecast the price for the **very next day**.

## Dataset
The model uses `MicrosoftStock.csv`, which contains historical stock data including Open, High, Low, Close prices, and Volume.

## Implementation Details
The project is implemented in the `LSTM_Model.ipynb` Jupyter Notebook. It includes:
- Data preprocessing and scaling using `StandardScaler`.
- Construction of a sequential LSTM model with two LSTM layers followed by Dense and Dropout layers.
- Visualization of actual vs. predicted stock prices.
