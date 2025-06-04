# Stock Price Prediction Using Recurrent Neural Networks (RNN)

## Overview
This project aims to predict future stock prices using Recurrent Neural Networks (RNN), a type of neural network that is well-suited for sequential data. The initial model will use RNNs in TensorFlow, with plans to explore Long Short-Term Memory (LSTM) networks for better performance in capturing long-term dependencies in stock price data.

## Project Setup

### Technologies Used:
- **TensorFlow**: A deep learning library used to implement and train the RNN model.
- **Keras**: High-level API to build the neural network in TensorFlow.
- **Pandas**: To handle and preprocess the dataset.
- **NumPy**: For numerical computations and data manipulation.
- **Matplotlib / Seaborn**: For data visualization.

### Current Dataset:
The stock data could be fetched from here https://media.geeksforgeeks.org/wp-content/uploads/20250408133653340682/all_stocks_5yr.csv . The dataset include daily stock prices and key indicators like the following:
- **Open**: The price of the stock at market open.
- **Close**: The price of the stock at market close.
- **High**: The highest price reached during the trading day.
- **Low**: The lowest price reached during the trading day.
- **Volume**: The number of shares traded on a particular day.

#### Dataset Change Notice:
While the project currently plans to use a general stock market dataset, there is a possibility that the dataset will be changed to something more specific in the future (e.g., a particular stock, industry, or region). This will be updated as the project progresses.

## Objective
The goal is to train a model that can predict the **closing price** of a stock for the next day based on the previous day's data. We aim to explore the following tasks:
- **Preprocessing the dataset**: Normalization, handling missing data, and feature engineering.
- **Building an RNN model**: Initial tests with basic RNN architecture in TensorFlow.
- **Evaluating performance**: Using metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).
- **Transitioning to LSTM models**: To handle longer sequences and improve accuracy.

## Stock Market Data Explanation
Stock market data generally includes several columns, each representing a specific aspect of the trading day:

1. **Open Price**:
   - The price of a stock when the market opens for the day. It’s often seen as a baseline for measuring the day's price movement.
  
2. **Close Price**:
   - The price of a stock when the market closes for the day. It’s considered the most important price as it reflects the value of the stock at the end of trading and is often used to compare price changes over time.

3. **High Price**:
   - The highest price at which a stock traded during the day. It can indicate the peak of demand during the trading session.

4. **Low Price**:
   - The lowest price at which a stock traded during the day. It reflects the lowest point the stock reached, showing the trough of trading activity.

5. **Volume**:
   - The number of shares traded during the day. A high volume often indicates strong investor interest, whereas low volume may suggest less market activity.

These features are used to understand the price action and trading dynamics of the stock. The model will attempt to predict future stock prices by learning from the past sequence of open, close, high, low, and volume data.

## Model Description

### Recurrent Neural Network (RNN)
RNNs are neural networks designed for sequence prediction tasks, such as time series forecasting. In this project, the RNN model will take a sequence of past stock prices and use them to predict future prices. The RNN has a hidden state that keeps track of information about previous time steps.

#### Transition to LSTM
Long Short-Term Memory (LSTM) networks are a special type of RNN that are particularly good at learning long-range dependencies. They mitigate issues such as the vanishing gradient problem and are well-suited for financial time-series data, which can have complex dependencies over long periods.

In future iterations of this project, we will experiment with LSTM models to see if they improve the accuracy of stock price predictions.

## Potential Project Steps

### 1. Data Collection:
   - Gather historical stock data (open, close, high, low, volume).
   - Clean and preprocess the data (handle missing values, normalize features).

### 2. Feature Engineering:
   - Choose relevant features (e.g., moving averages, technical indicators) to help improve model performance.

### 3. Build RNN Model:
   - Create an RNN model using TensorFlow/Keras to predict future stock prices based on previous data.
   - Experiment with hyperparameters like the number of layers, hidden units, and sequence length.

### 4. Train the Model:
   - Train the RNN model using the training dataset.
   - Use validation data to tune the model and prevent overfitting.

### 5. Evaluate the Model:
   - Use error metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE) to evaluate the model's performance.

### 6. Transition to LSTM:
   - Replace the RNN with an LSTM network to improve the model’s ability to capture long-term dependencies in stock price data.

