import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense

df = pd.read_csv('GOOGL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features]

sequence_length = 60
training_size = int(len(data) * 0.8)

train_data = data.iloc[:training_size]
test_data = data.iloc[training_size - sequence_length:]

scaler = MinMaxScaler()
scaler.fit(train_data)

scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)

x_train, y_train = [], []
for i in range(sequence_length, len(scaled_train)):
    x_train.append(scaled_train[i-sequence_length:i])
    y_train.append(scaled_train[i, features.index('Close')])

x_train, y_train = np.array(x_train), np.array(y_train)

x_test, y_test = [], []
for i in range(sequence_length, len(scaled_test)):
    x_test.append(scaled_test[i-sequence_length:i])
    y_test.append(scaled_test[i, features.index('Close')])

x_test, y_test = np.array(x_test), np.array(y_test)

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    GRU(64),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=30, batch_size=32)

predictions = model.predict(x_test)

close_index = features.index('Close')

pred_full = np.zeros((len(predictions), len(features)))
true_full = np.zeros((len(y_test), len(features)))

pred_full[:, close_index] = predictions[:, 0]
true_full[:, close_index] = y_test

predictions_rescaled = scaler.inverse_transform(pred_full)[:, close_index]
y_test_rescaled = scaler.inverse_transform(true_full)[:, close_index]

rmse = np.sqrt(np.mean((predictions_rescaled - y_test_rescaled) ** 2))
print(f'RMSE: {rmse:.2f}')

plt.figure(figsize=(14,6))
plt.plot(y_test_rescaled, label='Actual Price')
plt.plot(predictions_rescaled, label='Predicted Price')
plt.title('GOOGL Stock Price Prediction (GRU)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
