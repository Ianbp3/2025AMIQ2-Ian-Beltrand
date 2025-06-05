import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

#Data Import ---------------------------------------------------------
data = pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
data['date'] = pd.to_datetime(data['date'])

google = data[data['Name'] == 'GOOGL']
prediction_range = google.loc[(google['date'] > datetime(2013,1,1))
 & (google['date']<datetime(2018,1,1))]

close_data = google.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))

min_val = np.min(dataset)
max_val = np.max(dataset)
scaled_data = (dataset - min_val) / (max_val - min_val)
train_data = scaled_data[0:int(training), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping for model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=50, input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10)

