# 2019116278 강민규

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def sine_data(start_points, seq_length=51):
    x = np.linspace(0, 10, seq_length)
    data = [np.sin(x + start) for start in start_points]
    return np.array(data)

seq_length = 51
num_sequences = 100
n_units = 10
epochs = 50

start_points = np.random.uniform(0, 2*np.pi, num_sequences)
data = sine_data(start_points)

X = data[:, :-1]
y = data[:, -1]

train_size = int(num_sequences * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

LSTM_model = Sequential([
    LSTM(units=n_units, return_sequences=False, input_shape=(seq_length-1, 1)),
    Dense(1)
])

LSTM_model.compile(optimizer='adam', loss='mse')

history = LSTM_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

train_predict = LSTM_model.predict(X_train)
test_predict = LSTM_model.predict(X_test)

train_mse = np.mean((y_train - train_predict.squeeze())**2)
test_mse = np.mean((y_test - test_predict.squeeze())**2)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(2, 2, 2)
plt.scatter(y_train, train_predict, color='blue', label='train', alpha=0.5)
plt.scatter(y_test, test_predict, color='red', label='test', alpha=0.5)
plt.xlabel('y')
plt.ylabel('y_hat')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(y_train, label='y', color='black', linewidth=2)
plt.plot(train_predict, label='y_hat', color='red')
plt.title('Train')
plt.legend()
plt.text(0, 0.75, f'MSE: {train_mse:.6f}')

plt.subplot(2, 2, 4)
plt.plot(y_test, label='y', color='black', linewidth=2)
plt.plot(test_predict, label='y_hat', color='red')
plt.title('Test')
plt.legend()
plt.text(0, 0.75, f'MSE: {test_mse:.6f}')

plt.tight_layout()
plt.show()
