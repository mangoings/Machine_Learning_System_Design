# 2019116278 강민규

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler

def sine_data(start_points, seq_length=51):
    x = np.linspace(0, 10, seq_length)
    data = [np.sin(x + start) for start in start_points]
    return np.array(data)

seq_length = 51
num_sequences = 100
start_points = np.random.uniform(0, 2*np.pi, num_sequences)

data = sine_data(start_points)

X = data[:, :-1]
y = data[:, -1]

train_size = int(num_sequences * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def create_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'RNN':
        model.add(SimpleRNN(50, activation='tanh', input_shape=input_shape))
    elif model_type == 'LSTM':
        model.add(LSTM(50, activation='tanh', input_shape=input_shape))
    elif model_type == 'GRU':
        model.add(GRU(50, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_predict(model_type, X_train, y_train, X_test):
    model = create_model(model_type, (X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    predictions = model.predict(X_test)
    return predictions

rnn_predictions = train_predict('RNN', X_train[..., np.newaxis], y_train, X_test[..., np.newaxis])
lstm_predictions = train_predict('LSTM', X_train[..., np.newaxis], y_train, X_test[..., np.newaxis])
gru_predictions = train_predict('GRU', X_train[..., np.newaxis], y_train, X_test[..., np.newaxis])

np.random.seed(0)
random_index1 = np.random.randint(0, X_test.shape[0])
random_index2 = np.random.randint(0, X_test.shape[0])
random_index3 = np.random.randint(0, X_test.shape[0])

def plot_prediction(index, prediction, y_test, X_test):
    plt.figure(figsize=(10, 5))
    plt.scatter(np.linspace(0, 10, 50), X_test[index])
    plt.scatter(10.2, prediction[index], color='orange')
    plt.show()

plot_prediction(random_index1, rnn_predictions, y_test, X_test)
plot_prediction(random_index2, lstm_predictions, y_test, X_test)
plot_prediction(random_index3, gru_predictions, y_test, X_test)
