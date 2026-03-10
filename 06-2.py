# 2019116278 강민규

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# 데이터 로드
path = 'C:\\Users\\kui45\\'
data = pd.read_csv(path + 'nonlinear-1.csv')

X = data['x'].to_numpy().reshape(-1, 1)
y_label = data['y'].to_numpy().reshape(-1, 1)

# 모델 구축
model = keras.Sequential([
    keras.layers.Dense(6, activation='tanh', input_shape=(1,)),
    keras.layers.Dense(4, activation='tanh'),
    keras.layers.Dense(1, activation='tanh')
])

# 모델 컴파일
optimizer = keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='mse')

# 모델 학습
history = model.fit(X, y_label, epochs=100, verbose=0)

# 예측
domain = np.linspace(0, 1, 100).reshape(-1, 1)
y_hat = model.predict(domain)

# 그래프 그리기
plt.scatter(X, y_label)
plt.scatter(domain, y_hat, color='r')
plt.show()