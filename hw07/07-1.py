# 2019116278 강민규

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, batch_size=5, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Iris 데이터의 분류 정확도: {test_accuracy:.8f}')

plt.figure(figsize=(12, 6))

epochs = range(1, len(history.history['loss']) + 1)

plt.plot(epochs, history.history['loss'], color='b', label='loss value')
plt.plot(epochs, history.history['accuracy'], color='r', label='accuracy')

plt.title('Training and Validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.show()

