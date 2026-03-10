# 2019116278 강민규

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 5.0
test_images = test_images / 5.0

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='ReLU'),
    Dropout(0.2),
    Dense(32, activation='ReLU'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test 정확도: {test_accuracy:.8f}')

plt.figure(figsize=(12, 6))

epochs = range(1, len(history.history['loss']) + 1)

plt.plot(epochs, history.history['loss'], 'b-', label='train')
plt.plot(epochs, history.history['val_loss'], 'r--', label='validation')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(epochs, history.history['accuracy'], 'b-', label='training')
plt.plot(epochs, history.history['val_accuracy'], 'r--', label='validation')

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Dictionary mapping label to description
label_description = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Get the first 25 images from the test dataset
first_25_images = test_images[:25]
first_25_labels = test_labels[:25]

# Predict the labels for the first 25 images
predictions = model.predict(first_25_images)

plt.figure(figsize=(12, 12))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(first_25_images[i], cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = first_25_labels[i]
    plt.title(f"{label_description[true_label]}")

plt.tight_layout()
plt.show()

