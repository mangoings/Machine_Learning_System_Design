# 2019116278 강민규

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Input layer
model.add(layers.MaxPooling2D((2, 2)))  # MaxPooling
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Second Conv2D
model.add(layers.MaxPooling2D((2, 2)))  # Second MaxPooling
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Third Conv2D

model.add(layers.Flatten())

model.add(layers.Dense(1568, activation='relu'))
model.add(layers.Dense(128, activation='relu')) 
model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("테스트 데이터의 정확도: %.16f" % test_acc)

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

first_25_images = test_images[:25]
first_25_labels = test_labels[:25]

predictions = model.predict(first_25_images)

plt.figure(figsize=(12, 12))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(first_25_images[i], cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = 3
    plt.title(f"{label_description[true_label]}", fontsize=20)

plt.tight_layout()
plt.show()



