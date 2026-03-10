# 2019116278 강민규

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(42)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32')

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='ReLU', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='ReLU'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='ReLU'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='ReLU'))
model.add(layers.Dropout(0.5))  # Dropout 추가
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("테스트 데이터의 손실값: %.2f"%test_loss)
print("테스트 데이터의 정확도: %.2f"%test_acc)
