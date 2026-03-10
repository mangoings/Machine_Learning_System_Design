# 20192116278 강민규

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications.inception_v3 import InceptionV3
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train_preprocess = []
for i, img in enumerate(x_train[:10000]):
    img_resize = resize(img, (75, 75), anti_aliasing=True)
    x_train_preprocess.append(np.dstack([img_resize, img_resize, img_resize]))
x_train_preprocess = np.array(x_train_preprocess)

x_val_preprocess = []
for i, img in enumerate(x_val[:2000]):  
    img_resize = resize(img, (75, 75), anti_aliasing=True)
    x_val_preprocess.append(np.dstack([img_resize, img_resize, img_resize]))
x_val_preprocess = np.array(x_val_preprocess)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_preprocess, y_train[:10000], epochs=10, validation_data=(x_val_preprocess, y_val[:2000]))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss, marker='o', label='Training loss')
plt.plot(val_loss, marker='o', label='Validation loss')
plt.title('MNIST loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc, marker='o', label='Training accuracy')
plt.plot(val_acc, marker='o', label='Validation accuracy')
plt.title('MNIST accuracy')
plt.legend()

plt.tight_layout()
plt.show()
