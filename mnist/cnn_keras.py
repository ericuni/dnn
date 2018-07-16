import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import udf
from mnist_data import *
from tensorflow.python import keras

print(tf.__version__)
print(tf.keras.__version__)

data = load_data()

# Start construction of the Keras Sequential model.
model = keras.models.Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
model.add(keras.layers.InputLayer(input_shape=(img_size_flat,)))

# The input is a flattened array with 784 elements, but the convolutional layers expect images with shape (28, 28, 1)
img_shape_full = (img_size, img_size, 1)
model.add(keras.layers.Reshape(img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(keras.layers.Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(keras.layers.Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers to 2-rank that can be input to a fully-connected / dense layer.
model.add(keras.layers.Flatten())

# First fully-connected / dense layer with ReLU-activation.
model.add(keras.layers.Dense(128, activation='relu'))

# Last fully-connected / dense layer with softmax-activation for use in classification.
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=data.train.images, y=data.train.labels, epochs=1, batch_size=128)

result = model.evaluate(x=data.test.images, y=data.test.labels)
for name, value in zip(model.metrics_names, result):
	print(name, value)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

images = data.validation.images[0:9]
cls_true = data.validation.cls[0:9]
y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
print('true: {}'.format(cls_true))
print('pred: {}'.format(cls_pred))

sys.exit(0)

# needs `pip3 install --upgrade h5py` to save model 
path_model = 'model.keras'
model.save(path_model)
del model

from tensorflow.python.keras.models import load_model
model3 = load_model(path_model) ## load will error
images = data.test.images[0:9]
y_pred = model3.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
udf.plot_images(images=images, cls_pred=cls_pred, cls_true=cls_true)

