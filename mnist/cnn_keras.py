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

print(model.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
reshape (Reshape)            (None, 28, 28, 1)         0
_________________________________________________________________
layer_conv1 (Conv2D)         (None, 28, 28, 16)        416 = 5 * 5 * 16 + 16
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0
_________________________________________________________________
layer_conv2 (Conv2D)         (None, 14, 14, 36)        14436 = 5 * 5 * 16 * 36 + 36
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 36)          0
_________________________________________________________________
flatten (Flatten)            (None, 1764)              0
_________________________________________________________________
dense (Dense)                (None, 128)               225920 = 1764 * 128 + 128
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290 = 128 * 10 + 10
=================================================================
Total params: 242,062
Trainable params: 242,062
Non-trainable params: 0

<tensorflow.python.keras.layers.core.Reshape object at 0x128c7c898>,
<tensorflow.python.keras.layers.convolutional.Conv2D object at 0x128c7c518>,
<tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x128c7c940>,
<tensorflow.python.keras.layers.convolutional.Conv2D object at 0x1321bf9e8>,
<tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x1321ebcf8>,
<tensorflow.python.keras.layers.core.Flatten object at 0x1321ebf98>,
<tensorflow.python.keras.layers.core.Dense object at 0x1321ebd30>,
<tensorflow.python.keras.layers.core.Dense object at 0x1321ebfd0>]
'''

layer_conv1 = model.layers[1]
weights_conv1 = layer_conv1.get_weights()[0]
print('conv1 weights shape: {}'.format(weights_conv1.shape)) ## [5, 5, 1, 16]
input_channels = weights_conv1.shape[2]
for i in range(input_channels):
	name = 'conv1_weights_of_channel_{}'.format(i)
	print('plotting {}'.format(name))
	udf.plot_conv_weights(name, weights=weights_conv1, input_channel=i, save_name='{}.png'.format(name))

layer_conv2 = model.layers[3]
weights_conv2 = layer_conv2.get_weights()[0]
print('conv2 weights shape: {}'.format(weights_conv2.shape)) ## [5, 5, 16, 36]
input_channels = weights_conv2.shape[2]
for i in range(input_channels):
	name = 'conv2_weights_of_channel_{}'.format(i)
	print('plotting {}'.format(name))
	udf.plot_conv_weights(name, weights=weights_conv2, input_channel=i, save_name='{}.png'.format(name))

image1 = data.test.images[0]
udf.plot_image(image1, img_shape)
output_conv1 = keras.backend.function(inputs=[model.layers[0].input], outputs=[layer_conv1.output])
layer_output1 = output_conv1([[image1]])[0]
print('layer_output1.shape: {}'.format(layer_output1.shape)) ## layer_output1.shape: (1, 28, 28, 16)
udf.plot_conv_output('layer_output1', values=layer_output1)

output_conv2 = keras.backend.function(inputs=[model.layers[2].input], outputs=[layer_conv2.output])
print(output_conv2)

sys.exit(0)

