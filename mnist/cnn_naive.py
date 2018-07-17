import numpy as np
import tensorflow as tf
import sys
from mnist_data import *

## data
data = load_data()

## variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

## help functions
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

'''
input: the previous layer
num_input_channels: # of channels in previous layer
filter_size: width and height of each filter
num_filters: # of filters
use_pooling: use 2 * 2 max-pooling
'''
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	# Shape of the filter-weights for the convolution.
	# This format is determined by the TensorFlow API.
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	# Create new weights aka. filters with the given shape.
	weights = new_weights(shape=shape)

	# Create new biases, one for each filter.
	biases = new_biases(length=num_filters)

	# Create the TensorFlow operation for convolution.
	# Note the strides are set to 1 in all dimensions.
	# The first and last stride must always be 1, because the first is for the image-number and the last is for the input-channel.
	# But e.g. strides=[1, 2, 2, 1] would mean that the filter is moved 2 pixels across the x- and y-axis of the image.
	# The padding is set to 'SAME' which means the input image is padded with zeroes so the size of the output is the same.
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

	# Add the biases to the results of the convolution. A bias-value is added to each filter-channel.
	layer += biases

	# Use pooling to down-sample the image resolution?
	if use_pooling:
		# This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value in each window. Then we move 2 pixels to the next window.
		layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	# Rectified Linear Unit (ReLU).
	# It calculates max(x, 0) for each input pixel x.
	# This adds some non-linearity to the formula and allows us to learn more complicated functions.
	layer = tf.nn.relu(layer)

	# Note that ReLU is normally executed before the pooling,
	# but since relu(max_pool(x)) == max_pool(relu(x)) we can save 75% of the relu-operations by max-pooling first.

	# We return both the resulting layer and the filter-weights because we will plot the weights later.
	return layer, weights

def flatten_layer(layer):
	# Get the shape of the input layer.
	layer_shape = layer.get_shape()

	# The shape of the input layer is assumed to be: layer_shape == [num_images, img_height, img_width, num_channels]
	# The number of features is: img_height * img_width * num_channels
	# We can use a function from TensorFlow to calculate this.
	num_features = layer_shape[1:4].num_elements()

	# Reshape the layer to [num_images, num_features].
	# Note that we just set the size of the second dimension to num_features and the size of the first dimension to -1
	# which means the size in that dimension is calculated so the total size of the tensor is unchanged from the reshaping.
	layer_flat = tf.reshape(layer, [-1, num_features])

	# The shape of the flattened layer is now: [num_images, img_height * img_width * num_channels]

	# Return both the flattened layer and the number of features.
	return layer_flat, num_features


'''
input: the previous layer
num_inputs: # of features from the previous layer
num_outputs: # of output features
use_relu: use ReLU
'''
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	# Create new weights and biases.
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	# Calculate the layer as the matrix multiplication of the input and weights, and then add the bias-values.
	layer = tf.matmul(input, weights) + biases

	# Use ReLU?
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

## layers
filter_size1 = 5
num_filters1 = 16
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
filter_size2 = 5
num_filters2 = 36
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
fc_size = 1024
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

## Predicted Class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

## Cost-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

## Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

## Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## train
session = tf.Session()
session.run(tf.global_variables_initializer())

batch_size = 64
iterations = 100
feed_dict_test = {x: data.test.images, y_true_cls: data.test.cls}
for i in range(iterations):
	x_batch, y_true_batch = data.train.next_batch(batch_size)
	feed_dict_train = {x: x_batch, y_true: y_true_batch}
	session.run(optimizer, feed_dict=feed_dict_train)
	if i % 10 == 0 or i == iterations - 1:
		acc = session.run(accuracy, feed_dict = feed_dict_test)
		print('accuracy after {0} iterations: {1}'.format(i, acc))

## predict
result = session.run(y_pred_cls, feed_dict = {x: data.validation.images[0:19]})
print('true: {}'.format(data.validation.cls[0:19]))
print('pred: {}'.format(result))

## Visualization of Weights and Layers
image1 = data.test.images[0]
label1 = data.test.cls[0]
udf.plot_image(image1, img_shape)

image2 = data.test.images[13]
label2 = data.test.cls[13]
udf.plot_image(image2, img_shape)

w = session.run(weights_conv1)
print('weights_conv1 shape: {}'.format(w.shape)) ## (5, 5, 1, 16)
udf.plot_conv_weights('conv1 weights', weights=w)

values = session.run(layer_conv1, feed_dict={x: [image1]})
udf.plot_conv_output('conv1 output of {}'.format(label1), values)

values = session.run(layer_conv1, feed_dict={x: [image2]})
udf.plot_conv_output('conv1 output of {}'.format(label2), values)

w = session.run(weights_conv2)
print('weights_conv2 shape: {}'.format(w.shape)) ## (5, 5, 16, 36)
udf.plot_conv_weights('conv2 weights of channel 0', weights=w, input_channel=0)
## There are 16 input channels to the second convolutional layer, so we can make another 15 plots of filter-weights like this.
## We just make one more with the filter-weights for the second channel.
udf.plot_conv_weights('conv2 weights of channel 1', weights=w, input_channel=1)

values = session.run(layer_conv2, feed_dict={x: [image1]})
udf.plot_conv_output('conv2 output of {}'.format(label1), values)

values = session.run(layer_conv2, feed_dict={x: [image2]})
udf.plot_conv_output('conv2 output of {}'.format(label2), values)

session.close()

