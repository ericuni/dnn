import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_images(title, images, img_shape, cls_true, cls_pred=None):
	num = images.shape[0]
	size = math.ceil(math.sqrt(num))

	# Create figure with size * size sub-plots.
	fig, axes = plt.subplots(size, size)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		if i < num:
			ax.imshow(images[i].reshape(img_shape), cmap='binary')

			if cls_pred is None:
				xlabel = "True: {0}".format(cls_true[i])
			else:
				xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
			ax.set_xlabel(xlabel)

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	plt.suptitle(title)
	plt.show()


def plot_confusion_matrix(title, confusion_matrix, num_classes):
	plt.imshow(confusion_matrix, interpolation = 'nearest', cmap = plt.cm.Blues)
	plt.tight_layout()
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes))
	plt.yticks(tick_marks, range(num_classes))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title(title)
	plt.show()


'''
Positive weights are red and negative weights are blue. These weights can be intuitively understood as image-filters.
For example, the weights used to determine if an image shows a zero-digit have a positive reaction (red) to an image of a circle, and have a negative reaction (blue) to images with content in the centre of the circle.
Similarly, the weights used to determine if an image shows a one-digit react positively (red) to a vertical line in the centre of the image, and react negatively (blue) to images with content surrounding that line.
After training on several thousand images, the weights become more difficult to interpret because they have to recognize many variations of how digits can be written.
'''
def plot_weights(title, weights, img_shape):
	w_min = np.min(weights)
	w_max = np.max(weights)

	num_classes = weights.shape[1]
	size = math.ceil(math.sqrt(num_classes))
	fig, axes = plt.subplots(size, size)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

	for i, ax in enumerate(axes.flat):
		if i < num_classes:
			image = weights[:, i].reshape(img_shape)
			ax.set_xlabel('Weights: {0}'.format(i))
			ax.imshow(image, vmin = w_min, vmax = w_max, cmap = 'seismic')
		ax.set_xticks([])
		ax.set_yticks([])
	plt.suptitle(title)
	plt.show()


def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))


def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape = [length]))


'''
It is assumed that the input is a 4-dim tensor with the following dimensions:
1. image number.
2. Y-axis of each image.
3. X-axis of each image.
4. Channels of each image.
Note that the input channels may either be colour-channels, or it may be filter-channels if the input is produced from a previous convolutional layer.

The output is another 4-dim tensor with the following dimensions:
1. image number, same as input.
2. Y-axis of each image. If 2x2 pooling is used, then the height and width of the input images is divided by 2.
3. X-axis of each image. Ditto.
4. Channels produced by the convolutional filters.
'''
def new_conv_layer(input,			  # The previous layer.
				   num_input_channels, # Num. channels in prev. layer.
				   filter_size,		# Width and height of each filter.
				   num_filters,		# Number of filters.
				   use_pooling=True):  # Use 2x2 max-pooling.

	# Shape of the filter-weights for the convolution.
	# This format is determined by the TensorFlow API.
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	# Create new weights aka. filters with the given shape.
	weights = new_weights(shape)

	# Create new biases, one for each filter.
	biases = new_biases(num_filters)

	# Create the TensorFlow operation for convolution.
	# Note the strides are set to 1 in all dimensions.
	# The first and last stride must always be 1, because the first is for the image-number and the last is for the input-channel.
	# But e.g. strides=[1, 2, 2, 1] would mean that the filter is moved 2 pixels across the x- and y-axis of the image.
	# The padding is set to 'SAME' which means the input image is padded with zeroes so the size of the output is the same.
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

	# Add the biases to the results of the convolution.
	# A bias-value is added to each filter-channel.
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


'''
A convolutional layer produces an output tensor with 4 dimensions.
We will add fully-connected layers after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.
'''
def flatten_layer(layer):
	# Get the shape of the input layer.
	layer_shape = layer.get_shape()

	# The shape of the input layer is assumed to be: layer_shape == [num_images, img_height, img_width, num_channels]
	# The number of features is: img_height * img_width * num_channels
	# We can use a function from TensorFlow to calculate this.
	num_features = layer_shape[1:4].num_elements()

	# Reshape the layer to [num_images, num_features].
	# Note that we just set the size of the second dimension to num_features and the size of the first dimension to -1 which means the size in that dimension is calculated
	# so the total size of the tensor is unchanged from the reshaping.
	layer_flat = tf.reshape(layer, [-1, num_features])

	# The shape of the flattened layer is now: [num_images, img_height * img_width * num_channels]
	# Return both the flattened layer and the number of features.
	return layer_flat, num_features


'''
create a new fully connected layer
It is assumed that the input is a 2-dim tensor of shape [num_images, num_inputs]. The output is a 2-dim tensor of shape [num_images, num_outputs].
'''
def new_fc_layer(input,		  # The previous layer.
				 num_inputs,	 # Num. inputs from prev. layer.
				 num_outputs,	# Num. outputs.
				 use_relu=True): # Use Rectified Linear Unit (ReLU)?

	# Create new weights and biases.
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	# Calculate the layer as the matrix multiplication of the input and weights, and then add the bias-values.
	layer = tf.matmul(input, weights) + biases

	# Use ReLU?
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer


