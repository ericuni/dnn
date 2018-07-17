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


def plot_conv_weights(title, weights, input_channel=0, save_name=None):
	# Get the lowest and highest values for the weights.
	# This is used to correct the colour intensity across the images so they can be compared with each other.
	w_min = np.min(weights)
	w_max = np.max(weights)

	# Number of filters used in the conv. layer.
	# shape = [filter_size, filter_size, num_input_channels, num_filters]
	num_filters = weights.shape[3]

	# Number of grids to plot.
	# Rounded-up, square-root of the number of filters.
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots.
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot all the filter-weights.
	for i, ax in enumerate(axes.flat):
		# Only plot the valid filter-weights.
		if i < num_filters:
			# Get the weights for the i'th filter of the input channel.
			# See new_conv_layer() for details on the format of this 4-dim tensor.
			img = weights[:, :, input_channel, i]

			# Plot image.
			ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	plt.suptitle(title)
	if save_name is not None:
		plt.savefig(save_name)
		plt.close()
	else:
		plt.show()


'''
values: [channels, width, height, filters]
in this function, channels is set 0
'''
def plot_conv_output(title, values, save_name=None):
	# Number of filters used in the conv. layer.
	num_filters = values.shape[3]

	# Number of grids to plot.
	# Rounded-up, square-root of the number of filters.
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots.
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot the output images of all the filters.
	for i, ax in enumerate(axes.flat):
		# Only plot the images for valid filters.
		if i < num_filters:
			# Get the output image of using the i'th filter.
			img = values[0, :, :, i]

			# Plot image.
			ax.imshow(img, interpolation='nearest', cmap='binary')

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	plt.suptitle(title)
	if save_name is not None:
		plt.savefig(save_name)
		plt.close()
	else:
		plt.show()


def plot_image(image, img_shape):
	plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
	plt.show()

