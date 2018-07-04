import math
import matplotlib.pyplot as plt
import numpy as np

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

