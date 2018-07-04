import matplotlib.pyplot as plt
import numpy as np

def plot_images(title, images, img_shape, cls_true, cls_pred=None, w=3, h=3):
	assert len(images) == len(cls_true) == w * h

	# Create figure with 3x3 sub-plots.
	fig, axes = plt.subplots(w, h)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image.
		ax.imshow(images[i].reshape(img_shape), cmap='binary')

		# Show true and predicted classes.
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		ax.set_xlabel(xlabel)

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# Ensure the plot is shown correctly with multiple plots in a single Notebook cell.
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


def plot_weights(title, weights, img_shape):
	w_min = np.min(weights)
	w_max = np.max(weights)

	fig, axes = plt.subplots(3, 4)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

	for i, ax in enumerate(axes.flat):
		if i < 10:
			image = weights[:, i].reshape(img_shape)
			ax.set_xlabel('Weights: {0}'.format(i))
			ax.imshow(image, vmin = w_min, vmax = w_max, cmap = 'seismic')
		ax.set_xticks([])
		ax.set_yticks([])
	plt.suptitle(title)
	plt.show()

