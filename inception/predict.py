import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from IPython.display import Image, display

# Functions and classes for loading and using the Inception model.
import inception
'''
It takes several weeks for a monster-computer to train the Inception model,
but we can just download the finished model from the internet and use it on a normal PC for classifying images.
Unfortunately, the Inception model appears to have problems recognizing people. This may be due to the training-set that was used.
'''

## Download the data for the Inception model if it doesn't already exist in the directory. It is 85 MB.
inception.maybe_download()

model = inception.Inception()

def classify(image_path):
	# Display the image.
	display(Image(image_path))

	# Use the Inception model to classify the image.
	pred = model.classify(image_path=image_path)

	# Print the scores and names for the top-10 predictions.
	model.print_scores(pred=pred, k=10, only_first_name=True)

image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)

