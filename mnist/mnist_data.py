import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import udf

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10 ## 0 - 9 共10 个数字

def load_data():
	data = input_data.read_data_sets("./data", one_hot = True)
	## 每个label 是一个10 个元素的vector, eg: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] 将 7 点亮了, 因此label 是 7

	print('data set brief:')
	print('train set: {0} {1}'.format(data.train.images.shape, data.train.labels.shape)) ## (55000, 784) (55000, 10)
	print('test set: {0} {1}'.format(data.test.images.shape, data.test.labels.shape))
	print('validation set: {0} {1}'.format(data.validation.images.shape, data.validation.labels.shape))

	## 为了方便之后的比较, 把hot vector 转换为一个数字
	data.train.cls = np.array([np.argmax(label) for label in data.train.labels])
	data.test.cls = np.array([np.argmax(label) for label in data.test.labels])

	return data

