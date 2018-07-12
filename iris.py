import os
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

FEATURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#载入数据
train = pd.read_csv('./data/iris/iris_training.csv', names=FEATURES, header=0)
test = pd.read_csv('./data/iris/iris_test.csv', names=FEATURES, header=0)
train_x, train_y = train, train.pop('Species')
test_x, test_y = test, test.pop('Species')

#设定特征值的名称
feature_columns = []
for key in train_x.keys():
	feature_columns.append(tf.feature_column.numeric_column(key=key))
# 结果: _NumericColumn(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)

#选定评估器：深层神经网络分类器  
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 10], n_classes=3)

#针对训练的喂食函数
def train_input_fn(features, labels, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
	dataset = dataset.shuffle(1000).repeat().batch(batch_size) #每次随机调整数据顺序
	return dataset

#开始训练模型！
batch_size = 100
classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, batch_size), steps=1000)

#针对测试的喂食函数
def eval_input_fn(features, labels, batch_size):
	features = dict(features)
	if labels is None:
		inputs = features # No labels, use only features
	else:
		inputs = (features, labels)
	dataset = tf.data.Dataset.from_tensor_slices(inputs)
	dataset = dataset.batch(batch_size)
	return dataset

#评估我们训练出来的模型质量
eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))
print(eval_result)

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
	'SepalLength': [5.1, 5.9, 6.9],
	'SepalWidth': [3.3, 3.0, 3.1],
	'PetalLength': [1.7, 4.2, 5.4],
	'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=batch_size))

for pred_dict, expec in zip(predictions, expected):
	'''
	print(pred_dict) result
	{
		'logits': array([-16.18431473, 5.61845016, 13.43036652], dtype=float32),
		'probabilities': array([1.37509145e-13, 4.04717575e-04, 9.99595344e-01], dtype=float32),
		'class_ids': array([2]),
		'classes': array([b'2'], dtype=object)
	}
	'''
	class_id = pred_dict['class_ids'][0]
	probability = pred_dict['probabilities'][class_id]
	print('{0} and predicted to be {1} with prob {2}'.format(expec, SPECIES[class_id], probability))

