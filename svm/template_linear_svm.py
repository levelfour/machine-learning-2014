#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import cross_validation

class LinearSVM:
	def __init__(self):
		pass

	def fit(self, x, y):
		pass

	def predict(self, x):
		pass

if __name__ == '__main__':
	digits = datasets.load_digits()
	data = []
	target = []
	for i in range(len(digits.data)):
		# プリセットの手書き数字から0か1のデータのみを読み込む
		if digits.target[i] == 0 or digits.target[i] == 1:
			if data == [] or target == []:
				data = np.array([digits.data[i]])
				target = np.array([1 if digits.target[i] == 1 else -1])
			else:
				data = np.r_[data, [digits.data[i]]]
				# 1と-1にラベリングし直す
				target = np.r_[target, 1 if digits.target[i] == 1 else -1]

	train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)

	clf = LinearSVM()
	clf.fit(train_x, train_y)
	pred_y = np.array([clf.predict(x) for x in test_x])
	print metrics.accuracy_score(pred_y, test_y)
