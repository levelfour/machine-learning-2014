#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import cross_validation

class SimplePerceptron:
	def __init__(self, dim, eta=0):
		# wに初期値として単位ベクトルを設定する
		self.w = np.array([1 for i in range(dim)])
		self.eta = eta

	def fit(self, x, y):
		x = y * x # 学習データの反転
		if self.w.T.dot(x) >= 0:
			pass
		else:
			self.w = self.w + self.eta * x

	def predict(self, x):
		y = self.w.T.dot(x)
		return 1 if y >= 0 else -1

if __name__ == '__main__':
	digits = datasets.load_digits()
	data = None
	target = None
	for i in range(len(digits.data)):
		# プリセットの手書き数字から0か1のデータのみを読み込む
		if digits.target[i] == 0 or digits.target[i] == 1:
			if data == None or target == None:
				data = np.array([digits.data[i]])
				target = np.array([1 if digits.target[i] == 1 else -1])
			else:
				data = np.r_[data, [digits.data[i]]]
				# 1と-1にラベリングし直す
				target = np.r_[target, 1 if digits.target[i] == 1 else -1]

	train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)
	
	p = SimplePerceptron(dim = data[0].shape[0], eta = 0.3)
	for i in range(len(train_x)):
		# 学習データを逐次学習する
		p.fit(train_x[i], train_y[i])

	# テストデータに対して推定する
	pred_y = np.array([p.predict(x) for x in test_x])
	# 正答率を表示
	print metrics.accuracy_score(pred_y, test_y)
