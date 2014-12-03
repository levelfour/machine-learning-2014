#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import cross_validation

cmax = 1000	# 最急降下法の試行回数

class LinearSVM:
	def __init__(self, dim, eta, C=0):
		self.dim = dim	# 学習データの次元数
		self.C = C		# コストパラメータ
		self.eta = eta	# 最急降下法の学習率
		self.a = None	# 未定乗数ベクトル
		self.w = np.zeros((dim,))	# 識別関数

	def fit(self, X, Y):
		data_n = X.shape[0]	# 学習データ数
		# 未定乗数ベクトルαの初期値は0ベクトルとする
		self.a = np.zeros((data_n,))
		# ones = (1,1,...,1)というベクトルを便宜上用意する
		ones = np.array([1 for _ in range(data_n)])
		H = np.array([[Y[i]*Y[j]*np.dot(X[i],X[j]) \
				for i in range(data_n)] for j in range(data_n)])
		# αについてのLagrange関数
		def L(av): return np.dot(self.a,ones)-self.a.T.dot(H.dot(self.a))/2.
		# Lの導関数
		def dL(av): return ones - np.dot(H,av)

		# 勾配法でLを最大化→αを求める
		for _ in range(cmax):
			a =  self.a + self.eta*dL(self.a)
			if self.C == 0 or np.product(np.array(map(lambda x: 0 <= x and x <= self.C, a))) != 0:
				self.a = a
		
		# 最適な線形識別関数を求める
		self.w = np.zeros((self.dim,))
		for i in range(data_n):
			self.w = self.w + self.a[i]*Y[i]*X[i]

	def predict(self, x):
		if np.dot(self.w, x) >= 0:
			return 1
		else:
			return -1

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

	clf = LinearSVM(dim=train_x.shape[1], C=10, eta=1e-15)
	clf.fit(train_x, train_y)
	pred_y = np.array([clf.predict(x) for x in test_x])
	print metrics.accuracy_score(pred_y, test_y)
