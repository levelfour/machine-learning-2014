#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn import metrics

class MultiLayerPerceptron:
	def __init__(self, dim, n_mnodes, n_onodes=1, eta=1, beta=1):
		# expでオーバーフローが発生することがあるがひとまず無視する
		np.seterr(over='ignore')
		# 重みベクトルの初期値はランダムなベクトルとする
		self.w = np.random.uniform(-1., 1., (n_mnodes, dim))
		self.v = np.random.uniform(-1., 1., (n_mnodes,))
		self.n_mnodes = n_mnodes # 中間層のノード数
		self.n_onodes = n_onodes # 出力層のノード数
		self.eta = eta
		self.beta = beta

	# シグモイド関数
	def g(self, x):
		return 1.0/(1.0+np.exp(-self.beta*x))

	# シグモイド関数の1階導関数
	def g_(self, x):
		return self.beta*self.g(x)/(1-self.g(x))

	# 入力ベクトルから中間層の出力を得る
	def hidden_layer(self, x):
		return np.array([self.g(self.w[m].T.dot(x)) for m in range(self.n_mnodes)])

	def fit(self, x, t):
		z = self.predict(x)
		# 最急降下法で係数ベクトルを更新する
		self.w = np.array([
			self.w[m]-self.eta*2*(z-t)*self.v[m]*self.g_(x)*x 
			for m in range(self.n_mnodes)])
		self.v = self.v - self.eta*2*(z-t)*self.hidden_layer(x)
		
	def predict(self, x):
		y = self.v.T.dot(self.hidden_layer(x))
		return 1 if y >= 0 else -1

if __name__ == "__main__":
	digits = datasets.load_digits()
	data = None
	target = None
	for i in range(len(digits.data)):
		if digits.target[i] == 0 or digits.target[i] == 1:
			if data == None or target == None:
				data = np.array([digits.data[i]])
				target = np.array([1 if digits.target[i] == 1 else -1])
			else:
				data = np.r_[data, [digits.data[i]]]
				target = np.r_[target, 1 if digits.target[i] == 1 else -1]

	train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)

	p = MultiLayerPerceptron(dim = data[0].shape[0], eta = 0.3, n_mnodes = 10)
	for i in range(len(train_x)):
		p.fit(train_x[i], train_y[i])
	
	pred_y = np.array([p.predict(x) for x in test_x])
	print metrics.accuracy_score(pred_y, test_y)
