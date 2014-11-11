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
		np.seterr(divide='raise')
		# 重みベクトルの初期値はランダムなベクトルとする
		self.w = np.random.uniform(-1., 1., (n_mnodes, dim))
		self.v = np.random.uniform(-1., 1., (n_onodes, n_mnodes))
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
		# 統一的に扱うためにラベルがスカラーの場合もベクトルで扱う
		if not isinstance(t, np.ndarray):
			t = np.array([t])
		# 最急降下法で係数ベクトルを更新する
		# 計算式は頑張って読んでくださいm(_ _)m
		self.w = np.array([
			self.w[m]-self.eta*2*np.dot(z-t,self.v.T[m])*self.g_(np.dot(self.w[m],x))*x 
			for m in range(self.n_mnodes)])
		self.v = np.array([
			self.v[k]-self.eta*2*(z[k]-t[k])*self.hidden_layer(x) 
			for k in range(self.n_onodes)])
		
	def predict(self, x):
		yy = self.v.dot(self.hidden_layer(x))
		return np.array([1 if y >= 0 else -1 for y in yy])

if __name__ == "__main__":
	# 数字パターンデータを用意
	digits = datasets.load_digits()
	data = digits.data
	# ラベルは10次元ベクトルで、正しいラベルのインデックスのみ1、他の要素は-1
	# 例えば1なら[-1,1,-1,-1,...]
	target = np.array([[-1 for i in range(10)] for j in range(len(data))])
	for i in range(len(data)):
		target[i][digits.target[i]] = 1

	# 学習データとテストデータに分割
	train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)

	p = MultiLayerPerceptron(dim=data[0].shape[0], eta=0.1, beta=0.01, n_mnodes=10, n_onodes=10)
	for i in range(len(train_x)):
		p.fit(train_x[i], train_y[i])
	
	pred_y = np.array([p.predict(x) for x in test_x])
	print metrics.accuracy_score(pred_y, test_y)
