#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import cross_validation

class LinearSVM:
	def __init__(self, dim):
		self.dim = dim
		self.w = np.zeros((self.dim,))
		self.a = np.zeros((self.dim,))
		self.bias = 0
		self.margin = -1

	def fit(self, X, Y):
		def reload_w():
			self.w = np.zeros((self.dim,))
			for i in range(self.dim):
				self.w = self.w + self.a[i]*Y[i]*X[i]

		def check_kkt():
			def _check_kkt(a):
				for x, y in zip(X, Y):
					if not(a==0 and y*(np.dot(self.w,x)+self.bias)>1 or \
							a>0 and y*(np.dot(self.w,x)+self.bias)==1):
						return True
			illegals = []
			reload_w()
			for i, a in enumerate(self.a):
				if _check_kkt(a):
					illegals.append(i)
			return illegals 

		illegals = check_kkt()
		while len(illegals) > 1:
			print self.a
			i1 = illegals[0]
			i2 = illegals[1]
			L, H = 0, 0
			if Y[i1]==Y[i2]:
				L = max(0, self.a[i1]+self.a[i2])
				H = max(0, self.a[i1]+self.a[i2])
			else:
				L = max(0, self.a[i2]-self.a[i1])
				H = min(0, self.a[i1]-self.a[i2])
			unclipped_a2 = self.a[i2] \
					+ Y[i2]*(Y[i1]*(np.dot(self.w,X[i1])+self.bias)-Y[i2]*(np.dot(self.w,X[i2])+self.bias)) \
					/ (np.dot(X[i1],X[i1])-2*np.dot(X[i1],X[i2])+np.dot(X[i2],X[i2]))
			old_a2 = self.a[i2]
			if unclipped_a2 < L:
				self.a[i2] = L
			elif unclipped_a2 > H:
				self.a[i2] = H
			else:
				self.a[i2] = unclipped_a2
			self.a[i1] = self.a[i1] + Y[i1]*Y[i2]*(old_a2-self.a[i2])
			illegals = check_kkt()

	def predict(self, x):
		if np.dot(self.w, x) + self.bias >= 0:
			return 1
		else:
			return -1

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

	clf = LinearSVM(dim=train_x.shape[1])
	clf.fit(train_x, train_y)
	pred_y = np.array([clf.predict(x) for x in test_x])
	print metrics.accuracy_score(pred_y, test_y)
