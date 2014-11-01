#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import cross_validation
from sklearn import metrics

with open('wine.data', 'r') as f:
	data = np.loadtxt(f, delimiter=',')

# データの読み込み
X = data[:,1:]
# ラベルの読み込み
Y = np.array([
	[1 if t == i+1 else 0 for t in data[:,0]]
	for i in range(int(data[:,0].max()))
]).T

train_x, test_x, train_y, test_y = cross_validation.train_test_split(X, Y, test_size=0.2)

# 最小二乗法による学習
w = np.linalg.inv(train_x.T.dot(train_x)).dot(train_x.T).dot(train_y)

# 最小二乗法による推定
pred_y = np.array([np.argmax(w.T.dot(d)) for d in test_x])
true_y = np.array([np.argmax(d) for d in test_y])

# テストデータに対する正答率
print metrics.accuracy_score(true_y, pred_y)
