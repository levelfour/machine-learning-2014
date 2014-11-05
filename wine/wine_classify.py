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

# 最小二乗法による学習
w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

v = np.array([
	14,
	0,
	0,
	0,
	100,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0])
# w.T.dot(v) = [-0.15889022  0.63867965  0.44747914]
# np.argmax(w.T.dot(v)) = 1
# したがってクラス2
print np.argmax(w.T.dot(v))
