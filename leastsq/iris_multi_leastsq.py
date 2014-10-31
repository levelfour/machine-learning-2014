#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn import metrics

iris = datasets.load_iris()
data = iris.data
target = np.array([
		[1 if t == 0 else 0 for t in iris.target],
		[1 if t == 1 else 0 for t in iris.target],
		[1 if t == 2 else 0 for t in iris.target]]).T

train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)

# 最小二乗法で学習
w = np.linalg.inv(train_x.T.dot(train_x)).dot(train_x.T).dot(train_y)

# 最小二乗法で推定
pred_y = np.array([np.argmax(w.T.dot(d)) for d in test_x])
true_y = np.array([np.argmax(d) for d in test_y])

# テストデータに対する正答率
print metrics.accuracy_score(true_y, pred_y)
