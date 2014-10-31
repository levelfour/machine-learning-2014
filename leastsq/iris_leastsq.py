#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn import metrics

iris = datasets.load_iris()
data = iris.data[0:100]
target = [1 if t == 1 else -1 for t in iris.target[0:100]]

train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)

# 最小二乗法で学習
w = np.linalg.inv(train_x.T.dot(train_x)).dot(train_x.T).dot(train_y)

# 最小二乗法で推定
pred_y = np.array([1 if w.dot(x) > 0 else -1 for x in test_x])

# テストデータに対する正答率
print metrics.accuracy_score(test_y, pred_y)
