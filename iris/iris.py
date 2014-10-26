#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
# 先頭から100個のデータ(setosaとversicolorを抽出)
# 特徴は0番目(sepal length)と2列目(petal length)を使用
data = iris.data[0:100][:,::2]
target= iris.target[0:100]

from sklearn.svm import LinearSVC
from sklearn import cross_validation

# 学習データとテストデータを4:1に分割
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)
clf = LinearSVC() # 線形SVM
clf.fit(train_x, train_y) # 学習
pred = clf.predict(test_x) # テストデータの識別
print list(pred == test_y).count(True) / float(len(test_y))

h = .02
x_min, x_max = data[:,0].min()-1, data[:,0].max()+1
y_min, y_max = data[:,1].min()-1, data[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		             np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print xx.shape
print xx
# 等高線(contour)プロット
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# データの0列目(sepal length)と2列目(petal length)をプロット
# enumerate関数はループを回すときにイテレータにインデックスをセットする
for (i, d) in enumerate(data):
	if target[i] == 0: # target == setosa
		color = '#008800'
	elif target[i] == 1: # target == versicolor
		color = '#ff0000'
	plt.scatter(d[0], d[1], c=color)
plt.xlabel("sepal length[cm]")
plt.ylabel("petal length[cm]")
plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
plt.show()
