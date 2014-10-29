#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

c1 = []
c2 = []
m1 = (-1, 2)
m2 = (1, -2)

for i in range(10):
	c1.append([m1[0]+np.random.normal()/1.5, m1[1]+np.random.normal()/1.5])
	c2.append([m2[0]+np.random.normal()/1.5, m2[1]+np.random.normal()/1.5])

plt.scatter(map(lambda x: x[0], c1), map(lambda x: x[1], c1), c='#008800')
plt.scatter(map(lambda x: x[0], c2), map(lambda x: x[1], c2), c='#0000aa')

w = (0.5, 0)
xx = np.arange(-6, 6, 0.1)
yy = []
for x in xx:
	yy.append(w[0]*x+w[1])
plt.plot(xx, yy)

plt.quiver(0, 0, w[0], -1, angles='xy', scale_units='xy', scale=1)

lim = (-6, 4)
plt.xlim(lim)
plt.ylim(lim)
plt.grid()
plt.draw()
plt.show()
