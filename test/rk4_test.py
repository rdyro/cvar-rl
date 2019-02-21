from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from util import *

import matplotlib.pyplot as pl

# RK4 test ####################################################################
N = 50
K = 20
x = np.random.rand(N, 2, K)
a = np.ones((N, 1, 1))

def f(x, t, p):
  return np.hstack([x[:, 1:2], p - x[:, 0:1]])

M = int(1e2)
X = np.zeros((N, 2, K, M))
X[:, :, :, 0] = x[:, :, :]
t = np.linspace(0, 10, M)
h = 1e-2

for i in range(1, M):
  x = rk4_fn(f, x, t[i-1], t[i], h, a)
  X[:, :, :, i] = x[:, :, :]

pl.figure(10)
pl.clf()
for i in range(N):
  r = np.random.randint(K)
  pl.plot(t, X[i, 0, r, :].reshape(-1))

pl.show()
