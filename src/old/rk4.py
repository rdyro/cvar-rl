from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl
import time

def rk4_fn_(f_, x_, t_, h_, p_):
  k1_ = f_(x_, t_, p_)
  k2_ = f_(x_ + 0.5 * h_ * k1_, t_ + 0.5 * h_, p_)
  k3_ = f_(x_ + 0.5 * h_ * k2_, t_ + 0.5 * h_, p_)
  k4_ = f_(x_ + h_ * k3_, t_ + h_, p_)
  return x_ + (h_ / 6.0 * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_))

def f_(x_, t_, p_):
  col1 = tf.slice(x_, [0, 0], [-1, 1])
  col2 = tf.slice(x_, [0, 1], [-1, 1])
  return tf.concat([col2, -col1], axis=1)

def rk4_fn(f, x, t, h, p):
  k1 = f(x, t, p)
  k2 = f(x + 0.5 * h * k1, t + 0.5 * h, p)
  k3 = f(x + 0.5 * h * k2, t + 0.5 * h, p)
  k4 = f(x + h * k3, t + h, p)
  return x + (h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

def f(x, t, p):
  return np.hstack([x[:, 1:2], -x[:, 0:1]])

x_ = tf.placeholder(shape=(None, 2), dtype=tf.float32)

N = int(1e3)
t = np.linspace(0, 10, N)
h = np.mean(np.diff(t))

rk4_op = rk4_fn(f_, x_, 0.0, h, None)

M = int(5e5)
x = np.random.rand(M, 2) * 5.0
X = np.zeros((M, 2, N))
X[:, :, 0] = x[:, :]

sess = tf.Session()
t1 = time.time()
for i in range(N-1):
  x = sess.run(rk4_op, feed_dict={x_: x})
  X[:, :, i + 1] = x[:, :]
  print(i)
t2 = time.time()
sess.close()

print("ELAPSED TIME IS %e s" % (t2 - t1))

t1 = time.time()
for i in range(N-1):
  x = rk4_fn(f, x, 0.0, h, None)
  X[:, :, i + 1] = x[:, :]
  print(i)
t2 = time.time()
sess.close()

print("ELAPSED TIME IS %e s" % (t2 - t1))


for i in range(100):
  r = np.random.randint(M)
  pl.plot(t, X[r, 0, :].reshape(-1))
pl.show()
