from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from util import *
from tf_util import *
import mod

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import time

# RK4 test ####################################################################
N = int(1e3)
x = np.linspace(-5, 10, N)
sig = 1e-1
mu = 0.0
y = np.sin(x) + sig * np.random.randn(N) + mu

pl.plot(x, y)

x_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)
y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)
kp_ = tf.placeholder(dtype=tf.float32)

model = mod.NNModel(1, 1, np.repeat(32, 4))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.set_session(sess)
t1 = time.time()
model.train(x.reshape((-1, 1)), y.reshape((-1, 1)))
t2 = time.time()
print(t2 - t1)
pl.plot(x, model.predict(x.reshape((-1, 1))).reshape(-1))

model2 = mod.PolyLSModel(1, 1, 11)
t1 = time.time()
model2.train(x.reshape((-1, 1)), y.reshape((-1, 1)))
t2 = time.time()
print(t2 - t1)
pl.plot(x, model2.predict(x.reshape((-1, 1))).reshape(-1))



X, Y = np.meshgrid(*[np.linspace(-2, 2, 100) for i in range(2)])
Z = np.cos(X + 2.0 * Y)

model_nn = mod.NNModel(2, 1, np.repeat(32, 4))
model_nn.set_session(sess)
sess.run(tf.global_variables_initializer())
model_nn.train(np.hstack([X.reshape((-1, 1)), Y.reshape((-1, 1))]),
    Z.reshape((-1, 1)))

model_pl = mod.PolyLSModel(2, 1, 5, mix=True)
model_pl.train(np.hstack([X.reshape((-1, 1)), Y.reshape((-1, 1))]),
    Z.reshape((-1, 1)))

Znn = model_nn.predict(np.hstack([X.reshape((-1, 1)), Y.reshape((-1,
  1))])).reshape((100, 100))

Zpl = model_pl.predict(np.hstack([X.reshape((-1, 1)), Y.reshape((-1,
  1))])).reshape((100, 100))


fig = pl.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(X, Y, Z)

fig = pl.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(X, Y, Znn)

fig = pl.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(X, Y, Zpl)

pl.show()


sess.close()

"""
windowN = 50
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  loss_history = np.zeros(windowN)
  i = 0
  t1 = time.time()
  while i < windowN or np.mean(loss_history[0:(windowN // 2)] -
      loss_history[(windowN // 2):]) >= 0.0:
    idx = np.random.randint(N, size=(N // 100))
    feed_dict = {x_: x[idx].reshape((-1, 1)), y_: y[idx].reshape((-1, 1)), kp_:
        1.0}
    (loss, _, gnorm) = sess.run([loss_, train_, gnorm_], feed_dict=feed_dict)
    loss_history[0:-1] = loss_history[1:]
    loss_history[-1] = loss
    print("loss = %f, gnorm = %f" % (loss, gnorm))

    i += 1
  t2 = time.time()
  print("Time elapsed %e s" % (t2 - t1))
  pl.plot(x, sess.run(yp_, feed_dict={x_: x.reshape((-1, 1)), kp_:
    1.0}).reshape(-1))
pl.show()
"""
