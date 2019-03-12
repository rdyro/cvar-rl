from __future__ import division
from __future__ import print_function
from tf_util import *
import random

# Models ######################################################################
class Model:
  def __init__(self):
    pass

  def train(self, x, y):
    raise NotImplementedError

  def predict(self, x):
    raise NotImplementedError

class LSModel(Model):
  def __init__(self, in_nb, out_nb):
    self.in_nb = in_nb
    self.out_nb = out_nb

  def _features(self, x):
    raise NotImplementedError

class PolyLSModel(LSModel):
  def __init__(self, in_nb, out_nb, order, mix=False):
    super().__init__(in_nb, out_nb)
    self.order = (np.array(order).reshape(-1) if np.size(order) > 1 else
        np.repeat(order, in_nb).reshape(-1))
    self.mix = mix
    x_example = self._features(np.repeat(0, in_nb).reshape((1, -1)))
    self.th = np.zeros((x_example.size, out_nb))

  def _features(self, x):
    x = np.array(x)
    assert len(x.shape) == 2
    feat_cols = []
    if self.mix == True:
      mgrid = np.meshgrid(*[np.arange(el + 1) for el in self.order])
      mix_pow = np.hstack([el.reshape((-1, 1)) for el in mgrid])
      feat_cols += [np.prod(np.power(x[:, :], el), axis=1).reshape((-1, 1)) for
          el in mix_pow]
    else:
      feat_cols.append(np.ones(x.shape[0]).reshape((-1, 1)))
      feat_cols += [np.power(x[:, i:i+1], np.arange(self.order[i]).reshape((1,
        -1)) + 1) for i in range(x.shape[1])]
      
    return np.hstack(feat_cols)

  def train(self, x, y):
    assert np.shape(y)[1] == self.out_nb
    assert np.shape(x)[1] == self.in_nb
    A = self._features(x)
    (self.th, _, _, _) = np.linalg.lstsq(A, y, rcond=-1)

  def predict(self, x):
    A = self._features(x)
    return np.dot(A, self.th)

class NNModel:
  def __init__(self, in_nb, out_nb, layerN, **kwargs):
    self.params = dict({"h": 1e-2}, **kwargs)

    self.in_nb = in_nb
    self.out_nb = out_nb
    self.scope = random_scope()
    self.x_ = tf.placeholder(shape=(None, in_nb), dtype=tf.float32)
    self.y_ = tf.placeholder(shape=(None, out_nb), dtype=tf.float32)
    self.kp_ = tf.placeholder(dtype=tf.float32)
    self.pred_ = pred_op(self.x_, layerN, self.scope, out_nb, self.kp_)
    self.loss_ = loss_op(self.y_, self.pred_)

    self.train_ = optimizer_op(self.loss_, self.params["h"])

    self.sess = None

  def set_session(self, sess):
    self.sess = sess

  def train(self, x, y, times=-1, batch_frac=0.01):
    assert self.sess != None
    train_till_convergence_or_for(self.sess, self.loss_, self.train_,
        [self.x_, self.y_, self.kp_], [x, y, 1.0], [True, True, False],
        times=times, batch_frac=batch_frac)

  def predict(self, x):
    return self.sess.run(self.pred_, feed_dict={self.x_: x, self.kp_: 1.0})

  def loss(self, x, y, batch_frac=1.0):
    idx = batch_idx(int(np.ceil(N * batch_frac)), N)
    return self.sess.run(self.loss_, feed_dict={self.x_: x[:, idx], self.y_:
      y[:, idx], kp_: self.kp_})
