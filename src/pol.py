from __future__ import division
from __future__ import print_function

from util import *
from tf_util import *

# POLICIES ####################################################################
# general
class Policy:
  def __init__(self, sdim, adim):
    self.sdim = sdim
    self.adim = adim
    self.environment = None
    self.value_function = None

  def choose_action(self, s):
    raise NotImplementedError

  def choose_action_discrete(self, s, n, amin, amax):
    s = make3D(s, self.sdim)
    assert self.adim == np.size(amin)
    assert np.size(amin) == np.size(amax)
    amin = make3D(amin, self.adim)
    amax = make3D(amax, self.adim)
    n = (make3D(np.array(n), self.adim).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), self.adim), self.adim))
    a = self.choose_action(s)
    a = np.clip(a, amin, amax)
    a = np.round((n - 1) * (a - amin) / (amax - amin)).astype(int) / (n - 1)
    return a

  def set_environment(self, environment):
    self.environment = environment

  def set_value_function(self, value_function):
    self.value_function = value_function

class UnifRandPolicy(Policy):
  def __init__(self, sdim, amin, amax):
    super().__init__(sdim, np.size(amin))
    assert np.size(amin) == np.size(amax)
    self.amin = make3D(amin, self.adim)
    self.amax = make3D(amax, self.adim)

  def choose_action(self, s):
    s = make3D(s, self.sdim)
    return (np.random.rand(s.shape[0], self.adim, s.shape[2]) * (self.amax -
      self.amin) + self.amin)

  def choose_action_discrete(self, s, n):
    return super().choose_action_discrete(s, n, self.amin, self.amax)

class IndepNormRandPolicy(Policy):
  def __init__(self, sdim, mu, sig, bound=3.0):
    super().__init__(sdim, np.size(mu))
    assert np.size(mu) == np.size(sig)
    self.mu = np.array(mu).reshape(-1)
    self.sig = np.array(sig).reshape(-1)
    self.bound = bound

  def choose_action(self, s):
    s = make3D(s, self.sdim)
    return np.dstack([np.random.normal(loc=self.mu, scale=self.sig,
      size=(s.shape[0], self.adim)) for i in range(s.shape[2])])

  def choose_action_discrete(self, s, n):
    return super().choose_action_discrete(s, n, self.mu - self.bound *
        self.sig, self.mu + self.bound * self.sig)

class MultNormRandPolicy(Policy):
  def __init__(self, sdim, mu, S, bound=3.0):
    super().__init__(sdim, np.size(mu))
    assert np.size(mu) == np.diag(S).size
    self.mu = np.array(mu).reshape(-1)
    self.S = S
    self.bound = bound

  def choose_action(self, s):
    s = make3D(s, self.sdim)
    return (np.random.multivariate_normal(self.mu, self.S, (s.shape[0],
      s.shape[2]))).transpose((0, 2, 1))

  def choose_action_discrete(self, s, n):
    return super().choose_action_discrete(s, n, self.mu - self.bound *
        np.diag(self.S), self.mu + self.bound * np.diag(self.S))

# tabular
class DiscretePolicy(Policy):
  def __init__(self, sdim, amin, amax, n):
    super().__init__(sdim, np.size(amin))
    assert self.adim == np.size(amax)
    assert np.size(n) == 1 or np.size(n) == np.size(amin)
    self.amin = make3D(amin, self.adim)
    self.amax = make3D(amax, self.adim)
    self.n = (make3D(n, self.adim).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), self.adim), self.adim))

    alin = [np.linspace(self.amin.reshape(-1)[i], self.amax.reshape(-1)[i],
      self.n.reshape(-1)[i]) if self.n.reshape(-1)[i] > 1 else
      np.array([(self.amin.reshape(-1)[i] + self.amax.reshape(-1)[i]) / 2.0])
      for i in range(self.adim)]
    self.aspc = make3D(np.hstack([aarr.reshape((-1, 1)) for aarr in
      np.meshgrid(*alin)]), self.adim)
    self.aspcdim = self.aspc.shape[0]

  def choose_action(self, s, idx=None):
    s = make3D(s, self.sdim)
    idx = int(idx) if idx != None else (self.aspcdim // 2) + 1
    idx = np.array(idx) if np.size(idx) > 1 else np.repeat(idx, s.shape[0])
    return self.aspc[idx, 0:]

  def choose_action_discrete(self, s, n):
    return super().choose_action_discrete(s, n, self.amin, self.amax)

class OptimalDiscretePolicy(DiscretePolicy):
  def __init__(self, sdim, amin, amax, n):
    super().__init__(sdim, amin, amax, n)

  def choose_action(self, s):
    V = np.array([self.value_function.qvalue(self.environment, s,
      self.aspc[i:i+1, :, :]).reshape(-1) for i in range(self.aspcdim)])
    a_argmax = np.argmax(V, axis=0)
    return self.aspc[a_argmax, 0:]

  def choose_action_discrete(self, s, n):
    return super().choose_action_discrete(s, n, self.amin, self.amax)

# policy gradient
class GaussianPolicy(Policy):
  def __init__(self, sdim, adim, amin, amax, **kwargs):
    self.params = dict({"h": 1e-2, "layerN": np.repeat(16, 2), "scope": None},
        **kwargs)
    super().__init__(sdim, adim)
    self.amin = amin
    self.amax = amax
    assert np.size(self.amin) == self.adim
    assert np.size(self.amax) == self.adim
    self.amin = make3D(amin, self.adim)
    self.amax = make3D(amax, self.adim)

    self.scope = (random_scope() if self.params["scope"] is None else
        self.params["scope"])

    # placeholders
    self.adv_ = tf.placeholder(tf.float32, shape=(None,))
    self.s_ = tf.placeholder(tf.float32, shape=(None, self.sdim))
    self.a_ = tf.placeholder(tf.float32, shape=(None, self.adim))
    # variables
    self.a_mu_ = pred_op(self.s_, self.params["layerN"], self.scope, self.adim)
    with tf.variable_scope(self.scope):
      self.a_logstd_ = tf.get_variable("log_std", dtype=tf.float32, shape=(1,
        self.adim))
    # operations
    self.a_sample_ = (self.a_mu_ + tf.exp(self.a_logstd_) *
        tf.random_normal(tf.shape(self.a_mu_)))
    #self.logprob_ = tf.log(self._prob_(self.a_sample_, self.a_mu_,
    #  tf.exp(self.a_logstd_)))
    self.logprob_ = self._logprob_(self.a_, self.a_mu_, tf.exp(self.a_logstd_))
    self.loss_ = -tf.reduce_mean(self.logprob_ * self.adv_)
    self.train_ = (tf.train.AdamOptimizer(
      learning_rate=self.params["h"]).minimize(self.loss_))

    self.sess = None

  def _prob_(self, x_, mu_, std_):
    return ((2 * np.pi * tf.reduce_prod(std_))**(-0.5) * 
        tf.exp(-0.5 * tf.reduce_mean(x_**2 / std_, axis=1)))

  def _logprob_(self, x_, mu_, std_):
    M = tf.contrib.distributions.MultivariateNormalDiag(loc=mu_,
        scale_diag=std_)
    return M.log_prob(x_)

  def set_session(self, sess):
    self.sess = sess
    sess.run(tf.assign(self.a_logstd_,
      np.repeat(-2.0, self.adim).reshape((1, -1))))

  def train(self, s, a, adv, times=-1, batch_frac=0.01):
    assert self.sess != None

    (s, a, adv) = match_03(s, a, adv)
    (s, _) = unstack2D(s)
    (a, _) = unstack2D(a)
    (adv, _) = unstack2D(adv)
    adv = adv.reshape(-1)
    train_till_convergence_or_for(self.sess, self.loss_, self.train_,
        [self.s_, self.a_, self.adv_], [s, a, adv], times=times)
    print("STD is ", self.sess.run(tf.exp(self.a_logstd_)))

  def choose_action(self, s):
    assert self.sess != None
    (s, layer_nb) = unstack2D(s)
    return stack2D(self.sess.run(self.a_sample_, feed_dict={self.s_: s}),
        layer_nb)

  def choose_action_discrete(self, s, n, amin, amax):
    raise NotImplementedError

class SoftmaxPolicy(Policy):
  def __init__(self, sdim, adim, amin, amax, an, **kwargs):
    self.params = dict({"h": 1e-2, "layerN": np.repeat(32, 3), "scope": None},
        **kwargs)
    super().__init__(sdim, adim)
    self.amin = amin
    self.amax = amax
    assert np.size(self.amin) == self.adim
    assert np.size(self.amax) == self.adim
    self.amin = make3D(amin, self.adim)
    self.amax = make3D(amax, self.adim)
    self.an = (make3D(an, self.adim).astype(int) if np.size(an) > 1 else
        make3D(np.repeat(int(an), self.adim), self.adim))
    self.adim_lin = np.prod(self.an)

    self.scope = (random_scope() if self.params["scope"] is None else
        self.params["scope"])

    # placeholders
    self.adv_ = tf.placeholder(tf.float32, shape=(None,))
    self.s_ = tf.placeholder(tf.float32, shape=(None, self.sdim))
    self.a_lin_ = tf.placeholder(tf.int32, shape=(None,))
    # variables
    self.a_lin_logit_ = pred_op(self.s_, self.params["layerN"], self.scope,
        self.adim_lin)
    # operations
    self.a_lin_sample_ = tf.squeeze(tf.multinomial(self.a_lin_logit_, 1),
        axis=1)
    self.logprob_ = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.a_lin_, logits=self.a_lin_logit_)
    self.loss_ = -tf.reduce_mean(self.logprob_ * self.adv_)
    self.train_ = (tf.train.AdamOptimizer(
      learning_rate=self.params["h"]).minimize(self.loss_))

    self.sess = None

  def _prob_(self, x_, mu_, std_):
    return ((2 * np.pi * tf.reduce_prod(std_))**(-0.5) * 
        tf.exp(-0.5 * np.reduce_mean(x_**2 / std_, axis=1)))

  def set_session(self, sess):
    self.sess = sess

  def train(self, s, a, adv, times=-1, batch_frac=0.01):
    assert self.sess != None

    (s, a, adv) = match_03(s, a, adv)
    (s, _) = unstack2D(s)
    (a, _) = unstack2D(a)
    (adv, _) = unstack2D(adv)

    a_lin = np.ravel_multi_index(np.round(a).astype(int).T,
        self.an.reshape(-1))
    adv = adv.reshape(-1)
    train_till_convergence_or_for(self.sess, self.loss_, self.train_,
        [self.s_, self.a_lin_, self.adv_], [s, a_lin, adv], times=times)

  def choose_action(self, s):
    assert self.sess != None
    (s, layer_nb) = unstack2D(s)
    a_lin = self.sess.run(self.a_lin_sample_, feed_dict={self.s_: s})
    a = np.vstack(np.unravel_index(a_lin, self.an.reshape(-1))).T
    return stack2D(a, layer_nb)

  def choose_action_discrete(self, s, n, amin, amax):
    raise NotImplementedError
###############################################################################
