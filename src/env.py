from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

def rk4_fn(f, x, t, tn, h, p):
  steps = np.ceil((tn - t) / h).astype(int)
  h = (tn - t) / steps

  for _ in range(steps):
    k1 = f(x, t, p)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h, p)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h, p)
    k4 = f(x + h * k3, t + h, p)
    x += (h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
    t += h
  return x

class Environment:
  def __init__(self):
    pass

  # reward function
  def reward_(s_, a_, ns_):
    raise NotImplementedError

  # possible actions
  def actions_(s_, n=None):
    raise NotImplementedError

  # transitioon probabilities
  def probability_(s_, a_):
    raise NotImplementedError

# POLICIES ####################################################################
class Policy:
  def __init__(self, sdim, adim):
    self.sdim = sdim
    self.adim = adim

  def choose_action(self, s):
    raise NotImplementedError

  def choose_action_discrete(self, s, n, amin, amax):
    assert self.adim == np.size(amin)
    assert np.size(amin) == np.size(amax)
    amin = np.array(amin).reshape(-1)
    amax = np.array(amax).reshape(-1)
    n = (np.array(n).reshape(-1).astype(int) if np.size(n) > 1 else
        np.repeat(int(n), self.adim).reshape(-1))
    a = self.choose_action(s)
    a = np.clip(a, amin, amax)
    a = np.round((n - 1) * (a - amin) / (amax - amin)).astype(int) / (n - 1)
    return a

class UnifRandPolicy(Policy):
  def __init__(self, sdim, amin, amax):
    super(UnifRandPolicy, self).__init__(sdim, np.size(amin))
    assert np.size(amin) == np.size(amax)
    self.amin = np.reshape(amin, (1, -1))
    self.amax = np.reshape(amax, (1, -1))

  def choose_action(self, s):
    s = np.array(s).reshape((-1, self.sdim))
    return (np.random.rand(s.shape[0], self.adim) * (self.amax - self.amin) +
        self.amin)

  def choose_action_discrete(self, s, n):
    s = np.array(s).reshape((-1, self.sdim))
    return super().choose_action_discrete(s, n, self.amin, self.amax)

class IndepNormRandPolicy(Policy):
  def __init__(self, sdim, mu, sig, bound=3.0):
    super(IndepNormRandPolicy, self).__init__(sdim, np.size(mu))
    assert np.size(mu) == np.size(sig)
    self.mu = np.reshape(mu, -1)
    self.sig = np.reshape(sig, -1)
    self.bound = bound

  def choose_action(self, s):
    s = np.array(s).reshape((-1, self.sdim))
    return (np.random.normal(loc=self.mu, scale=self.sig, size=(s.shape[0],
      self.adim)))

  def choose_action_discrete(self, s, n):
    s = np.array(s).reshape((-1, self.sdim))
    return super().choose_action_discrete(s, n, self.mu - self.bound *
        self.sig, self.mu + self.bound * self.sig)

class MultNormRandPolicy(Policy):
  def __init__(self, sdim, mu, S, bound=3.0):
    super(MultNormRandPolicy, self).__init__(sdim, np.size(mu))
    assert np.size(mu) == np.diag(S).size
    self.mu = np.reshape(mu, -1)
    self.S = S
    self.bound = bound

  def choose_action(self, s):
    s = np.array(s).reshape((-1, self.sdim))
    return np.random.multivariate_normal(self.mu, self.S, s.shape[0])

  def choose_action_discrete(self, s, n):
    s = np.array(s).reshape((-1, self.sdim))
    return super().choose_action_discrete(s, n, self.mu - self.bound *
        np.diag(self.S), self.mu + self.bound * np.diag(self.S))

class DiscretePolicy(Policy):
  def __init__(self, sdim, amin, amax, n):
    assert np.size(amin) == np.size(amax)
    assert np.size(n) == 1 or np.size(n) == np.size(amin)
    super(DiscretePolicy, self).__init__(sdim, np.size(amin))
    self.amin = np.array(amin).reshape(-1)
    self.amax = np.array(amax).reshape(-1)
    self.n = (np.array(n).reshape(-1).astype(int) if np.size(n) > 1 else
        np.repeat(int(n), self.adim).reshape(-1))

    alin = [np.linspace(self.amin[i], self.amax[i], self.n[i]) for i in
        range(self.adim)]
    self.aspc = np.hstack([aarr.reshape((-1, 1)) for aarr in
      np.meshgrid(*alin)])
    self.aspcdim = self.aspc.shape[0]

  def choose_action(self, s, idx=None):
    s = np.array(s).reshape((-1, self.sdim))
    idx = int(idx) if idx != None else (self.aspcdim // 2) + 1
    return np.repeat(self.aspc[idx:(idx + 1), 0:], np.shape(s)[0], axis=0)
###############################################################################


# Value Function ##############################################################
class ValueFunction:
  def __init__(self, sdim):
    self.sdim = sdim

  def value(self, s):
    raise NotImplementedError

class TabularValueFunction(ValueFunction):
  def __init__(self, smin, smax, n):
    sdim = np.size(smin)
    assert sdim == np.size(smax)
    assert np.size(n) == 1 or np.size(n) == sdim
    super(TabularValueFunction, self).__init__(sdim)
    n = (np.array(n).reshape(-1).astype(int) if np.size(n) > 1 else
        np.repeat(int(n), self.adim).reshape(-1))
    self.s_nb = np.prod(n)
    self.value_table = np.zeros(s_nb)

  def _sidx(self, s):
    s = np.array(s).reshape((-1, self.sdim))
    s_idx = np.round((self.n - 1) * (s - self.smin) / (self.smax -
      self.smin)).astype(int)
    return np.ravel_multi_index(s_idx.T, self.n)

  def value(self, s):
    sidx = self._sidx(s)
    return self.value_table[sidx]

  def set_value(self, s, v):
    sidx = self._sidx(s)
    self.value_table[sidx] = v
    return
###############################################################################


# Environments ################################################################
class FrozenLake(Environment):
  def __init__(self):
    self.s_min = np.array([[0.0, -1.0, 0.0, -1.0]])
    self.s_max = np.array([[1.0, 1.0,  1.0,  1.0]])
    self.a_min = np.array([0.0, 0.0])
    self.a_max = np.array([0.1, 0.1])

    self.gridn = 8
    self.holes = [[1, 1], [3, 5], [7, 1], [7, 2], [5, 5], [2, 6], [0, 4], [3,
      6]]
    self.goals = [[6, 2]]

    self.dist_a_mu = 0
    self.dist_a_S = 1e-7 * np.eye(len(self.a_min))
    self.dist_a_Sinv = np.linalg.inv(self.dist_a_S)
    self.dist_ns_mu = 0
    self.dist_ns_S = 1e-7 * np.eye(len(self.a_min))
    self.dist_ns_Sinv = np.linalg.inv(self.dist_ns_S)
    self.lam_dist_a_ns_weight = 1.0

    def double_int(x, t, p):
      a = p
      return np.hcat([x[:, 1:2], a[:, 0:1], x[:, 3:4], a[:, 1:2]])
    self.f = double_int

    self.h = 1e-2
    self.dt = 1e-1

    self.mu = np.array([0.0, 0.0])
    self.S = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
    self.n = np.array([1, 1, 1, 1, 5, 5])

    def reward(self, s, a, ns):
      hole_mask = self.is_in_discrete(self.holes)
      return -1.0 * hole_mask

    def is_in_discrete(self, points):
      pos_min = self.s_min[:, 0:2]
      pos_max = self.s_min[:, 0:2]
      pos = s[:, 0:2]
      n = np.array([[self.gridn, self.gridn]])
      posd = np.round(n * (pos - pos_min) / (pos_max - pos_min))
      mask = np.any(np.array([np.all(posd == hole, axis=1) for hole in
        self.holes]), axis=0).reshape((-1, 1))

    def is_terminal(self, s):
      hole_mask = self.is_in_discrete(self.holes)
      goal_mask =   self.is_in_discrete(self.goals)
      return np.logical_or(goal_mask, hole_mask)

    def next_state(self, s, a):
      h = 1e-1
      ns = rk4_fn(self.f, s, 0.0, self.dt, self.h, a)
      ns = np.clip(ns, self.smin, self.smax)
      return ns

    def sample_states(self, N):
      return (np.random.rand(N, self.s_min.size) * (self.s_max - self.s_min) +
          self.s_min)

class DiscreteFrozenLake(FrozenLake):
  def __init__(self, n):
    super(DiscreteFrozenLake, self).__init__()
    assert np.size(n) == 1 or np.size(n) == 4
    self.n = (np.array(n).reshape(-1).astype(int) if np.size(n) > 1 else
        np.repeat(int(n), 4).reshape(-1))

    def _s2sd(self, s):
      s = (np.round((self.n - 1) * (s - self.smin) / (self.smax - self.smin)) /
          (self.n - 1))

    def next_state(self, s, a):
      s = np.array(s).reshape((-1, self.sdim))
      sd = self._s2sd(s)
      ns = super(DiscreteFrozenLake, self).next_state(sd, a)
      nsd = self._s2sd(ns)
      return nsd
###############################################################################
