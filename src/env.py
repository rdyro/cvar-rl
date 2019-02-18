from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


# utility functions ###########################################################
def normal_dist(S, n=5, sig_span=3.0, **kwargs):
  Sinv = kwargs["Sinv"] if "Sinv" in kwargs else np.linalg.inv(S)

  dim = S.shape[0]

  n = np.array(n)
  n = np.repeat(n, dim) if n.size == 1 else n
  sig = np.sqrt(np.diag(S))
  rngs = [np.linspace(-sig_span * sig[i], sig_span * sig[i], n[i]) if n[i] > 1
      else np.array([0.0]) for i in range(dim)]
  dists = np.meshgrid(*rngs)
  dists = [dist.reshape((-1, 1)) for dist in dists]
  d = np.hstack(dists)
  p = (2 * np.pi * np.linalg.det(S))**(-0.5) * np.exp(-0.5 * np.sum(d *
    np.dot(Sinv, d.T).T, axis=1)).reshape(-1)
  p /= np.sum(p)
  return (d, p)

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

def make3D(x, xdim):
  return (np.array(x).reshape((-1, xdim, 1)) if len(np.shape(x)) <= 2 else
      np.array(x))
###############################################################################

# POLICIES ####################################################################
class Policy:
  def __init__(self, sdim, adim):
    self.sdim = sdim
    self.adim = adim

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

class DiscretePolicy(Policy):
  def __init__(self, sdim, amin, amax, n):
    assert np.size(amin) == np.size(amax)
    assert np.size(n) == 1 or np.size(n) == np.size(amin)
    super().__init__(sdim, np.size(amin))
    self.amin = make3D(amin, self.adim)
    self.amax = make3D(amax, self.adim)
    self.n = (make3D(n, self.adim).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), self.adim), self.adim))

    alin = [np.linspace(self.amin.reshape(-1)[i], self.amax.reshape(-1)[i],
      self.n.reshape(-1)[i]) for i in range(self.adim)]
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
    super().__init__(sdim)
    self.n = (make3D(n).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), self.adim), self.adim))
    self.sspcdim = np.prod(self.n)
    self.value_table = np.zeros(sspcdim)

  def _sidx(self, s):
    s_idx = np.round((self.n - 1) * (s - self.smin) / (self.smax -
      self.smin)).astype(int)
    return np.ravel_multi_index(s_idx.transpose((1, 0, 2)),
        self.n.reshape(-1)).reshape((s.shape[0], 1, s.shape[2]))

  def value(self, s):
    s = make3D(s, self.sdim)
    sidx = self._sidx(s)
    return self.value_table[sidx]

  def set_value(self, s, v):
    v = make3D(v, 1)
    s = make3D(s, self.sdim)
    sidx = self._sidx(s)

    dv = np.linalg.norm(self.value_table[sidx] - v)

    self.value_table[sidx] = v
    return dv
###############################################################################


# Environments ################################################################
class Environment:
  def __init__(self):
    pass

  def next_state(self, a, ns):
    raise NotImplementedError

  def reward(self, s, a, ns):
    raise NotImplementedError

class FrozenLake(Environment):
  def __init__(self):
    self.smin = make3D([0.0, -1.0, 0.0, -1.0], 4)
    self.smax = make3D([1.0, 1.0,  1.0,  1.0], 4)
    self.amin = make3D([0.0, 0.0], 2)
    self.amax = make3D([0.1, 0.1], 2)
    self.sdim = 4
    self.adim = 2

    self.gridn = 8
    self.holes = np.array([[1, 1], [3, 5], [7, 1], [7, 2], [5, 5], [2, 6], [0,
      4], [3, 6]])
    self.goals = np.array([[6, 2]])

    def double_int(x, t, p):
      a = p
      return np.hstack([x[:, 1:2], a[:, 0:1], x[:, 3:4], a[:, 1:2]])
    self.f = double_int

    self.h = 1e-2
    self.dt = 1e-1

    self.mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    self.S = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
    self.Sinv = np.linalg.inv(self.S)
    self.dn = np.array([1, 1, 1, 1, 5, 5])

    self.gamma = 0.5

  def reward(self, s, a, ns): # required
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    ns = make3D(ns, self.sdim)
    term_mask = self.is_terminal(s)
    hole_mask = self._is_in_discrete(ns, self.holes)
    goal_mask = self._is_in_discrete(ns, self.goals)
    return (1.0 - term_mask) * (-1.0 * hole_mask + 1.0 * goal_mask)

  def _is_in_discrete(self, s, points):
    pos_min = self.smin[:, [0, 2]]
    pos_max = self.smax[:, [0, 2]]
    pos = s[:, [0, 2]]
    gridn = make3D([self.gridn, self.gridn], 2)
    posd = np.round((gridn - 1) * (pos - pos_min) / (pos_max - pos_min))

    mask = np.any(np.array([np.all(np.equal(posd, make3D(point, 2)), axis=1)
      for point in points]), axis=0).reshape((pos.shape[0], 1, pos.shape[2]))
    return mask

  def is_terminal(self, s):
    s = make3D(s, self.sdim)
    hole_mask = self._is_in_discrete(s, self.holes)
    goal_mask = self._is_in_discrete(s, self.goals)
    return np.logical_or(goal_mask, hole_mask)

  def disturb(self):
    (d, p) = normal_dist(self.S, self.dn, Sinv=self.Sinv)
    return (d + self.mu.reshape(-1, 1), p)

  def next_state(self, s, a): # required
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    h = 1e-1
    ns = rk4_fn(self.f, s, 0.0, self.dt, self.h, a)
    ns = np.clip(ns, self.smin, self.smax)
    return ns

  def sample_states(self, N):
    return (np.random.rand(N, self.smin.size) * (self.smax - self.smin) +
        self.smin)

class DiscreteEnvironment(Environment):
  def __init__(self, env, n):
    self.env = env
    self.sdim = self.env.sdim
    self.smin = make3D(self.env.smin, self.sdim)
    self.smax = make3D(self.env.smax, self.sdim)

    assert np.size(n) == 1 or np.size(n) == self.sdim
    self.n = (make3D(n, env.sdim).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), env.sdim), env.sdim))

    self.gamma = self.env.gamma

  def _s2sd(self, s):
    s = (np.round((self.n - 1) * (s - self.smin) / (self.smax - self.smin)) /
        (self.n - 1))
    return s

  def all_states(self):
    slin = [np.linspace(self.smin.reshape(-1)[i], self.smax.reshape(-1)[i],
      self.n.reshape(-1)[i]) if self.n.reshape(-1)[i] > 1 else
      np.array([(self.smin.reshape(-1)[i] + self.smax.reshape(-1)[i]) / 2]) for
      i in range(self.sdim)]
    sspc = np.hstack([sarr.reshape((-1, 1)) for sarr in np.meshgrid(*slin)])
    sspc = make3D(sspc, self.sdim)
    return sspc

  def next_state(self, s, a):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    ns = env.next_state(sd, a)
    ns = make3D(ns, self.sdim)
    nsd = self._s2sd(ns)
    return nsd

  def reward(self, s, a, ns):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    ns = make3D(ns, self.sdim)
    nsd = self._s2sd(ns)
    return self.env.reward(sd, a, nsd)
###############################################################################


# Value Iteration #############################################################
class PolicyIteration:
  def __init__(self, env, pol):
    self.env = env
    self.pol = pol

  def iterate(self):
    raise NotImplementedError

class TabularPolicyIteration:
  def __init__(self, env, pol, sn, an):
    super().__init__(env, pol)
    self.value_function = TabularValueFunction(env.smin, env.smax, sn)
    self.env = DiscreteEnvironment(env, sn)
    self.all_s = self.env.all_states()

  def iterate(self):
    a = self.pol.choose_action(all_s)

    (d, p) = self.env.disturb()
    d = make3D(d, self.sdim + self.adim).transpose((2, 1, 0))
    p = make3D(p, 1)
    all_s += d[:, 0:self.env.sdim, :]
    a += d[:, self.env.sdim:, :]

    ns = self.env.next_state(all_s, a)
    v = self.value_function.value(ns)
    r = self.env.reward(all_s, a, ns)

    expected_v = make3D(np.sum(p * (r + self.env.gamma * v), axis=2), 1)

    dv = self.value_function.set_value(all_s, expected_v)

    return dv
###############################################################################
