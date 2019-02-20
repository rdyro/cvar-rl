from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


# utility functions ###########################################################
def normal_dist(S, n=5, sig_span=3.0, **kwargs):
  Sinv = kwargs["Sinv"] if "Sinv" in kwargs else np.linalg.inv(S)

  dim = S.shape[0]

  n = np.repeat(n, dim) if np.size(n) == 1 else np.array(n)
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

def match_03(*args):
  args = [np.array(arg) for arg in args]
  max_shape = np.max([arg.shape for arg in args], axis=0)
  args = [arg if arg.shape[0] == max_shape[0] and arg.shape[2] == max_shape[2]
      else np.copy(np.broadcast_to(arg, (max_shape[0], arg.shape[1],
        max_shape[2]))) for arg in args]
  return args

def is_in_discrete(s, points, gridn, smin, smax):
  sdim = np.size(smin)
  assert sdim == np.size(smax)
  s = make3D(s, sdim)
  smin = make3D(smin, sdim)
  smax = make3D(smax, sdim)
  gridn = (make3D(gridn, sdim) if np.size(gridn) > 1 else
      make3D(np.repeat(gridn, sdim), sdim))
  sd = np.round((gridn - 1) * (s - smin) / (smax - smin))

  mask = np.any(np.array([np.all(np.equal(sd, make3D(point, 2)), axis=1)
    for point in points]), axis=0).reshape((s.shape[0], 1, s.shape[2]))
  return mask
###############################################################################

# POLICIES ####################################################################
class Policy:
  def __init__(self, sdim, adim):
    self.sdim = sdim
    self.adim = adim
    self.env = None
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

  def set_env(self, env):
    self.env = env
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
    V = np.array([self.value_function.qvalue(self.env, s, self.aspc[i:i+1, :,
      :]).reshape(-1) for i in range(self.aspcdim)])
    print("CHOOSE OPTIMAL ACTION")
    print(V)
    a_argmax = np.argmax(V, axis=0)
    print(a_argmax)
    print(self.aspc[a_argmax, 0:])
    return self.aspc[a_argmax, 0:]

  def choose_action_discrete(self, s, n):
    return super().choose_action_discrete(s, n, self.amin, self.amax)
###############################################################################


# Value Function ##############################################################
class ValueFunction:
  def __init__(self, sdim):
    self.sdim = sdim

  def value(self, s):
    raise NotImplementedError

  def qvalue(self, env, s, a):
    raise NotImplementedError

class TabularValueFunction(ValueFunction):
  def __init__(self, smin, smax, n):
    super().__init__(np.size(smin))
    assert self.sdim == np.size(smax)
    assert np.size(n) == 1 or np.size(n) == self.sdim
    self.n = (make3D(n).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), self.sdim), self.sdim))
    self.smin = make3D(smin, self.sdim)
    self.smax = make3D(smax, self.sdim)
    self.sspcdim = np.prod(self.n)
    self.value_table = np.zeros(self.sspcdim)

  def _sidx(self, s):
    s_idx = np.round((self.n - 1) * (s - self.smin) / (self.smax -
      self.smin)).astype(int)
    return np.ravel_multi_index(s_idx.transpose((1, 0, 2)),
        self.n.reshape(-1)).reshape((s.shape[0], 1, s.shape[2]))

  def value(self, s):
    s = make3D(s, self.sdim)
    sidx = self._sidx(s)
    return self.value_table[sidx]

  def qvalue(self, env, s, a):
    assert self.sdim == env.sdim
    s = make3D(s, env.sdim)
    a = make3D(a, env.adim)
    (ns, p) = env.next_state_full(s, a)

    v = self.value(ns)

    (s, a, ns) = match_03(s, a, ns)
    r = env.reward(s, a, ns)

    term_mask = env.is_terminal(s)
    expected_v = make3D(np.sum(p * (1.0 - term_mask) * (r + env.gamma * v),
      axis=2), 1)
    return expected_v

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

  def next_state_full(self, s, a):
    raise NotImplementedError

  def reward(self, s, a, ns):
    raise NotImplementedError

class Mars(Environment):
  def __init__(self):
    self.sdim = 2
    self.adim = 1
    self.smin = make3D([0, 0], self.sdim)
    self.smax = make3D([8, 8], self.sdim)
    self.amin = make3D([0], self.adim)
    self.amax = make3D([3], self.adim)

    self.holes = np.array([[3, 3], [6, 6]])
    #self.holes = np.array([[9, 9]])
    self.goals = np.array([[0, 0]])
    self.gridn = 9

    self.gamma = 0.95
    
    self.d = np.array([-10, 0, 1, 2, 3])
    self.p = np.array([0.6, 0.1, 0.1, 0.1, 0.1]).reshape((1, 1, -1))

  def reward(self, s, a, ns):
    hole_mask = is_in_discrete(ns, self.holes, self.gridn, self.smin,
        self.smax)
    goal_mask = is_in_discrete(ns, self.goals, self.gridn, self.smin,
        self.smax)
    return (-10.0 * hole_mask + 5.0 * goal_mask) 

  def next_state_full(self, s, a):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    a = np.clip(np.round(a), self.amin, self.amax)

    a = np.copy(np.broadcast_to(a, (a.shape[0], self.adim, self.d.size)))
    for i in range(1, self.d.size):
      a[:, :, i] = self.d[i]
    (s, a) = match_03(s, a)

    dist = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    aspc = np.array([0, 1, 2, 3])

    ns = np.copy(s)
    for i in range(aspc.size):
      (idx1, idx2) = np.where(np.all(a == aspc[i], axis=1))
      ns[idx1, :, idx2] = s[idx1, :, idx2] + dist[i].reshape((1, 2))

    ns = np.clip(ns, self.smin, self.smax)
    return (ns, self.p)

  def is_terminal(self, s):
    s = make3D(s, self.sdim)
    hole_mask = is_in_discrete(s, self.holes, self.gridn, self.smin,
        self.smax)
    goal_mask = is_in_discrete(s, self.goals, self.gridn, self.smin,
        self.smax)
    return np.logical_or(hole_mask, goal_mask)

  def _disturb(self, s, a, ns):
    d = make3D([[0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0]], 2 * self.sdim + self.adim)
     
    p = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    return (d, p)

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
    self.S = np.eye(self.sdim + self.adim + self.sdim)
    self.S[self.sdim + 0, self.sdim + 0] = 2
    self.S[self.sdim + 1, self.sdim + 1] = 2
    self.Sinv = np.linalg.inv(self.S)
    self.dn = np.array([1, 1, 1, 1, 5, 5, 1, 1, 1, 1])

    self.gamma = 0.5

  def reward(self, s, a, ns):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    ns = make3D(ns, self.sdim)

    pos = make3D(s[:, [0, 2]], 2)
    hole_mask = is_in_discrete(pos, self.holes, self.gridn, self.smin,
      self.smax)
    goal_mask = is_in_discrete(pos, self.goals, self.gridn, self.smin,
      self.smax)
    return (-1.0 * hole_mask + 1.0 * goal_mask)

  def is_terminal(self, s):
    s = make3D(s, self.sdim)
    pos = make3D(s[:, [0, 2]], 2)
    hole_mask = is_in_discrete(pos, self.holes, self.gridn, self.smin,
        self.smax)
    goal_mask = is_in_discrete(pos, self.goals, self.gridn, self.smin,
        self.smax)
    return np.logical_or(goal_mask, hole_mask)

  def disturb(self):
    (d, p) = normal_dist(self.S, self.dn, Sinv=self.Sinv)
    return (d + self.mu.reshape((1, -1)), p)

  def next_state_full(self, s, a):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)

    (s, a) = match_03(s, a)

    s = np.clip(s, self.smin, self.smax)
    a = np.clip(a, self.amin, self.amax)

    (d, p) = normal_dist(self.S, self.dn, Sinv=self.Sinv)
    d = make3D(d, 2 * env.sdim + env.adim).transpose((2, 1, 0))
    p = p.reshape((1, 1, -1))
    
    s = s + d[:, 0:env.sdim, :]
    s = np.clip(s, env.smin, env.smax)

    a = a + d[:, env.sdim:(env.sdim + env.adim), :]
    a = np.clip(a, env.amin, env.amax)

    ns = ns + d[:, -env.sdim:, :]
    ns = np.clip(ns, self.smin, self.smax)

    h = 1e-1
    ns = rk4_fn(self.f, s, 0.0, self.dt, self.h, a)
    ns = np.clip(ns, self.smin, self.smax)

    ns = ns + d[:, -env.sdim:, :]
    ns = np.clip(ns, self.smin, self.smax)
    return (ns, p)

  def sample_states(self, N):
    return (np.random.rand(N, self.smin.size) * (self.smax - self.smin) +
        self.smin)

class DiscreteEnvironment(Environment):
  def __init__(self, env, n):
    self.env = env
    self.sdim = self.env.sdim
    self.smin = make3D(self.env.smin, self.sdim)
    self.smax = make3D(self.env.smax, self.sdim)

    self.adim = self.env.adim
    self.amin = make3D(self.env.amin, self.adim)
    self.amax = make3D(self.env.amax, self.adim)

    assert np.size(n) == 1 or np.size(n) == self.sdim
    self.n = (make3D(n, env.sdim).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), env.sdim), env.sdim))

    self.gamma = self.env.gamma

  def _s2sd(self, s):
    s = (np.round((self.n - 1) * (s - self.smin) / (self.smax - self.smin)) /
        (self.n - 1) * (self.smax - self.smin))
    return s

  def all_states(self):
    slin = [np.linspace(self.smin.reshape(-1)[i], self.smax.reshape(-1)[i],
      self.n.reshape(-1)[i]) if self.n.reshape(-1)[i] > 1 else
      np.array([(self.smin.reshape(-1)[i] + self.smax.reshape(-1)[i]) / 2.0]) for
      i in range(self.sdim)]
    sspc = np.hstack([sarr.reshape((-1, 1)) for sarr in np.meshgrid(*slin)])
    sspc = make3D(sspc, self.sdim)
    return sspc

  def next_state_full(self, s, a):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    (ns, p) = self.env.next_state_full(sd, a)
    ns = make3D(ns, self.sdim)
    nsd = self._s2sd(ns)
    return (nsd, p)

  def reward(self, s, a, ns):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    ns = make3D(ns, self.sdim)
    nsd = self._s2sd(ns)
    return self.env.reward(sd, a, nsd)

  def is_terminal(self, s):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    return self.env.is_terminal(sd)

  def disturb(self):
    return self.env.disturb()
###############################################################################


# Value Iteration #############################################################
class Solver:
  def __init__(self, env, pol):
    self.env = env
    self.pol = pol

  def iterate(self):
    raise NotImplementedError

class TabularDiscreteSolver(Solver):
  def __init__(self, env, pol, n):
    super().__init__(env, pol)
    self.value_function = TabularValueFunction(env.smin, env.smax, n)
    self.env = DiscreteEnvironment(env, n)
    self.all_s = self.env.all_states()
    pol.set_env(self.env)
    pol.set_value_function(self.value_function)

  def iterate(self):
    a = self.pol.choose_action(self.all_s)
    expected_v = self.value_function.qvalue(self.env, self.all_s, a)
    dv = self.value_function.set_value(self.all_s, expected_v)

    return dv
###############################################################################
