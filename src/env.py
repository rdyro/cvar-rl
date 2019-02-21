from __future__ import division
from __future__ import print_function

from util import *

# Environments ################################################################
class Environment:
  def __init__(self):
    pass

  def next_state_full(self, s, a):
    raise NotImplementedError

  def next_state_sample(self, s, a):
    raise NotImplementedError

  def reward(self, s, a, ns):
    raise NotImplementedError

  def is_terminal(self, s):
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
###############################################################################
