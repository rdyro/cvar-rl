from __future__ import division
from __future__ import print_function

from util import *
import gym
from copy import deepcopy
from copy import copy
#import fortran

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

  def sample_states(self, N):
    return (np.random.rand(N, self.smin.shape[1], 1) * (self.smax - self.smin)
        + self.smin)

class GymWrapper(Environment):
  def __init__(self, env_name):
    self.gym_env = gym.make(env_name).env
    self.gym_env.reset()

    self.gamma = 1.0

    self.is_adiscrete = isinstance(self.gym_env.action_space,
        gym.spaces.Discrete)
    if self.is_adiscrete:
      self.adim = 1
      self.amin = make3D([0], self.adim)
      self.amax = make3D([self.gym_env.action_space.n - 1], self.adim)
    else:
      self.adim = np.size(self.gym_env.action_space.low)
      self.amin = make3D(self.gym_env.action_space.low, self.adim)
      self.amax = make3D(self.gym_env.action_space.high, self.adim)

    self.is_sdiscrete = isinstance(self.gym_env.observation_space,
        gym.spaces.Discrete)
    if self.is_sdiscrete:
      self.sdim = 1
      self.smin = make3D([0], self.sdim)
      self.smax = make3D([self.gym_env.observation_space.n], self.sdim)
    else:
      self.sdim = np.size(self.gym_env.observation_space.low)
      self.smin = make3D(self.gym_env.observation_space.low, self.sdim)
      self.smax = make3D(self.gym_env.observation_space.high, self.sdim)

  def next_state_sample(self, s, a):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    (s, a) = match_03(s, a)

    assert s.shape[2] == 1
    assert a.shape[2] == 1

    s_list = [tuple(s[i, :, :].reshape(-1)) for i in range(s.shape[0])]
    a_list = [tuple(a[i, :, :].reshape(-1)) for i in range(s.shape[0])]

    envs = [copy(self.gym_env) for i in range(s.shape[0])]
    for i in range(s.shape[0]):
      envs[i].state = s[i, :, :]

    if self.is_adiscrete:
      (ns_list, r_list, done_list, _) = zip(*[
          envs[i].step(int(round(a_list[i][0]))) for i in range(s.shape[0])])
    else:
      (ns_list, r_list, done_list, _) = zip(*[
          envs[i].step(a_list[i]) for i in range(s.shape[0])])

    ns = np.vstack([ns.reshape((1, -1)) for ns in ns_list])
    ns_list = [tuple(ns.reshape(-1)) for ns in ns_list]

    # overwriting cached rewards
    self.r_dict = dict(zip(zip(s_list, a_list, ns_list), r_list))
    # overwriting cached done
    self.done_dict = dict(zip(ns_list, done_list))

    return (make3D(ns, self.sdim), make3D(np.repeat(1.0, ns.shape[0]), 1))

  def sample_states(self, N, init=False):
    #if init:
    envs = [copy(self.gym_env) for i in range(N)]
    [envs[i].reset() for i in range(N)]
    s = make3D(np.vstack([envs[i].state.reshape((1, -1)) for i in range(N)]), 
        self.sdim)
    #else:
    #s = np.random.randn(N, 1, self.adim)
    #s = s * (self.smax - self.smin) + self.smin
    return s

  def reward(self, s, a, ns):
    # this only works when called immediately after next_state_sample
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    ns = make3D(ns, self.sdim)
    (s, a, ns) = match_03(s, a, ns)
    (s, layer_nb) = unstack2D(s)
    (a, _) = unstack2D(a)
    (ns, _) = unstack2D(ns)
    s_list = [tuple(s[i, :].reshape(-1)) for i in range(s.shape[0])]
    a_list = [tuple(a[i, :].reshape(-1)) for i in range(s.shape[0])]
    ns_list = [tuple(ns[i, :].reshape(-1)) for i in range(s.shape[0])]
    
    r_list = [self.r_dict[key] for key in zip(s_list, a_list, ns_list)]
    done_list = [self.done_dict[key] for key in ns_list]

    r = make3D(np.array(r_list), 1)
    done = make3D(np.array(done_list), 1)
    return stack2D(r, layer_nb)

  def is_terminal(self, s):
    # this only works when called immediately after next_state_sample
    s = make3D(s, self.sdim)
    (s, layer_nb) = unstack2D(s)

    s_list = [tuple(s[i, :].reshape(-1)) for i in range(s.shape[0])]

    done_list = [self.done_dict[key] for key in s_list]

    done = make3D(np.array(done_list), 1)
    return stack2D(done, layer_nb)

  def render(self, s):
    s = make3D(s, self.sdim)
    assert s.size == self.sdim
    old_s = self.gym_env.state
    self.gym_env.state = s.reshape(-1)
    self.gym_env.render()
    self.gym_env.state = old_s
    return

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
    
    self.p = np.array([0.1, 0.1, 0.1, 0.1]).reshape((1, 1, -1))
    self.move = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]
        ).reshape((4, 2, 1)).transpose((2, 1, 0))
    self.aspc = np.array([0, 1, 2, 3])

  def reward(self, s, a, ns):
    hole_mask = is_in_discrete(ns, self.holes, self.gridn, self.smin,
        self.smax)
    goal_mask = is_in_discrete(ns, self.goals, self.gridn, self.smin,
        self.smax)
    return (-10.0 * hole_mask + 5.0 * goal_mask) 

  def next_state_full(self, s, a):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    assert s.shape[2] == 1
    assert a.shape[2] == 1
    a = np.clip(np.round(a), self.amin, self.amax)

    ns = np.copy(s)
    (ns, p) = match_03(ns, self.p)

    a_full = np.broadcast_to(a, (p.shape[0], a.shape[0], 1))
    for i in self.aspc:
      p[np.all(a_full == i, axis=1).reshape(-1), :, i] += 0.6

    ns = ns + self.move
    ns = np.clip(ns, self.smin, self.smax)

    return (ns, p)

  def next_state_sample(self, s, a):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)
    assert s.shape[2] == 1
    assert a.shape[2] == 1
    a = np.clip(np.round(a), self.amin, self.amax)

    # obtain cp for each action
    ns = np.copy(s)
    (_, p) = match_03(ns, self.p)
    a_full = np.broadcast_to(a, (p.shape[0], self.adim, 1))
    for i in self.aspc:
      p[np.all(a_full == i, axis=1).reshape(-1), :, i] += 0.6
    cp = (np.cumsum(p, axis=2) / np.sum(p, axis=2).reshape((p.shape[0],
      p.shape[1], 1)))

    # sample actions from each cp separately
    r = np.random.rand(s.shape[0])
    idx = np.sum(r.reshape((-1, 1, 1)) > cp, axis=2).reshape(-1)

    ns = s + make3D(self.move[np.repeat(0, s.shape[0]), :, idx], self.sdim)
    ns = np.clip(ns, self.smin, self.smax)

    # ensure a terminal state is terminal
    term_mask = self.is_terminal(s)
    term_mask = np.repeat(term_mask, self.sdim, axis=1)
    ns[term_mask] = s[term_mask]

    p = make3D(p[np.arange(s.shape[0]), :, idx], 1)
    return (ns, p)

  def is_terminal(self, s):
    s = make3D(s, self.sdim)
    hole_mask = is_in_discrete(s, self.holes, self.gridn, self.smin,
        self.smax)
    goal_mask = is_in_discrete(s, self.goals, self.gridn, self.smin,
        self.smax)
    return np.logical_or(hole_mask, goal_mask)

  def sample_states(self, N):
    return np.random.randint(8 + 1, size=(N, self.sdim, 1))

"""
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
    d = make3D(d, 2 * self.sdim + self.adim).transpose((2, 1, 0))
    p = p.reshape((1, 1, -1))
    
    s = s + d[:, 0:self.sdim, :]
    s = np.clip(s, self.smin, self.smax)

    a = a + d[:, self.sdim:(self.sdim + self.adim), :]
    a = np.clip(a, self.amin, self.amax)

    h = 1e-1
    ns = rk4_fn(self.f, s, 0.0, self.dt, self.h, a)

    ns = ns + d[:, -self.sdim:, :]
    ns = np.clip(ns, self.smin, self.smax)
    return (ns, p)

  def next_state_sample(self, s, a):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)

    (s, a) = match_03(s, a)

    s = np.clip(s, self.smin, self.smax)
    a = np.clip(a, self.amin, self.amax)

    (s, layer_nb) = unstack2D(s)
    (a, layer_nb2) = unstack2D(s)
    assert layer_nb == layer_nb2

    (d, p) = normal_dist(self.S, self.dn, Sinv=self.Sinv)
    cp = np.cumsum(p) / np.sum(p)
    r = np.random.rand(s.shape[0])
    #idx = fortran.sample_from_cp(cp, r)
    idx = None
    
    s = s + d[idx, 0:self.sdim]
    s = np.clip(s, self.smin, self.smax)

    a = a + d[idx, self.sdim:(self.sdim + self.adim)]
    a = np.clip(a, self.amin, self.amax)

    h = 1e-1
    ns = rk4_fn(self.f, s, 0.0, self.dt, self.h, a)

    ns = ns + d[idx, -self.sdim:]
    ns = np.clip(ns, self.smin, self.smax)

    ns = stack2D(ns, layer_nb)
    return (ns, p[idx])
"""


class DiscreteEnvironment(Environment):
  def __init__(self, environment, n):
    self.environment = environment
    self.sdim = self.environment.sdim
    self.smin = make3D(self.environment.smin, self.sdim)
    self.smax = make3D(self.environment.smax, self.sdim)

    self.adim = self.environment.adim
    self.amin = make3D(self.environment.amin, self.adim)
    self.amax = make3D(self.environment.amax, self.adim)

    assert np.size(n) == 1 or np.size(n) == self.sdim
    self.n = (make3D(n, environment.sdim).astype(int) if np.size(n) > 1 else
        make3D(np.repeat(int(n), environment.sdim), environment.sdim))

    self.gamma = self.environment.gamma

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
    (ns, p) = self.environment.next_state_full(sd, a)
    ns = make3D(ns, self.sdim)
    nsd = self._s2sd(ns)
    return (nsd, p)

  def next_state_sample(self, s, a):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    (ns, p) = self.environment.next_state_sample(sd, a)
    ns = make3D(ns, self.sdim)
    nsd = self._s2sd(ns)
    return (nsd, p)

  def reward(self, s, a, ns):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    ns = make3D(ns, self.sdim)
    nsd = self._s2sd(ns)
    return self.environment.reward(sd, a, nsd)

  def is_terminal(self, s):
    s = make3D(s, self.sdim)
    sd = self._s2sd(s)
    return self.environment.is_terminal(sd)

  def sample_states(self, N):
    return self._s2sd(self.environment.sample_states(N))
###############################################################################
