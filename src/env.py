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

from gym import spaces

def drone_du(u, t, p):
  g = p[0]
  m = p[1]
  rl = p[2]
  rr = p[3]
  com = p[4]
  I = p[5]
  Fmax = p[6]
  F = p[7]

  Fl = Fmax / 10.0 * F[:, 0:1]
  Fr = Fmax / 10.0 * F[:, 1:2]
  x = u[:, 0:1]
  dx = u[:, 1:2]
  y = u[:, 2:3]
  dy = u[:, 3:4]
  th = u[:, 4:5]
  dth = u[:, 5:6]
  du = np.hstack([dx, 
    -Fl * np.cos(th) / m - Fr * np.cos(th) / m,
    dy,
    Fl * np.sin(th) / m + Fr * np.sin(th) / m - g,
    #Fl * np.sin(th) / m + Fr * np.sin(th) / m,
    dth,
    (rl + com) * Fl / I - (rr - com) * Fr / I])
  return du

class Drone2D(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 30
      }

  def __init__(self):
    xmin = -10.0 # (m)
    xmax = 10.0 # (m)

    ymin = -10.0 # (m)
    ymax = 10.0 # (m)
    Tt = 0.5 # characteristic translation time (s)
    Tr = 0.2 # characteristic rotation time (s)

    self.sdim = 6
    self.smin = make3D(np.array([xmin, 
      -(xmax - xmin) / Tt,
      ymin,
      -(ymax - ymin) /Tt,
      0,
      -(2 * np.pi) / Tr
      ]), self.sdim)
    self.smax = make3D(np.array([xmax, 
      (xmax - xmin) / Tt,
      ymax,
      (ymax - ymin) /Tt,
      2 * np.pi,
      (2 * np.pi) / Tr
      ]), self.sdim)

    self.smin_reward = np.copy(self.smin)
    self.smax_reward = np.copy(self.smax)
    self.smin_reward[:, [0, 2], :] = -3.0
    self.smax_reward[:, [0, 2], :] = 3.0

    self.adim = 2
    self.amin = make3D(np.array([-10.0, -10.0]), self.adim)
    self.amax = make3D(np.array([10.0, 10.0]), self.adim)

    self.gravity = 9.81 # (m s^-2)
    self.m = 0.5 # (kg)
    self.rl = 0.2 # (m)
    self.rr = 0.2 # (m)
    self.com = 0.0 # (m)
    self.I = 5e-3 # (kg m^2)
    self.Fmax = 100.0 # (N)
    self.f = drone_du

    self.gamma = 0.99

    # openai gym requirements
    self.viewer = None

    self.action_space = spaces.Box(low=self.amin.reshape(-1),
        high=self.amax.reshape(-1))
    self.observation_space = spaces.Box(low=self.smin.reshape(-1),
        high=self.smax.reshape(-1))

    self.state = None
    self.reset()

  def reset(self):
    smin0 = np.copy(self.smin)
    smax0 = np.copy(self.smax)

    """
    smin0.reshape(-1)[0:4] = -1.0 # (m / m s^-1)
    smax0.reshape(-1)[0:4] = 1.0 # (m / m s^-1)

    smin0.reshape(-1)[4] = np.pi / 2 - 0.3 # (1)
    smax0.reshape(-1)[4] = np.pi / 2 + 0.3 # (1)

    smin0.reshape(-1)[5] = -0.3 # (s^-1)
    smax0.reshape(-1)[5] = 0.3 # (s^-1)
    """

    #s = make3D(np.random.rand(self.sdim), self.sdim) * (smax0 - smin0) + smin0
    s = make3D(np.random.rand(self.sdim), self.sdim) * (self.smax_reward -
        self.smin_reward) + self.smin_reward

    self.state = s.reshape(-1)
    return self.state
  
  def sample_states(self, N):
    smin0 = np.copy(self.smin)
    smax0 = np.copy(self.smax)

    smin0.reshape(-1)[0:4] = -1.0 # (m / m s^-1)
    smax0.reshape(-1)[0:4] = 1.0 # (m / m s^-1)

    smin0.reshape(-1)[4] = np.pi / 2 - 0.3 # (1)
    smax0.reshape(-1)[4] = np.pi / 2 + 0.3 # (1)

    smin0.reshape(-1)[5] = -0.3 # (s^-1)
    smax0.reshape(-1)[5] = 0.3 # (s^-1)
    #s = (np.random.rand(N, self.sdim, 1) * 0.5 * (self.smax_reward -
    #  self.smin_reward) + self.smin_reward + (self.smax_reward +
    #    self.smin_reward) / 2.0)

    s = np.random.rand(N, self.sdim, 1) * (smax0 - smin0) + smin0
    return s
  
  def next_state_sample(self, s, a):
    s = make3D(s, self.sdim)
    a = make3D(a, self.adim)

    (s, a) = match_03(s, a)
    s = np.clip(s, self.smin, self.smax)
    a = np.clip(a, self.amin, self.amax)

    (s, layer_nb) = unstack2D(s)
    (a, _) = unstack2D(a)

    h = 1e-2
    t = 0.0
    tn = 1.0 / 60.0
    p = [self.gravity, self.m, self.rl, self.rr, self.com, self.I, self.Fmax,
        a]
    ns = rk4_fn(self.f, s, t, tn, h, p)
    ns = make3D(ns, self.sdim)
    ns[:, 4] = np.mod(ns[:, 4], 2 * np.pi)

    ns = stack2D(ns, layer_nb)
    return (ns, np.ones((ns.shape[0], 1, ns.shape[2])))

  def step(self, a):
    assert np.prod(np.shape(a)) == self.adim
    s = self.state
    (ns, p) = self.next_state_sample(s, a)
    self.state = ns.reshape(-1)
    r = self.reward(s, a, ns)
    return (self.state, r, self.is_terminal(self.state), {})

  def reward(self, s, a, ns):
    mask = self.is_terminal(ns)
    distance = np.linalg.norm(np.hstack([
      ns[:, 0:1],
      ns[:, 2:3],
      ns[:, 4:5] - np.pi / 2.0]), axis=1)

    #r = (100.0 - 0.1 * ns[:, 5]**2) * np.logical_not(mask)
    #r = 1.0 * np.logical_not(mask)
    r = (50.0 - distance**2) * np.logical_not(mask)
    return r

  def is_terminal(self, s):
    s = make3D(s, self.sdim)
    s0 = s[0, :, :].reshape(-1)

    """
    desc = ["x", "dx", "y", "dy", "th", "dth"]
    for i in range(self.sdim):
      if (s0[i] < self.smin_reward.reshape(-1)[i] or s0[i] > self.smax_reward.reshape(-1)[i]):
        print("Fails on [%s]" % desc[i])
    """

    mask = np.logical_or(np.any(s < self.smin_reward, axis=1), 
        np.any(s > self.smax_reward, axis=1))
    return mask

  def render2(self, s):
    s = s.reshape(-1)
    assert s.size == self.sdim
    old_s = self.state
    self.state = s
    self.render()
    self.state = old_s
    return

  def render(self, mode="human", refresh=False):
    screen_width = 800
    screen_height = 600

    world_width = self.smax.reshape(-1)[0] - self.smin.reshape(-1)[0]
    world_scale = screen_width / world_width
    
    body_scale = 1.0
    len_scale = 20
    width = 20 * body_scale
    height = 10 * body_scale
    arm_width = 3 * body_scale
    arm_rl = 2.0 * self.rl / (self.rl + self.rr) * len_scale * body_scale
    arm_rr = 2.0 * self.rr / (self.rl + self.rr) * len_scale * body_scale
    arm_v = height
    prop_width = 1.5 * (arm_rl + arm_rr) / 2.0
    leg_h = width / 3.0
    leg_v = height
    com_r = body_scale * 2.5
    com_x = 2.0 * self.com / (self.rl + self.rr) * len_scale * body_scale

    if self.viewer is None or refresh == True:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)

      (l, r, t, b) = (-width / 2.0, width / 2.0, height / 2.0, -height / 2.0)
      body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      body.add_attr(rendering.Transform(translation=(-com_x, 0)))

      # left arm
      (l, r, t, b) = (-arm_rl, 0, arm_width / 2.0, -arm_width / 2.0)
      arm_l_h = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      arm_l_h.add_attr(rendering.Transform(translation=(-com_x, 0)))

      (l, r, t, b) = (-arm_rl, -arm_rl + arm_width, arm_v, 0)
      arm_l_v = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      arm_l_v.add_attr(rendering.Transform(translation=(-com_x, 0)))

      # right arm
      (l, r, t, b) = (0, arm_rr, arm_width / 2.0, -arm_width / 2.0)
      arm_r_h = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      arm_r_h.add_attr(rendering.Transform(translation=(-com_x, 0)))

      (l, r, t, b) = (arm_rr, arm_rr - arm_width, arm_v, 0)
      arm_r_v = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      arm_r_v.add_attr(rendering.Transform(translation=(-com_x, 0)))

      # propellers
      (lx, rx, ly, ry) = (-arm_rl + arm_width / 2.0 - prop_width / 2.0, 
          -arm_rl + arm_width / 2.0 + prop_width / 2.0, arm_v, arm_v)
      prop_l = rendering.Line((lx, ly), (rx, ry))
      prop_l.add_attr(rendering.Transform(translation=(-com_x, 0)))

      (lx, rx, ly, ry) = (arm_rr - arm_width / 2.0 - prop_width / 2.0, 
          arm_rr - arm_width / 2.0 + prop_width / 2.0, arm_v, arm_v)
      prop_r = rendering.Line((lx, ly), (rx, ry))
      prop_r.add_attr(rendering.Transform(translation=(-com_x, 0)))

      # landing leg
      (lx, rx, ly, ry) = (-width / 2.0, -width / 2.0 + leg_h,
          -height / 2.0 - leg_v, 0)
      leg_l = rendering.Line((lx, ly), (rx, ry))
      leg_l.add_attr(rendering.Transform(translation=(-com_x, 0)))

      (lx, rx, ly, ry) = (width / 2.0 - leg_h, width / 2.0,
          0, -height / 2.0 - leg_v)
      leg_r = rendering.Line((lx, ly), (rx, ry))
      leg_r.add_attr(rendering.Transform(translation=(-com_x, 0)))

      # center of mass
      com = rendering.make_circle(com_r)
      com.add_attr(rendering.Transform(translation=(0, 0)))
      com.set_color(1.0, 1.0, 0.0)

      # add transforms
      self.trans = rendering.Transform()
      body.add_attr(self.trans)
      arm_l_h.add_attr(self.trans)
      arm_l_v.add_attr(self.trans)
      arm_r_h.add_attr(self.trans)
      arm_r_v.add_attr(self.trans)
      prop_l.add_attr(self.trans)
      prop_r.add_attr(self.trans)
      leg_l.add_attr(self.trans)
      leg_r.add_attr(self.trans)
      com.add_attr(self.trans)

      # append objects to visualize
      self.viewer.add_geom(body)
      self.viewer.add_geom(arm_l_h)
      self.viewer.add_geom(arm_l_v)
      self.viewer.add_geom(arm_r_h)
      self.viewer.add_geom(arm_r_v)
      self.viewer.add_geom(prop_l)
      self.viewer.add_geom(prop_r)
      self.viewer.add_geom(leg_l)
      self.viewer.add_geom(leg_r)
      self.viewer.add_geom(com)

    x = self.state[0]
    y = self.state[2]
    th = self.state[4]
    self.trans.set_translation((x - self.smin.reshape(-1)[0]) * world_scale,
        (y - self.smin.reshape(-1)[2] * screen_height / screen_width) *
        world_scale)
    self.trans.set_rotation(-(th - np.pi / 2))

    return self.viewer.render(return_rgb_array=(mode=='rgb_array'))

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
