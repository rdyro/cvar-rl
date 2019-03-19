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

