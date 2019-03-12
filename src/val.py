from __future__ import division
from __future__ import print_function

from util import *
import mod

# Value Function ##############################################################
class ValueFunction:
  def __init__(self, sdim):
    self.sdim = sdim

  def value(self, s):
    raise NotImplementedError

  def qvalue(self, environment, s, a):
    raise NotImplementedError

class ModelValueFunction(ValueFunction):
  def __init__(self, smin, smax, model):
    super().__init__(np.size(smin))
    assert self.sdim == np.size(smax)
    self.model = model

  def set_value(self, s, v):
    s = make3D(s, self.sdim)
    v = make3D(v, self.sdim)
    s_2D = np.vstack([s[:, :, i] for i in range(s.shape[2])])
    v_2D = np.vstack([v[:, :, i] for i in range(v.shape[2])])

    old_v = self.model.predict(s_2D)
    dv = np.linalg.norm(v_2D - old_v)

    self.model.train(s_2D, v_2D, 250)
    return dv

  def qvalue(self, environment, s, a):
    assert self.sdim == environment.sdim
    s = make3D(s, environment.sdim)
    a = make3D(a, environment.adim)
    try:
      (ns, p) = environment.next_state_full(s, a)
    except:
      (ns, _) = environment.next_state_sample(s, a)
      p = make3D(np.repeat(1.0, ns.shape[0]), 1)

    ns = make3D(ns, self.sdim)

    (s, a, ns) = match_03(s, a, ns)
    r = environment.reward(s, a, ns)

    ns_2D = np.vstack([ns[:, :, i] for i in range(ns.shape[2])])
    v_2D = self.model.predict(ns_2D)
    v = np.dstack([v_2D[(s.shape[0] * i):(s.shape[0] * (i + 1)), :] for i in
      range(ns.shape[2])])

    term_mask = environment.is_terminal(ns)
    expected_v = make3D(
        np.sum(p * (r + (1.0 - term_mask) * environment.gamma * v), 
          axis=2), 1)
    return expected_v

  def value(self, s):
    s = make3D(s, self.sdim)
    s_2D = np.vstack([s[:, :, i] for i in range(s.shape[2])])
    v_2D = self.model.predict(s_2D)
    v = np.dstack([v_2D[(s.shape[0] * i):(s.shape[0] * (i + 1))] for i in
      range(s.shape[2])])
    return v

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

  def qvalue(self, environment, s, a):
    assert self.sdim == environment.sdim
    s = make3D(s, environment.sdim)
    a = make3D(a, environment.adim)
    (ns, p) = environment.next_state_full(s, a)

    v = self.value(ns)

    (s, a, ns) = match_03(s, a, ns)
    r = environment.reward(s, a, ns)

    term_mask = environment.is_terminal(s)
    expected_v = make3D(
        np.sum(p * (1.0 - term_mask) * (r + environment.gamma * v), 
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

