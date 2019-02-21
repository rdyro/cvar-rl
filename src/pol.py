from __future__ import division
from __future__ import print_function

from util import *

# POLICIES ####################################################################
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
###############################################################################
