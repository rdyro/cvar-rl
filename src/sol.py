from __future__ import division
from __future__ import print_function

from util import *
from tf_util import *
import val
import env
import mod

# Solver ######################################################################
class Solver:
  def __init__(self, environment, policy):
    self.environment = environment
    self.policy = policy

  def iterate(self):
    raise NotImplementedError

class TabularDiscreteSolver(Solver):
  def __init__(self, environment, policy, n):
    super().__init__(environment, policy)
    self.value_function = val.TabularValueFunction(environment.smin,
        environment.smax, n)
    self.environment = env.DiscreteEnvironment(environment, n)
    self.all_s = self.environment.all_states()
    self.policy.set_environment(self.environment)
    self.policy.set_value_function(self.value_function)

  def iterate(self):
    a = self.policy.choose_action(self.all_s)
    expected_v = self.value_function.qvalue(self.environment, self.all_s, a)
    dv = self.value_function.set_value(self.all_s, expected_v)

    return dv

class ModelDiscreteSolver(Solver):
  def __init__(self, environment, policy, n, model_str):
    super().__init__(environment, policy)

    if model_str == "nn":
      model = mod.NNModel(environment.sdim, environment.adim, np.repeat(32, 2))
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      model.set_session(sess)
    elif model_str == "polyls":
      model = mod.PolyLSModel(environment.sdim, environment.adim, 10, mix=True)
    else:
      raise NotImplementedError

    self.value_function = val.ModelValueFunction(environment.smin,
        environment.smax, model)
    self.environment = env.DiscreteEnvironment(environment, n)
    self.all_s = self.environment.all_states()
    #self.all_s = self.environment.sample_states(int(1e3))
    self.policy.set_environment(self.environment)
    self.policy.set_value_function(self.value_function)

  def iterate(self):
    a = self.policy.choose_action(self.all_s)
    expected_v = self.value_function.qvalue(self.environment, self.all_s, a)
    dv = self.value_function.set_value(self.all_s, expected_v)

    return dv
 
class PolicyGradientDiscreteSolver(Solver):
  def __init__(self, environment, n, **kwargs):
    self.params = dict({"episodes_nb": 500, "episode_len": 20, "baseline":
      False}, **kwargs)
    policy = pol.SoftmaxPolicy(environment.sdim, environment.adim,
        environment.amin, environment.amax, n)
    self.sess = tf.Session()
    policy.set_session(self.sess)
    self.sess.run(tf.global_variables_initializer())
    super().__init(environment, policy)

    
    if self.params["baseline"] == True:
      self.s_ = tf.placeholder(tf.float32, shape=(None, self.environment.sdim))
      self.baseline_ = 

  def _episodes(self):
    N = self.params["episodes_nb"]
    s = self.environment.sample_states(N)
    a = np.zeros((s.shape[0], self.environment.adim, 0))
    r = np.zeros((s.shape[0], 1, 0))
    for i in range(self.params["episode_len"]):
      cs = s[:, :, -1]

      na = self.policy.choose_action(cs)
      ns = self.environment.next_state_sample(cs, na)
      nr = self.environment.reward(cs, na, ns)

      a = np.dstack([a, na])
      s = np.dstack([s, ns])
      r = np.dstack([r, nr])
    # delete last state
    s = s[:, :, :-1]
    return (s, a, r)

  def _mc_gt(self, r):
    gamma = np.power(self.environment.gamma, 
        np.arange(r.shape[1]).reshape((1, -1, 1)))
    gt = np.cumsum((r * gamma)[::-1])[::-1] / gamma
    return gt

  def iterate(self):
    (s, a, r) = self._episodes()
    gt = self._mc_gt(r)

###############################################################################,
