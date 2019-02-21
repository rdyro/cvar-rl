from __future__ import division
from __future__ import print_function

from util import *
import val
import env

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
###############################################################################
