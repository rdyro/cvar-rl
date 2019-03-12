from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append(os.path.dirname(__file__) + "../src")
from tf_util import *
from util import *

# test stack2D, unstack2D
A = np.ones((10, 2, 5))
for i in range(A.shape[2]):
  A[:, :, i] += i
(B, layer_nb) = unstack2D(A)
C = stack2D(B, layer_nb)
assert np.all(A == C)

v1 = np.ones((100, 20))
v2 = np.ones(100)
assert np.all(unstack2D(v1)[0] == v1)
assert np.all(unstack2D(v2)[0] == v2)
assert len(unstack2D(v2)[0].shape) == 2
