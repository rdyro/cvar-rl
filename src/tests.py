from env import *
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy.interpolate import griddata


# Policy Tests ################################################################
N = int(1e5)

p = Policy(5, 2)

amin = [0, 0]
amax = [5, 3]
sdim = 5
adim = 2
up = UnifRandPolicy(sdim, amin, amax)
a_up = up.choose_action(np.ones((N, sdim)))
assert len(a_up.shape) == 3
assert a_up.shape[2] == 1
assert np.all(np.logical_and(a_up >= make3D(amin, adim), a_up <= make3D(amax,
  adim)))
print("UnifRandomPolicy")
print("  mu ~= [2.5, 1.5]")
print(np.mean(a_up, axis=0))
print("  mu  = [%f, %f]" % tuple(np.mean(a_up, axis=0).reshape(-1)))

mu = [-1, 2]
sig = [1e-1, 10]
ip = IndepNormRandPolicy(sdim, mu, sig)
a_ip = ip.choose_action(np.ones((N, sdim)))
assert len(a_ip.shape) == 3
print(ip.choose_action(np.ones((N, sdim, 5))).shape)
assert np.all(np.equal(ip.choose_action(np.ones((N, sdim, 5))).shape, [N, adim,
  5]))
assert a_ip.shape[2] == 1
print("IndepRandomPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))
print("  std ~= [1e-1, 10]")
print("  std  = [%f, %f]" % tuple(np.std(a_ip, axis=0).reshape(-1)))

mu = [-1, 2]
sig = [1e-1, 10]
ip = IndepNormRandPolicy(sdim, mu, sig)
a_ip = ip.choose_action(np.ones((N, sdim)))
print("IndepNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(0)
pl.title("IndepNormRandPolicy")
pl.hist2d(a_ip[:, 0, 0], a_ip[:, 1, 0], bins=100)

mu = [-1, 2]
S = np.array([[2, -0.99], [-0.99, 1]])
mp = MultNormRandPolicy(sdim, mu, S)
a_mp = mp.choose_action(np.ones((N, sdim)))
print("MultNormRandPolicy")
print("  mu ~= [-1, 2]")
print("  mu  = [%f, %f]" % tuple(np.mean(a_ip, axis=0).reshape(-1)))

pl.figure(1)
pl.title("MultNormRandPolicy")
pl.hist2d(a_mp[:, 0, 0], a_mp[:, 1, 0], bins=100)

print("DiscretePolicy")
dp = DiscretePolicy(sdim, amin, amax, 5)
a_dp = dp.choose_action(np.ones((N, sdim)))
assert np.all(np.equal(dp.choose_action(np.ones((N, sdim)), 0), make3D(amin,
  adim))) 
assert np.all(np.equal(dp.choose_action(np.ones((N, sdim)), -1), make3D(amax,
  adim)))
print("  middle action = ", a_dp[1, :, 0])
print("  first action = ", dp.choose_action(np.ones((1, sdim, 1)),
  0).reshape(-1))
print("  last action = ", dp.choose_action(np.ones((1, sdim, 1)),
  -1).reshape(-1))

# Environment Tests ###########################################################
e = Environment()

fl = FrozenLake()
dfl = DiscreteEnvironment(fl, 20)
assert np.all(np.equal(dfl.all_states().shape, (20**fl.sdim, fl.sdim, 1)))
n = 20
pos = np.hstack([c.reshape((-1, 1)) for c in np.meshgrid(*[np.linspace(0, 1, n)
  for i in range(2)])])
ns = np.hstack([pos[:, 0:1], np.zeros((n**2, 1)), pos[:, 1:2],
  np.zeros((n**2, 1))])
s = np.hstack([np.ones((n**2, 1)) for i in range(fl.sdim)])

ns = make3D(ns, fl.sdim)
s = make3D(s, fl.sdim)

holes = -1 * is_in_discrete(ns[:, [0, 2]], fl.holes, fl.gridn, fl.smin[:, [0, 2]],
    fl.smax[:, [0, 2]]).reshape((n, n))
goals = 1 * is_in_discrete(ns[:, [0, 2]], fl.goals, fl.gridn, fl.smin[:, [0, 2]],
    fl.smax[:, [0, 2]]).reshape((n, n))
pl.figure(7)
pl.imshow(holes + goals)

mars = Mars()

# Solver Tests ################################################################
pol = OptimalDiscretePolicy(mars.sdim, mars.amin, mars.amax, 4)
sol = TabularDiscreteSolver(mars, pol, 9)
for i in range(100):
  sol.iterate()
all_s = sol.env.all_states()
val = sol.value_function.value(all_s)
all_a = sol.pol.choose_action(all_s)

for i in range(all_a.shape[0]):
  print("[%d, %d] -> %d" % (all_s[i, 0, 0], all_s[i, 1, 0], all_a[i, 0, 0]))

(ns, p) = sol.env.next_state_full(all_s, all_a)
(all_s_exp, all_a_exp, ns) = match_03(all_s, all_a, ns)
r = np.sum(p * sol.env.reward(all_s_exp, all_a_exp, ns), axis=2)

idxs = sol.value_function._sidx(make3D([2, 3], 2)).reshape(-1)[0]
print(ns[idxs, :, :])
print(ns[idxs, :, :].transpose((1, 0)))

uv_list = []
for a in all_a:
  a_arr = None
  if a == 0:
    a_arr = [1, 0]
  elif a == 1:
    a_arr = [0, 1]
  elif a == 2:
    a_arr = [-1, 0]
  elif a == 3:
    a_arr = [0, -1]
  uv_list.append(a_arr)
#uv = np.array([[1, 0] if a == 0 else [0, 1] if a == 1 else [-1, 0] if a == 2 else [0, -1] for a in all_a])

print(sol.pol.choose_action(make3D([4, 8], 2)))

uv = np.array(uv_list)
u = uv[:, 0]
v = uv[:, 1]
pl.figure(17)
pl.imshow(val.reshape((9, 9)), origin="lower")
pl.colorbar()
pl.quiver(all_s[:, 0].reshape((9, 9)), all_s[:, 1].reshape((9, 9)), 
          u.reshape((9, 9)), v.reshape((9, 9)))
print(TabularValueFunction(mars.smin, mars.smax, 9).qvalue(mars, [2, 3], 2))
# RK4 test ####################################################################
N = 50
K = 20
x = np.random.rand(N, 2, K)
a = np.ones((N, 1, 1))

def f(x, t, p):
  return np.hstack([x[:, 1:2], p - x[:, 0:1]])

M = int(1e2)
X = np.zeros((N, 2, K, M))
X[:, :, :, 0] = x[:, :, :]
t = np.linspace(0, 10, M)
h = 1e-2

for i in range(1, M):
  x = rk4_fn(f, x, t[i-1], t[i], h, a)
  X[:, :, :, i] = x[:, :, :]

pl.figure(10)
pl.clf()
for i in range(N):
  r = np.random.randint(K)
  pl.plot(t, X[i, 0, r, :].reshape(-1))

pl.show()
