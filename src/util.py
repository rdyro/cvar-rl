from __future__ import division
from __future__ import print_function
import numpy as np

# utility functions ###########################################################
def normal_dist(S, n=5, sig_span=3.0, **kwargs):
  Sinv = kwargs["Sinv"] if "Sinv" in kwargs else np.linalg.inv(S)

  dim = S.shape[0]

  n = np.repeat(n, dim) if np.size(n) == 1 else np.array(n)
  sig = np.sqrt(np.diag(S))
  rngs = [np.linspace(-sig_span * sig[i], sig_span * sig[i], n.reshape(-1)[i])
      if n.reshape(-1)[i] > 1 else np.array([0.0]) for i in range(dim)]
  dists = np.meshgrid(*rngs)
  dists = [dist.reshape((-1, 1)) for dist in dists]
  d = np.hstack(dists)
  p = (2 * np.pi * np.linalg.det(S))**(-0.5) * np.exp(-0.5 * np.sum(d *
    np.dot(Sinv, d.T).T, axis=1)).reshape(-1)
  p /= np.sum(p)
  return (d, p)

def rk4_fn(f, x, t, tn, h, p):
  steps = np.ceil((tn - t) / h).astype(int)
  h = (tn - t) / steps

  for _ in range(steps):
    k1 = f(x, t, p)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h, p)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h, p)
    k4 = f(x + h * k3, t + h, p)
    x += (h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
    t += h
  return x

def make3D(x, xdim):
  return (np.array(x).reshape((-1, xdim, 1)) if len(np.shape(x)) <= 2 else
      np.array(x))

def match_03(*args):
  args = [np.array(arg) for arg in args]
  max_shape = np.max([arg.shape for arg in args], axis=0)
  args = [arg if arg.shape[0] == max_shape[0] and arg.shape[2] == max_shape[2]
      else np.copy(np.broadcast_to(arg, (max_shape[0], arg.shape[1],
        max_shape[2]))) for arg in args]
  return args

def is_in_discrete(s, points, gridn, smin, smax):
  sdim = np.size(smin)
  assert sdim == np.size(smax)
  s = make3D(s, sdim)
  smin = make3D(smin, sdim)
  smax = make3D(smax, sdim)
  gridn = (make3D(gridn, sdim) if np.size(gridn) > 1 else
      make3D(np.repeat(gridn, sdim), sdim))
  sd = np.round((gridn - 1) * (s - smin) / (smax - smin))

  mask = np.any(np.array([np.all(np.equal(sd, make3D(point, 2)), axis=1)
    for point in points]), axis=0).reshape((s.shape[0], 1, s.shape[2]))
  return mask

def unstack2D(x):
  x = np.array(x)
  x = x if len(x.shape) != 1 else x.reshape((-1, 1))
  layer_nb = x.shape[2] if len(x.shape) == 3 else 1
  x = (np.vstack([x[:, :, i] for i in range(x.shape[2])]) if
      len(x.shape) == 3 else x.reshape((-1, x.shape[1])))
  return (x, layer_nb)

def stack2D(x, layer_nb):
  x = np.array(x)
  assert len(x.shape) <= 2 or x.shape[2] == 1
  x = x if len(x.shape) == 2 else x.reshape((-1, 1))
  assert x.shape[0] % layer_nb == 0
  rows_per_layer = x.shape[0] // layer_nb
  x = np.dstack([x[(i * rows_per_layer):((i + 1) * rows_per_layer), :] for i in
    range(layer_nb)])
  return x
###############################################################################
