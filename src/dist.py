import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
import time

def normal_dist(S, n=5, sig_span=3.0, **kwargs):
  Sinv = kwargs["Sinv"] if "Sinv" in kwargs else np.linalg.inv(S)

  dim = S.shape[0]

  n = np.array(n)
  n = np.repeat(n, dim) if n.size == 1 else n
  sig = np.sqrt(np.diag(S))
  rngs = [np.linspace(-sig_span * sig[i], sig_span * sig[i], n[i]) if n[i] > 1
      else np.array([0.0]) for i in range(dim)]
  dists = np.meshgrid(*rngs)
  dists = [dist.reshape((-1, 1)) for dist in dists]
  d = np.hstack(dists)
  p = (2 * np.pi * np.linalg.det(S))**(-0.5) * np.exp(-0.5 * np.sum(d *
    np.dot(Sinv, d.T).T, axis=1))
  p /= np.sum(p)
  return (d, p)

S = np.array([[30, 0.5],
              [0.5, 1]])
S = np.eye(7)
#Sinv = np.linalg.inv(S)

t1 = time.time()
(d, p) = normal_dist(S)
t2 = time.time()

print(d)
print(p)
print(t2 - t1)

"""
(d, p) = normal_dist(S, Sinv, 30)

xx, yy = np.meshgrid(d[:, 0], d[:, 1])
zz = griddata((d[:, 0], d[:, 1]), p, (xx, yy), method="linear")

fig = pl.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=True)
pl.show()
"""
