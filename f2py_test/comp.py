import time
import fortran as f
import numpy as np
import matplotlib.pyplot as pl

N = 5
M = 10**np.linspace(0, 6, 10)
L = 5

p = np.random.rand(N)
cp = np.cumsum(p) / np.sum(p)

def numpy_sample(cp, r):
  idx = np.sum(r.reshape((-1, 1)) > cp.reshape((1, -1)), axis=1)
  idx = np.clip(idx, 0, len(cp) - 1)
  return idx

def python_sample(cp, r):
  idx = [len([j for j in range(len(cp)) if r[i] > cp[j]]) for i in
      range(len(r))]
  idx = [len(cp) if i >= len(cp) else i for i in idx]
  return idx

def naive_sample(cp, r):
  idx = []
  for i in range(len(r)):
    j = 0 
    while r[i] > cp[j] and j < len(cp):
      j += 1
    idx.append(j)
  return idx

names = ["fortran", "numpy", "python", "naive"]
T = dict(zip(names, [[] for name in names]))
funs = {"fortran": f.sample_from_cp, "numpy": numpy_sample, "python":
    python_sample, "naive": naive_sample}

for i in range(len(M)):
  r = np.random.rand(int(M[i]))
  
  for name in names:
    t1 = time.time()
    for j in range(L):
      idx = funs[name](cp, r)
    t2 = time.time()
    T[name].append((t2 - t1) / L)

for name in names:
  pl.loglog(M, T[name], label=name)
pl.legend()
pl.show()
