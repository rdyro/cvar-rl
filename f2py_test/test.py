import fortran
import numpy as np

x = np.array([1, 2, -3, 4])
def f(x): return x < 0
idx = fortran.find_first(lambda x: x < 0, x)
print(fortran.find_first.__doc__)
print(idx)
print(f(2.0))
print(f(-3.0))

"""
cp = np.cumsum(np.random.rand(5))
cp /= cp[-1]
print(cp)

a = np.random.rand(10, 2)
b = fortran.sample_from_cp(cp, a)

print(a)
print(b)


N = 10**np.arange(0, 5)
for i in range(
"""
