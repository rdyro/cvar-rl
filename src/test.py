import env
import time
import numpy as np

d = env.Drone2D()
d.render()
d.state[5] = 0.0
d.state[4] = np.pi / 2

for i in range(2000):
  a = [0.5, 0.5]
  #(ns, _) = d.next_state_sample(d.state, a) 
  #d.state = ns.reshape(-1)
  (ns, r, done, _) = d.step(a)
  print(done)
  print(ns)
  if done:
    break
  d.render()
  time.sleep(1.0 / 30.0)
