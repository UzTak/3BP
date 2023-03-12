import requests
import json
import pandas as pd
from propagate import propagate_cr3bp, propagate_cr3bp_with_STM
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from Lindstedt_Poincare import initial_guess_halo, find_lagrangian_pts




f = open('data.json',)
data = json.load(f)
# print(type(data["data"]))

df = pd.DataFrame(data["data"])
df = df.astype(float)

tof = 3.1
row = df.loc[(df.iloc[:,7]-tof).abs().argsort()[0]]

s0 = row[0:6]
mu = float(data["system"]["mass_ratio"])

print(s0)
pos, vel = propagate_cr3bp(s0, mu, tof, step=3000)

# Plot the orbits
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(pos[:,0], pos[:,1], pos[:,2], label="sol")


# find the fixed points
l1, l2, l3, l4, l5 = find_lagrangian_pts(mu)
ax.scatter(l1[0], l1[1], l1[2], marker='D', s=25.0, facecolors="none", color='black', label='L1')
ax.scatter(1-mu, 0.0, 0.0, marker='*', s=25.0, color='orange', label='Moon')

plt.legend()
plt.show()