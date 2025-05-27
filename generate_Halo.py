import requests
import json
import pandas as pd
from propagate import propagate_cr3bp, propagate_cr3bp_with_STM
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from Lindstedt_Poincare import initial_guess_halo, find_lagrangian_pts


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plt_sphere(list_center, list_radius):
  for c, r in zip(list_center, list_radius):
    ax = fig.gca()

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)

    ax.plot_surface(x+c[0], y+c[1], z+c[2], color=np.random.choice(['orange']), alpha=1)


# define the animation function
def update(frame, x,y,z):
    # update the data of the line object with the current position of the spacecraft
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    return line,


scale = 1
R_L = 1737.4 * scale  # km, Moon radius
LU  = 389703.264829278 * scale  # km
TU  = 382981.289129055  # s

steps = 10000

f = open('data_L2S.json',)
data = json.load(f)
# print(type(data["data"]))

df = pd.DataFrame(data["data"])
df = df.astype(float)
mu = float(data["system"]["mass_ratio"])
fig = plt.figure()
ax = plt.axes(projection='3d')

# e.g., 9:2 NRHO, 6.5 days
tof = 13.3 * (24*60*60 / TU)
row = df.loc[(df.iloc[:,7]-tof).abs().argsort()[0]]
tof = row[7]
s0  = row[0:6]
pos, vel = propagate_cr3bp(s0, mu, tof, step=steps)
ax.plot(pos[:,0]*LU - (1-mu)*LU, pos[:,1]*LU, pos[:,2]*LU, label="9:2 resonance")
# create a line object to store the orbit path

# pos, vel = propagate_cr3bp(s0, mu, tof, step=501)
state = np.hstack((pos, vel))
tvec = np.linspace(0, tof, steps)


# find the fixed points
l1, l2, l3, l4, l5 = find_lagrangian_pts(mu)
ax.scatter(l1[0]*LU - (1-mu)*LU, l1[1], l1[2], marker='D', s=25.0, facecolors="none", color='black', label='L1')
ax.scatter(l2[0]*LU - (1-mu)*LU, l2[1], l2[2], marker='D', s=25.0, facecolors="none", color='red', label='L2')
# ax.scatter(1-mu, 0.0, 0.0, marker='*', s=25.0, color='orange', label='Moon')

plt_sphere([(0.0, 0.0, 0.0)], [R_L])

ax.set_xlabel(r"$x, \times 10^4$ km")
ax.set_ylabel(r"$y, \times 10^4$ km")
ax.set_zlabel(r"$z, \times 10^4$ km")
plt.legend()
ax.view_init(elev=30, azim=225)
set_axes_equal(ax)
plt.show()

# save the state history 
data = {
    "mu"   : mu,
    "LU"   : LU,
    "TU"   : TU,
    "t"    : tvec.tolist(),
    "state": state.tolist(),
}


filename = '13_3day_S_Halo.json'
with open(filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)
    

# writergif = PillowWriter(fps=30)
# ani.save("NRHO_EMrot.gif", writer=writergif)