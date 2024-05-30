"""
Generate the Halo orbit based on Shooting method
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Lindstedt_Poincare import initial_guess_halo, find_lagrangian_pts
from optimize_halo import optimize_halo
from propagate import propagate_cr3bp, propagate_cr3bp_with_STM
from plot_inv_manifold import plot_inv_manifold




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

# set-up of problem
mu = 1.215058560962404E-2
Az_km = 6000
lstar = 389703
fam = 3  # north
lp = 2

# initial guess of Halo Orbit: Lindstedt-Poincare Method
s0, period0, s0_ini = initial_guess_halo(lp, fam, mu, Az_km, lstar)
# Optimization of the trajectory
sopt_ini, period_opt = optimize_halo(s0_ini, period0, mu)

# Generate the Trajectory with the optimized variables
(pos, vel, acc_f, stm_f) = propagate_cr3bp_with_STM(sopt_ini, mu, period_opt)

# store cartesian state
# state0 = s0[:,0].T
# (pos_,vel_, acc_f_, stm_f_) = propagate_cr3bp_with_STM(state0, mu, period0/2)



# find the fixed points
l1, l2, l3, l4, l5 = find_lagrangian_pts(mu)

# Plot the orbits
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = plt.axes()


# plot invariant manifolds
fig, ax = plot_inv_manifold(fig, ax, mu, sopt_ini, period_opt, eps='fix', arc_num=20, t_prop=10, d=3)


# ax.plot3D(s0[0,:], s0[1,:], s0[2,:], color='gray', label='initial guess')
# ax.scatter(sopt_ini[0], sopt_ini[1], sopt_ini[2], color='orange')
# ax.scatter(s0_ini[0], s0_ini[1], s0_ini[2], color='red')
# ax.plot3D(pos[:,0], pos[:,1], pos[:,2], color='blue', label='accurate')
# ax.plot3D(pos_[:,0], pos_[:,1], pos_[:,2], color='purple', label='test')
# ax.scatter(l2[0], l2[1], l2[2], marker='*', color='orange', label='L2')
ax.scatter(l2[0], l2[1], marker='d', color='orange', s=20, label='L2')
# set_axes_equal(ax)
ax.axis("equal")
ax.grid(visible=True, which='both')




ax.set_xlabel('x, LU')
ax.set_ylabel('y, LU')
# ax.set_zlabel('z')
plt.legend()
fig.tight_layout()
plt.show()
