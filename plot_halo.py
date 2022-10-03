"""
Generate the Halo orbit based on Shooting method
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Lindstedt_Poincare import initial_guess_halo, find_lagrangian_pts
from optimize_halo import optimize_halo
from propagate import propagate_cr3bp, propagate_cr3bp_with_STM


# set-up of problem
mu = 3.04036e-6
Az_km = 200000
lstar = 1.49598e8
fam = 1
lp = 2

# initial guess of Halo Orbit: Lindstedt-Poincare Method
s0, period0, s0_ini = initial_guess_halo(lp, fam, mu, Az_km, lstar)
# Optimization of the trajectory
sopt_ini, period_opt = optimize_halo(s0_ini, period0, mu)

# Generate the Trajectory with the optimized variables

# store cartesian state
# state0 = s0[:,0].T
# (pos_,vel_, acc_f_, stm_f_) = propagate_cr3bp_with_STM(state0, mu, period0/2)

(pos, vel, acc_f, stm_f) = propagate_cr3bp_with_STM(sopt_ini, mu, period_opt)

l1, l2, l3, l4, l5 = find_lagrangian_pts(mu)


# Plot the orbits
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(s0[0,:], s0[1,:], s0[2,:], color='gray', label='initial guess')
# ax.scatter(sopt_ini[0], sopt_ini[1], sopt_ini[2], color='orange')
# ax.scatter(s0_ini[0], s0_ini[1], s0_ini[2], color='red')
ax.plot3D(pos[:,0], pos[:,1], pos[:,2], color='blue', label='accurate')
# ax.plot3D(pos_[:,0], pos_[:,1], pos_[:,2], color='purple', label='test')
ax.scatter(l2[0], l2[1], l2[2], marker='*', color='orange', label='L2')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()
