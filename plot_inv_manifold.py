import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Lindstedt_Poincare import initial_guess_halo, find_lagrangian_pts
from optimize_halo import optimize_halo
from propagate import propagate_cr3bp, propagate_cr3bp_with_STM

def plot_inv_manifold(mu, s0, period, eps='fix', arc_num=15, t_prop=1e5):

    if eps == 'fix':
        eps = 1e-6

    # Plot the orbits
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # 1. create monodromy matrix (propagating STM)
    (_, _, _, monodromy) = propagate_cr3bp_with_STM(s0, mu, period)

    # 2. take eigensystem -> identify the unstable & stable eignevalues
    w, v = la.eig(monodromy)

    # vu = unstable eigenvector, vs = stable eigenvector

    vu = 1
    vs =1

    # 3. translate eigensystems -> make multiple starting point
    t_list = np.linspace(0, period, arc_num)

    for i in range(arc_num):
        t = t_list[i]
        (pos_t, vel_t, _, stm_t) = propagate_cr3bp_with_STM(s0, mu, t)
        vu_t = np.dot(stm_t, vu)
        vs_t = np.dot(stm_t, vs)

        xi = np.reshape([pos_t[-1,:], vel_t[-1,:]], (1,))

        # add perturbation to the initial state
        xui1 = xi + eps * vu_t / np.norm(vu_t)
        xui2 = xi - eps * vu_t / np.norm(vu_t)
        xsi1 = xi + eps * vs_t / np.norm(vs_t)
        xsi2 = xi - eps * vs_t / np.norm(vs_t)

        # 5. progagate the state with perturbation
        (pos_u1, vel_u1, _, _) = propagate_cr3bp_with_STM(xui1, mu, t_prop)
        (pos_u2, vel_u2, _, _) = propagate_cr3bp_with_STM(xui2, mu, t_prop)
        (pos_s1, vel_s1, _, _) = propagate_cr3bp_with_STM(xsi1, mu, t_prop)
        (pos_s2, vel_s2, _, _) = propagate_cr3bp_with_STM(xsi2, mu, t_prop)

        ax.plot3D(pos_u1[:,0], pos_u1[:,1], pos_u1[:,2], color='blue')

    return fig, ax