import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Lindstedt_Poincare import initial_guess_halo, find_lagrangian_pts
from optimize_halo import optimize_halo
from propagate import propagate_cr3bp, propagate_cr3bp_with_STM

def plot_inv_manifold(fig, ax, mu, s0, period, eps='fix', arc_num=10, t_prop=100):

    if eps == 'fix':
        eps = 1e-6

    # 1. create monodromy matrix (propagating STM)
    (_, _, _, monodromy) = propagate_cr3bp_with_STM(s0, mu, period)

    # 2. take eigensystem -> identify the unstable & stable eignevalues
    w, v = la.eig(monodromy)

    # vu = unstable eigenvector, vs = stable eigenvector
    for j in range(len(w)):
        if w[j].real > 1 and w[j].imag == 0:
            wu = w[j].real
            vu = v[:,j].real
        if w[j].real < 1 and w[j].imag == 0:
            ws = w[j].real
            vs = v[:,j].real

    # 3. translate eigensystems -> make multiple starting point
    t_list = np.linspace(0, period, arc_num)

    for i in range(arc_num):
        t = t_list[i]
        (pos_t, vel_t, _, stm_t) = propagate_cr3bp_with_STM(s0, mu, t, step=round(t/0.01)+2)
        vu_t = np.dot(stm_t, vu)
        vs_t = np.dot(stm_t, vs)

        xi = np.reshape([pos_t[-1,:], vel_t[-1,:]], (-1))

        # add perturbation to the initial state
        xui1 = xi + eps * vu_t / np.linalg.norm(vu_t)
        xui2 = xi - eps * vu_t / np.linalg.norm(vu_t)
        xsi1 = xi + eps * vs_t / np.linalg.norm(vs_t)
        xsi2 = xi - eps * vs_t / np.linalg.norm(vs_t)

        # 5. progagate the state with perturbation
        (pos_u1, vel_u1) = propagate_cr3bp(xui1, mu, t_prop, step=100)
        (pos_u2, vel_u2) = propagate_cr3bp(xui2, mu, t_prop, step=100)
        (pos_s1, vel_s1) = propagate_cr3bp(xsi1, mu, -t_prop, step=100)
        (pos_s2, vel_s2) = propagate_cr3bp(xsi2, mu, -t_prop, step=100)

        ax.plot3D(pos_u1[:,0], pos_u1[:,1], pos_u1[:,2], color='brown', linewidth=1)
        ax.plot3D(pos_u2[:,0], pos_u2[:,1], pos_u2[:,2], color='red', linewidth=1)
        ax.plot3D(pos_s1[:,0], pos_s1[:,1], pos_s1[:,2], color='lime', linewidth=1)
        ax.plot3D(pos_s2[:,0], pos_s2[:,1], pos_s2[:,2], color='green', linewidth=1)
        # ax.scatter(pos_s1[0,0], pos_s1[0,1], pos_s1[0,2], color='lime', marker="x")
        # ax.scatter(pos_s2[0,0], pos_s2[0,1], pos_s2[0,2], color='green', marker="x")

    return fig, ax