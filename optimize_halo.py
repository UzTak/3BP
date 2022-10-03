""" 
Based on the initial solution created, optimzie the Halo Orbit using a single-shooting method.
"""

import numpy as np
from propagate import propagate_cr3bp_with_STM, propagate_cr3bp_with_STM2

from single_shooting import single_shooting, ssdc
import copy

def optimize_halo(s0_, p0, mu, itermax=2000, tol=1e-5):
    """
    
    args: 
        s0 (np.array): state initial guess
        p0 (float): period initial guess
    """
    if np.abs(s0_[1]) > 1e-9:
        print("WARNING: initial state may not lie on xy-plane")

    s0 = copy.deepcopy(s0_)
    # initial tf and xi
    # Here, z-component is fixed -> xi = [x_0, ydot_0, p/2]
    tf = p0 / 2

    for i in range(itermax):
        # propagate half period
        (pos, vel, acc_f, stm_f) = propagate_cr3bp_with_STM(s0, mu, tf)
        # (pos, vel, acc_f, stm_f) = propagate_cr3bp_with_STM2(s0, mu, tf)

        # f_x = [y_(P/2), xdot_(P/2), zdot_(P/2)]; perturbation after the propagation
        f_x = np.array([pos[-1, 1], vel[-1, 0], vel[-1,2]])

        if np.linalg.norm(f_x) < tol:
            print("solution is converged.")
            s0[0], s0[4] = pos[0,0], vel[0,1]
            break

        else:

            # obtain Jacobian matrix (df)
            # df = [[STM_21, STM_25, ydot_(P/2)],[STM_41, STM_45, xddot_(P/2)],[STM_61, STM_65, zddot_(P/2)]]
            df = np.array([[stm_f[1,0], stm_f[1,4], vel[-1, 1]],
                           [stm_f[3,0], stm_f[3,4], acc_f[0]],
                           [stm_f[5,0], stm_f[5,4], acc_f[2]]])

            # perform one step of single shooting differential correction
            # xi_new = single_shooting(np.array([s0[0], s0[4], tf]), df, f_x)
            xi_new = ssdc(np.array([s0[0], s0[4], tf]), df, f_x)


            # update the guess of xi
            s0[0] = xi_new[0]
            s0[4] = xi_new[1]
            tf = xi_new[2]

            print(f"iter {i+1}; error = {f_x}")

        if i == itermax - 1:
            raise Exception("Solution didn't converge.")

    # optimized state, period
    return s0, xi_new[2] * 2
