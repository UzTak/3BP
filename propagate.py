"""
Integaration based on scipy.integrate.odeint
"""


import numpy as np
from scipy.integrate import odeint
from ode_cr3bp import ode_cr3bp_with_STM, ode_cr3bp
from _rhs_functions import rhs_cr3bp, rhs_cr3bp_with_STM


def propagate_cr3bp_with_STM(s0, mu, tf, step=3000):

    tvec = np.linspace(0, tf, step)
    state0 = np.zeros((42,))
    # store cartesian state
    state0[:6] = s0
    # store initial identity STM into extended state vector
    state0[6] = 1
    state0[13] = 1
    state0[20] = 1
    state0[27] = 1
    state0[34] = 1
    state0[41] = 1

    sol, infodict = odeint(
        func=ode_cr3bp_with_STM,
        y0=state0,
        t=tvec,
        args=(mu,),
        full_output=True,
        rtol=1e-12, atol=1e-12,
        tfirst=True,
    )
    
    if infodict["message"] != "Integration successful.":
        print(f"WARNING: It looks like the integration was not successful")

    # time history of state
    pos = sol[:, 0:3]
    vel = sol[:, 3:6]

    # stm at the last step
    stm_f = sol[-1, 6:]
    stm_f = np.reshape(stm_f, (6, 6))

    # acceleration at the last step
    state_f = np.array([sol[-1,0], sol[-1,1], sol[-1,2], sol[-1,3], sol[-1,4], sol[-1,5]])
    ds_f = ode_cr3bp(tvec[-1], state_f, mu)
    acc_f = np.array([ds_f[3], ds_f[4], ds_f[5]])

    return pos, vel, acc_f, stm_f


def propagate_cr3bp_with_STM2(s0, mu, tf, step=3000):
    tvec = np.linspace(0, tf, step)
    state0 = np.zeros((42,))
    # store cartesian state
    state0[:6] = s0
    # store initial identity STM into extended state vector
    state0[6] = 1
    state0[13] = 1
    state0[20] = 1
    state0[27] = 1
    state0[34] = 1
    state0[41] = 1

    sol, infodict = odeint(
        func=rhs_cr3bp_with_STM,
        y0=state0,
        t=tvec,
        args=(mu,),
        full_output=True,
        rtol=1e-12, atol=1e-12,
        tfirst=True,
    )

    if infodict["message"] != "Integration successful.":
        print(f"WARNING: It looks like the integration was not successful")

    # time history of state
    pos = sol[:, 0:3]
    vel = sol[:, 3:6]

    # stm at the last step
    stm_f = sol[-1, 6:]
    stm_f = np.reshape(stm_f, (6, 6))

    # acceleration at the last step
    state_f = np.array([sol[-1, 0], sol[-1, 1], sol[-1, 2], sol[-1, 3], sol[-1, 4], sol[-1, 5]])
    ds_f = ode_cr3bp(tvec[-1], state_f, mu)
    acc_f = np.array([ds_f[3], ds_f[4], ds_f[5]])

    return pos, vel, acc_f, stm_f


def propagate_cr3bp(s0, mu, tf, step=3000):
    tvec = np.linspace(0, tf, step)

    sol, infodict = odeint(
        func=ode_cr3bp,
        y0=s0,
        t=tvec,
        args=(mu,),
        full_output=True,
        rtol=1e-12, atol=1e-12,
        tfirst=True,
    )

    if infodict["message"] != "Integration successful.":
        print(f"WARNING: It looks like the integration was not successful")

    # time history of state
    pos = sol[:, 0:3]
    vel = sol[:, 3:6]

    return pos, vel