"""
Initial Guess generation based on Lindstedt-Poincare method (3rd order approximation)
Reference:
Richardson, D.,
"Analytic Construction of Periodic Orbits about the Collinear Points," 1979

[Key Implementation]
In the original paper, n1 (mean motion) and gamma_L "absorbs" the normalization of
(un-normalized) a1 [km] (mean distance of M1 & M2).
To make the algorithm compatible to many other 3BP (which normalize a1 as well), we set a1 = 1, so that n1 = 1
(i.e., canonical form).
"""

import numpy as np
from scipy import optimize
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def find_lagrangian_pts(mu):
    """
    Find 5 Lagrangian points in the synodic rotating frame
    :param mu: M1/(M1+M2)
    :return: cooridnates of L1 ~ L5
    """

    # collinear points
    # Lagrange point is a equilibrium of energy gradient
    # i.e., find the points s.t. dU/dx = 0

    def fun(x):  # dU/dx
        return -(1-mu)/(np.abs(x+mu))**3 * (x+mu) - mu / (np.abs(x+mu-1))**3 * (x+mu-1) + x

    # location of M1 (minor body)
    l = 1 - mu

    sol1 = optimize.root(fun, 0.95 * l, method='hybr')  # L1
    l1 = np.array([sol1.x[0], 0, 0])

    sol2 = optimize.root(fun, 1.03 * l, method='hybr')  # L2
    l2 = np.array([sol2.x[0], 0, 0])

    sol3 = optimize.root(fun, -l, method='hybr')  # L3
    l3 = np.array([sol3.x[0], 0, 0])

    # equilateral points
    # L4
    l4 = np.array([np.cos(np.pi / 3) - mu, np.sin(np.pi / 3), 0])
    # L5
    l5 = np.array([np.cos(np.pi / 3) - mu, -np.sin(np.pi / 3), 0])

    print(
        f"L1 = {l1[0]:.4f}, L2 = {l2[0]:.4f}, L3 = {l3[0]:.4f}"
    )

    return l1, l2, l3, l4, l5


def get_lambda(c2):
    """
    Root-find the 4th order equation about lambda:
    lambda^4 + (c2-2) * lambda^2 - (c2-1)*(1+2*c2) = 0
    :param c2: param defined in R3BP's EoM
    :return: lambda
    """

    bb = c2 - 2
    cc = -(c2 - 1)*(1 + 2 * c2)

    lam_sqr = 1/2 * (-bb + np.sqrt(bb**2 - 4 * cc))
    lam = np.sqrt(lam_sqr)

    print(f"lam = {lam:.4f}")

    if not np.isreal(lam):
        raise ValueError("Generated linearized frequency (lambda) is not real.")

    return lam


def get_c(gl, lp, mu):
    """
    obtain coefficients c2, c3, c4
    :param gl: gamma_L
    :param lp: Lagrange point (L1, L2 L3)
    :param mu: ratio M1/(M1+M2), M1 = minor body, M2 = major body
    :return: c2, c3, c4
    """

    # obtain c2, c3, and c4
    if lp == 1:
        c2 = 1 / (gl**3) * (1**2 * mu + (-1)**2 * (1 - mu) * gl**3 / (1 - gl)**3)
        c3 = 1 / (gl**3) * (1**3 * mu + (-1)**3 * (1 - mu) * gl**4 / (1 - gl)**4)
        c4 = 1 / (gl**3) * (1**4 * mu + (-1)**4 * (1 - mu) * gl**5 / (1 - gl)**5)
    elif lp == 2:
        c2 = 1 / (gl**3) * ((-1)**2 * mu + (-1)**2 * (1 - mu) * gl**3 / (1 + gl)**3)
        c3 = 1 / (gl**3) * ((-1)**3 * mu + (-1)**3 * (1 - mu) * gl**4 / (1 + gl)**4)
        c4 = 1 / (gl**3) * ((-1)**4 * mu + (-1)**4 * (1 - mu) * gl**5 / (1 + gl)**5)
    elif lp == 3:
        c2 = 1 / (gl ** 3) * (1 - mu + mu * gl ** 3 / (1 + gl) ** 3)
        c3 = 1 / (gl ** 3) * (1 - mu + mu * gl ** 4 / (1 + gl) ** 4)
        c4 = 1 / (gl ** 3) * (1 - mu + mu * gl ** 5 / (1 + gl) ** 5)

    print(
        f"c2 = {c2:.3f}, c3 = {c3:.3f}, c4 = {c4:.3f}"
    )

    return c2, c3, c4


def get_coeff(lam, c2, c3, c4, k):
    # d1 and d2
    d1 = 3 * lam ** 2 / k * (k * (6 * lam ** 2 - 1) - 2 * lam)
    d2 = 8 * lam ** 2 / k * (k * (11 * lam ** 2 - 1) - 2 * lam)

    # X_2x coefficients
    a21 = 3 * c3 * (k ** 2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = -3 * c3 * lam / (4 * k * d1) * (3 * k ** 3 * lam - 6 * k * (k - lam) + 4)
    a24 = -3 * c3 * lam / (4 * k * d1) * (2 + 3 * k * lam)
    b21 = -3 * c3 * lam / (2 * d1) * (3 * k * lam - 4)
    b22 = 3 * c3 * lam / d1
    d21 = - c3 / (2 * lam ** 2)

    # X_3x coefficients
    a31 = -9 * lam / (4 * d2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)) \
          + (9 * lam ** 2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k ** 2))
    a32 = -1 / d2 * ((9 * lam / 4 * (4 * c3 * (k * a24 - b22) + k * c4))
                     + 3 / 2 * (9 * lam ** 2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4))
    b31 = 3 / (8 * d2) * (8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k ** 2))
                          + (9 * lam ** 2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)))
    b32 = 1 / d2 * (9 * lam * (c3 * (k * b22 + d21 - 2 * a24) - c4)
                    + 3 / 8 * (9 * lam ** 2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4))
    d31 = 3 / (64 * lam ** 2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lam ** 2) * (4 * c3 * (a23 - d21) + c4 * (4 + k ** 2))

    # Amplitude-constraint relationship to get l1 & l2
    a1 = -3 / 2 * c3 * (2 * a21 + a23 + 5 * d21) - 3 / 8 * c4 * (12 - k ** 2)
    a2 = 3 / 2 * c3 * (a24 - 2 * a22) + 9 / 8 * c4

    s1 = (3 / 2 * c3 * (2 * a21 * (k ** 2 - 2) - a23 * (k ** 2 + 2) - 2 * k * b21)
          - 3 / 8 * c4 * (3 * k ** 4 - 8 * k ** 2 + 8)) / (2 * lam * (lam * (1 + k ** 2) - 2 * k))
    s2 = (3 / 2 * c3 * (2 * a22 * (k ** 2 - 2) + a24 * (k ** 2 + 2) + 2 * k * b22 + 5 * d21)
          + 3 / 8 * c4 * (12 - k ** 2)) / (2 * lam * (lam * (1 + k ** 2) - 2 * k))

    l1 = a1 + 2 * lam ** 2 * s1
    l2 = a2 + 2 * lam ** 2 * s2

    return (d1, d2,
            a21, a22, a23, a24, b21, b22, d21,
            a31, a32, b31, b32, d31, d32,
            a1, a2,
            s1, s2,
            l1, l2)


def initial_guess_halo(lp, fam, mu, Az_km, lstar):
    """
    input:
        lp (= 1,2,3): Lagrange Point (L1, L2, L3)
        fam (= 1,3): choosing North (1) or South (3) Halo orbit
        mu (float): M1/(M1 + M2), where M1 is the minor body
        Az (float): z-direction amplitude [km] (>= 0)
        lstar (float): # canonical length (Earth - Moon distance, etc.) [km]
    reutrn:
        s0 (np.array): initial state guess (based on the analytical method)
        period (float): period of the Halo orbit
        s0_ini (np.array): initial state vector of the Halo orbit (maintain the symmetry of the Halo oribt)
    """

    phi = 0  # phase delay of x and y component. psi = phi + n * pi /2

    # obtain the lagrangian points in synodic rotating frame
    l1,l2,l3,_,_ = find_lagrangian_pts(mu)

    l_s = np.zeros((1,3))
    if lp == 1:
        l_s = l1
    elif lp == 2:
        l_s = l2
    elif lp == 3:
        l_s = l3

    # get gammaL, normalizing term of Length
    if lp == 1:
        gl = np.abs((1 - mu) - l1[0])
    elif lp == 2:
        gl = np.abs((1 - mu) - l2[0])
    elif lp == 3:
        gl = np.abs(mu - l3[0])
    else:
        raise Exception('lp has to be 1, 2, or 3')

    # normalize Az_km based on r1 = 1 (L1,L2) or r2 = 1 (L3)
    # Note lstar * gl = a1 * (r1(2) / a1) = r1(2)
    Az = Az_km / (lstar * gl)

    # get constants & parameters
    c2, c3, c4 = get_c(gl, lp, mu)
    lam = get_lambda(c2)
    Delta = lam**2 - c2
    k = (lam**2 + 1 + 2*c2) / (2 * lam)

    (d1, d2,
     a21, a22, a23, a24, b21, b22, d21,
     a31, a32, b31, b32, d31, d32,
     a1, a2,
     s1, s2,
     l1, l2) = get_coeff(lam, c2, c3, c4, k)

    # Obtain Ax based on the amplitude relationship
    Ax = np.sqrt(np.abs(-(l2 * Az**2 + Delta) / l1))

    # minimum Ax permissible (when Az = 0)
    Ax_min = np.sqrt(np.abs(Delta/l1))
    if Ax < Ax_min:
        raise ValueError("Error happened when obtaining Ax.")

    # frequency correction term. omega1 = 0
    # cf. omega is NOT a frequency.
    omega2 = s1 * Ax**2 + s2 * Az**2
    omega = 1 + omega2

    # true period [s], (NOT non-dimensional time)
    period = 2 * np.pi / (lam * omega)

    tau = np.linspace(0, omega * period, 400)

    # switching function delta_n
    d_n = 2 - fam

    # create state vectors
    x = np.zeros(len(tau))
    y = np.zeros(len(tau))
    z = np.zeros(len(tau))
    xdot = np.zeros(len(tau))
    ydot = np.zeros(len(tau))
    zdot = np.zeros(len(tau))

    for i in range(len(tau)):
        tau1 = lam * tau[i] + phi

        # location
        x[i] = a21 * Ax ** 2 + a22 * Az ** 2 \
                - Ax * np.cos(tau1) \
                + (a23 * Ax ** 2 - a24 * Az ** 2) * np.cos(2 * tau1) \
                + (a31 * Ax ** 3 - a32 * Ax * Az ** 2) * np.cos(3 * tau1)

        y[i] = k * Ax * np.sin(tau1) \
                + (b21 * Ax ** 2 - b22 * Az ** 2) * np.sin(2 * tau1) \
                + (b31 * Ax ** 3 - b32 * Ax * Az ** 2) * np.sin(3 * tau1)

        z[i] = d_n * Az * np.cos(tau1) \
                + d_n * d21 * Ax * Az * (np.cos(2 * tau1 - 3)) \
                + d_n * (d32 * Az * Ax ** 2 - d31 * Az ** 3) * np.cos(3 * tau1)

        # velocity
        xdot[i] = lam * Ax * np.sin(tau1) \
                   - 2 * lam * (a23 * Ax ** 2 - a24 * Az ** 2) * np.sin(2 * tau1) \
                   - 3 * lam * (a31 * Ax ** 3 - a32 * Ax * Az ** 2) * np.sin(3 * tau1)

        ydot[i] = lam * k * Ax * np.cos(tau1) \
                   + 2 * lam * (b21 * Ax ** 2 - b22 * Az ** 2) * np.cos(2 * tau1) \
                   + 3 * lam * (b31 * Ax ** 3 - b32 * Ax * Az ** 2) * np.cos(3 * tau1)

        zdot[i] = - lam * d_n * Ax * np.sin(tau1) \
                   - 2 * lam * d_n * d21 * Ax * Az * (np.sin(2 * tau1 - 3)) \
                   - 3 * lam * d_n * (d32 * Az * Ax ** 2 - d31 * Az ** 3) * np.sin(3 * tau1)

    # bring the state vectors (in r1(2) = 1 normalization) back in the synodic frame (a1 = 1 normalization)
    # for x-coordinates, consider the location of L-points as well

    x_s = gl * x + l_s[0]
    y_s = gl * y
    z_s = gl * z
    xdot_s = gl * xdot
    ydot_s = gl * ydot
    zdot_s = gl * zdot

    s0 = np.array([x_s, y_s, z_s, xdot_s, ydot_s, zdot_s])
    s0_ini = np.array([x_s[0], 0.0, z_s[0], 0.0, ydot_s[0], 0.0])
    return s0, period, s0_ini

