"""
Equation of Motion of CR3BP with State Transition Matrix 
"""

import numpy as np


def ode_cr3bp_with_STM(t, s, mu):
    
    ds = np.zeros((42,))  # 6 states + 6*6 entries of STM

    r1 = np.sqrt((s[0] + mu) ** 2 + s[1] ** 2 + s[2] ** 2)
    r2 = np.sqrt((s[0] + mu - 1) ** 2 + s[1] ** 2 + s[2] ** 2)

    # position
    ds[0] = s[3]
    ds[1] = s[4]
    ds[2] = s[5]
    # velocity
    ds[3] = - (1 - mu) * (s[0] + mu) / r1 ** 3 \
            - mu * (s[0] + mu - 1) / r2 ** 3 \
            + s[0] \
            + 2 * s[4]
    ds[4] = - (1 - mu) * s[1] / r1 ** 3 \
            - mu * s[1] / r2 ** 3 \
            + s[1] \
            - 2 * s[3]
    ds[5] = - (1 - mu) * s[2] / r1 ** 3 \
            - mu * s[2] / r2 ** 3

    # 2nd order derivative of U = (1 - mu)/r1 + mu/r2 + 1/2 (x**2 + y**2)
    dU_dxx = 1 \
            - (1 - mu) * (1 / r1**3 - 3 * (s[0] + mu)**2 / r1**5) \
            - mu * (1 / r2**3 - 3 * (s[0] - (1 - mu))**2 / r2**5)

    dU_dyy = 1 \
            - (1 - mu) * (1 / r1**3 - 3 * s[1]**2 / r1**5) \
            - mu * (1 / r2**3 - 3 * s[1]**2 / r2**5)

    dU_dzz = - (1 - mu) * (1 / r1**3 - 3 * s[2]**2 / r1**5) \
            - mu * (1 / r2**3 - 3 * s[2]**2 / r2**5)

    dU_dxy = 3 * (1 - mu) * (s[0] + mu) * s[1] / r1**5 \
             + 3 * mu * (s[0] + mu - 1) * s[1] / r2**5

    dU_dxz = 3 * (1 - mu) * (s[0] + mu) * s[2] / r1**5 \
             + 3 * mu * (s[0] + mu - 1) * s[2] / r2**5

    dU_dyz = 3 * (1 - mu) * s[1] * s[2] / r1**5 \
             + 3 * mu * s[1] * s[2] / r2**5

    # define A matrix

    a00, a01, a02, a03, a04, a05 = 0, 0, 0, 1, 0, 0
    a10, a11, a12, a13, a14, a15 = 0, 0, 0, 0, 1, 0
    a20, a21, a22, a23, a24, a25 = 0, 0, 0, 0, 0, 1
    a30, a31, a32, a33, a34, a35 = dU_dxx, dU_dxy, dU_dxz, 0, 2, 0
    a40, a41, a42, a43, a44, a45 = dU_dxy, dU_dyy, dU_dyz, -2, 0, 0
    a50, a51, a52, a53, a54, a55 = dU_dxz, dU_dyz, dU_dzz, 0, 0, 0

    
    # derivatives of STM
    # 1st row 
    ds[6]  = a00 * s[6]  + a01 * s[12] + a02 * s[18] + a03 * s[24] + a04 * s[30] + a05 * s[36]
    ds[7]  = a00 * s[7]  + a01 * s[13] + a02 * s[19] + a03 * s[25] + a04 * s[31] + a05 * s[37]
    ds[8]  = a00 * s[8]  + a01 * s[14] + a02 * s[20] + a03 * s[26] + a04 * s[32] + a05 * s[38]
    ds[9]  = a00 * s[9]  + a01 * s[15] + a02 * s[21] + a03 * s[27] + a04 * s[33] + a05 * s[39]
    ds[10] = a00 * s[10] + a01 * s[16] + a02 * s[22] + a03 * s[28] + a04 * s[34] + a05 * s[40]
    ds[11] = a00 * s[11] + a01 * s[17] + a02 * s[23] + a03 * s[29] + a04 * s[35] + a05 * s[41]
    
    # 2nd row
    ds[12] = a10 * s[6]  + a11 * s[12] + a12 * s[18] + a13 * s[24] + a14 * s[30] + a15 * s[36]
    ds[13] = a10 * s[7]  + a11 * s[13] + a12 * s[19] + a13 * s[25] + a14 * s[31] + a15 * s[37]
    ds[14] = a10 * s[8]  + a11 * s[14] + a12 * s[20] + a13 * s[26] + a14 * s[32] + a15 * s[38]
    ds[15] = a10 * s[9]  + a11 * s[15] + a12 * s[21] + a13 * s[27] + a14 * s[33] + a15 * s[39]
    ds[16] = a10 * s[10] + a11 * s[16] + a12 * s[22] + a13 * s[28] + a14 * s[34] + a15 * s[40]
    ds[17] = a10 * s[11] + a11 * s[17] + a12 * s[23] + a13 * s[29] + a14 * s[35] + a15 * s[41]
    
    # 3rd row
    ds[18] = a20 * s[6]  + a21 * s[12] + a22 * s[18] + a23 * s[24] + a24 * s[30] + a25 * s[36]
    ds[19] = a20 * s[7]  + a21 * s[13] + a22 * s[19] + a23 * s[25] + a24 * s[31] + a25 * s[37]
    ds[20] = a20 * s[8]  + a21 * s[14] + a22 * s[20] + a23 * s[26] + a24 * s[32] + a25 * s[38]
    ds[21] = a20 * s[9]  + a21 * s[15] + a22 * s[21] + a23 * s[27] + a24 * s[33] + a25 * s[39]
    ds[22] = a20 * s[10] + a21 * s[16] + a22 * s[22] + a23 * s[28] + a24 * s[34] + a25 * s[40]
    ds[23] = a20 * s[11] + a21 * s[17] + a22 * s[23] + a23 * s[29] + a24 * s[35] + a25 * s[41]

    # 4th row
    ds[24] = a30 * s[6]  + a31 * s[12] + a32 * s[18] + a33 * s[24] + a34 * s[30] + a35 * s[36]
    ds[25] = a30 * s[7]  + a31 * s[13] + a32 * s[19] + a33 * s[25] + a34 * s[31] + a35 * s[37]
    ds[26] = a30 * s[8]  + a31 * s[14] + a32 * s[20] + a33 * s[26] + a34 * s[32] + a35 * s[38]
    ds[27] = a30 * s[9]  + a31 * s[15] + a32 * s[21] + a33 * s[27] + a34 * s[33] + a35 * s[39]
    ds[28] = a30 * s[10] + a31 * s[16] + a32 * s[22] + a33 * s[28] + a34 * s[34] + a35 * s[40]
    ds[29] = a30 * s[11] + a31 * s[17] + a32 * s[23] + a33 * s[29] + a34 * s[35] + a35 * s[41]
    
    # 5th row
    ds[30] = a40 * s[6]  + a41 * s[12] + a42 * s[18] + a43 * s[24] + a44 * s[30] + a45 * s[36]
    ds[31] = a40 * s[7]  + a41 * s[13] + a42 * s[19] + a43 * s[25] + a44 * s[31] + a45 * s[37]
    ds[32] = a40 * s[8]  + a41 * s[14] + a42 * s[20] + a43 * s[26] + a44 * s[32] + a45 * s[38]
    ds[33] = a40 * s[9]  + a41 * s[15] + a42 * s[21] + a43 * s[27] + a44 * s[33] + a45 * s[39]
    ds[34] = a40 * s[10] + a41 * s[16] + a42 * s[22] + a43 * s[28] + a44 * s[34] + a45 * s[40]
    ds[35] = a40 * s[11] + a41 * s[17] + a42 * s[23] + a43 * s[29] + a44 * s[35] + a45 * s[41]
    
    # 6th row
    ds[36] = a50 * s[6]  + a51 * s[12] + a52 * s[18] + a53 * s[24] + a54 * s[30] + a55 * s[36]
    ds[37] = a50 * s[7]  + a51 * s[13] + a52 * s[19] + a53 * s[25] + a54 * s[31] + a55 * s[37]
    ds[38] = a50 * s[8]  + a51 * s[14] + a52 * s[20] + a53 * s[26] + a54 * s[32] + a55 * s[38]
    ds[39] = a50 * s[9]  + a51 * s[15] + a52 * s[21] + a53 * s[27] + a54 * s[33] + a55 * s[39]
    ds[40] = a50 * s[10] + a51 * s[16] + a52 * s[22] + a53 * s[28] + a54 * s[34] + a55 * s[40]
    ds[41] = a50 * s[11] + a51 * s[17] + a52 * s[23] + a53 * s[29] + a54 * s[35] + a55 * s[41]
    
    return ds


def ode_cr3bp(t, s, mu):
    ds = np.zeros((6,))  # 6 states

    r1 = np.sqrt((s[0] + mu)**2 + s[1]**2 + s[2]**2)
    r2 = np.sqrt((s[0] + mu - 1)**2 + s[1]**2 + s[2]**2)

    # position
    ds[0] = s[3]
    ds[1] = s[4]
    ds[2] = s[5]
    # velocity
    ds[3] = - (1 - mu) * (s[0] + mu) / r1 ** 3 \
            - mu * (s[0] + mu - 1) / r2 ** 3 \
            + s[0] \
            + 2 * s[4]
    ds[4] = - (1 - mu) * s[1] / r1 ** 3 \
            - mu * s[1] / r2 ** 3 \
            + s[1] \
            - 2 * s[3]
    ds[5] = - (1 - mu) * s[2] / r1 ** 3 \
            - mu * s[2] / r2 ** 3

    return ds
