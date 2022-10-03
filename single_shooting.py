"""
One step of Single Shooting Differntial Correction based on the Newton Method 
"""

import numpy as np


def single_shooting(x, df, f_x):
    # reshape the input 
    x = np.reshape(x, (len(x),))
    f_x = np.reshape(f_x, (len(f_x),))
    
    xnew = x - np.dot(np.linalg.pinv(df), f_x)
    xnew = np.reshape(xnew, (len(xnew),))
    return xnew


# Yuri's
def ssdc(xi, df, ferr):
    """Apply single-shooting differential correction

    Args:
        xi (np.array): length-n array of free variables
        ferr (np.array): length n array of residuals
        df (np.array): n by n np.array of Jacobian
    Returns:
        (np.array): length-n array of new free variables
    """
    xi_vert = np.reshape(xi, (len(xi), -1))
    ferr_vert = np.reshape(ferr, (len(ferr), -1))
    xii_vert = xi_vert - np.dot(np.linalg.pinv(df), ferr_vert)
    xii = np.reshape(xii_vert, (len(xii_vert),))
    return xii
