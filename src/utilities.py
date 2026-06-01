import numpy as np
from numpy import cos, sin

from tqdm import tqdm

from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from NbodyIMRI import distributionfunctions as df
from NbodyIMRI import units as u
from NbodyIMRI import tools

import binaries

import math

def to_scientific(x, decimal_places=1):
    """
    Convert a floating-point number to LaTeX scientific notation.

    Parameters
    ----------
    x : float
        The number to format.
    decimal_places : int, optional
        Number of decimal places in the significand.

    Returns
    -------
    str
        A LaTeX-formatted scientific notation string.

    Examples
    --------
    >>> to_scientific_latex(310000)
    '$3.1 \\\\times 10^{5}$'

    >>> to_scientific_latex(0.0004567, decimal_places=3)
    '$4.567 \\\\times 10^{-4}$'
    """

    if x == 0:
        return f"$0.{'0' * decimal_places} \\times 10^{{0}}$"

    exponent = int(math.floor(math.log10(abs(x))))
    significand = x / (10 ** exponent)

    significand_str = f"{significand:.{decimal_places}f}"

    return f"{significand_str} \\times 10^{{{exponent}}}"
    
def moving_avg(x, N):
    return np.convolve(x, np.ones(N)/N, mode='same')
    
import numpy as np

def block_avg(x, y, N):
    """
    Average every N entries of x and y.

    Parameters
    ----------
    x, y : array-like
        Input arrays of equal length.
    N : int
        Block size for averaging.

    Returns
    -------
    x_new, y_new : np.ndarray
        Downsampled arrays.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Trim arrays so length is divisible by N
    n_trim = len(x) // N * N

    x_trim = x[:n_trim]
    y_trim = y[:n_trim]

    # Reshape into blocks of size N and average
    x_new = x_trim.reshape(-1, N).mean(axis=1)
    y_new = y_trim.reshape(-1, N).mean(axis=1)

    return x_new, y_new
    
def density_avg(x, y, N):
    """
    Average every N entries of x and y.

    Parameters
    ----------
    x, y : array-like
        Input arrays of equal length.
    N : int
        Block size for averaging.

    Returns
    -------
    x_new, y_new : np.ndarray
        Downsampled arrays.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Trim arrays so length is divisible by N
    n_trim = len(x) // N * N

    x_trim = x[:n_trim]
    y_trim = y[:n_trim]

    # Reshape into blocks of size N and average
    x_new = x_trim.reshape(-1, N).mean(axis=1)
    N_blocks = len(x_new)
    
    y_new = 0.0*x_new
    for i in range(N_blocks):
        _x = x_trim[(i*N):((i+1)*N)]
        _y = y_trim[(i*N):((i+1)*N)]
        assert len(_x) == N
        
        y_new[i] = np.trapz(_y*_x**2, _x)/np.trapz(_x**2, _x)
    #y_new = y_trim.reshape(-1, N).mean(axis=1)

    return x_new, y_new
    
def normalize(Ax, Ay, Az):
    norm = np.sqrt(Ax**2 + Ay**2 + Az**2)
    return Ax/norm, Ay/norm, Az/norm
    
    
def full_cross(Ax, Ay, Az, Bx, By, Bz, normed=False):
    Cx = +(Ay*Bz - Az*By)
    Cy = -(Ax*Bz - Az*Bx)
    Cz = +(Ax*By - Ay*Bx)
    
    if (normed):
        Cx, Cy, Cz = normalize(Cx, Cy, Cz)
    return Cx, Cy, Cz
    
#Return a random direction, defined by:
#   - theta, the polar angle
#   - phi, the azimuthal angle
def get_random_direction(N=1):
    theta = np.arccos(2*np.random.rand(N)-1)
    phi = 2*np.pi*np.random.rand(N)
    return theta, phi
    
    