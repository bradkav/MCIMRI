import numpy as np
from numpy import cos, sin

from tqdm import tqdm

from scipy.interpolate import interp1d

try:
    from numpy import trapz
except:
    from numpy import trapezoid as trapz

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from NbodyIMRI import distributionfunctions as df
from NbodyIMRI import units as u
from NbodyIMRI import tools

import binaries
import orbits

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
        
        y_new[i] = trapz(_y*_x**2, _x)/trapz(_x**2, _x)
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
    
    
class TimeSeries:
    def __init__(self, m1):
        self.m1 = m1
        
        self.N          = None
        self.E_tot      = None
        self.E_tot_free = None
        self.rho_r      = None
        self.rho_r_free = None
        
    def calc_E_tot(self, E, w):
        return np.sum(E*w)
        
    def calc_rho_r(self, r, E, L, w):
        return orbits.reconstruct_density_full(np.array([r,]), E, L, w, self.m1)[0]

    def add(self, N, Es, Ls, w, w_c, r):
        
        Etot = self.calc_E_tot(Es, w)
        Etot_free = self.calc_E_tot(Es, w_c)
        
        rho_r = self.calc_rho_r(r, Es, Ls, w)
        rho_r_free = self.calc_rho_r(r, Es, Ls, w_c)
        
        if (self.N is None):
            self.N = [N]
            self.E_tot = [Etot]
            self.E_tot_free = [Etot_free]
        
            self.rho_r = [rho_r]
            self.rho_r_free = [rho_r_free]
        else:
            self.N.append(N)
            self.E_tot.append(Etot)
            self.E_tot_free.append(Etot_free)
        
            self.rho_r.append(rho_r)
            self.rho_r_free.append(rho_r_free)
        
    def save(self, datapath, fstr):
        N_arr = np.array(self.N)
        E_arr = np.array(self.E_tot)
        E_free_arr = np.array(self.E_tot_free)
        rho_r_arr = np.array(self.rho_r)
        rho_r_free_arr = np.array(self.rho_r_free)
        
        hdrtxt = "Columns: N_orbs, Total DM energy (no capture) [Msun (km/s)^2], Total DM energy (including capture) [Msun (km/s)^2]"
        np.savetxt(datapath + f"Etot_{fstr}.txt.gz", np.column_stack((N_arr, E_arr/(u.Msun*(u.km/u.s)**2), E_free_arr/(u.Msun*(u.km/u.s)**2))), header=hdrtxt, fmt='%.5e')
        
        hdrtxt = "Columns: N_orbs, rho(r_2), no capture [Msun/pc**3], rho(r_2), including capture [Msun/pc**3]"
        np.savetxt(datapath + f"Density_r_{fstr}.txt.gz", np.column_stack((N_arr, rho_r_arr/(u.Msun/u.pc**3), rho_r_free_arr/(u.Msun/u.pc**3))), header=hdrtxt, fmt='%.5e')


