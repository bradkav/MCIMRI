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


class CircularBinary:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        self.M  = m1 + m2
        
        self.Mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
        self.aV = (u.c**3/(np.pi*u.G_N*self.Mc))/16
        
        self.r_s = 2*u.G_N*m1/u.c**2
        self.r_isco = 3*self.r_s
        
    def t_merge(self, r):
        t_merge_circ = (5/256)*u.c**5*r**4/(u.G_N**3*self.m1*self.m2*self.M)
        return t_merge_circ
        
    def T_orb(self, r):
        return (2*np.pi)*np.sqrt(r**3/(u.G_N*self.M))
        
    def f_of_r(self, r):
        return 2/self.T_orb(r)
        
    def r_of_t(self, t, r0):
        return r0*(1 - t/self.t_merge(r0))**(1/4)
        
    def Norb_to_r(self, r, r0):
        return (u.c**5/(64*np.pi*u.G_N**(5/2)))*(r0**(5/2) - r**(5/2))/(self.m1*self.m2*np.sqrt(self.M))
        
    def Norb_to_merger(self, r0):
        return self.Norb_to_r(self.r_isco, r0)
        
    def v_orb(self, r):
        return np.sqrt(u.G_N*self.M/r)
        