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

#Specify the binary system
m1 = 4e6*u.Msun

binaryA = binaries.CircularBinary(m1, m2 = 1e3*u.Msun)
binaryB = binaries.CircularBinary(m1, m2 = (10**(3.5))*u.Msun)
binaryC = binaries.CircularBinary(m1, m2 = 1e4*u.Msun)

r = np.geomspace(1, 20, 1000)*binaryA.r_isco

#Test
Ncount = 0
Nvals = [0]
_t = 0
r_i = 1e-5*u.pc
rvals = [r_i]
_r = r_i

#print(binaryC.T_orb(_r), binaryC.t_merge(_r))

while _r > binaryC.r_isco:
    _t += binaryC.T_orb(_r)
    _r = binaryC.r_of_t(_t, r_i)
    Ncount += 1
    
    rvals.append(_r)
    Nvals.append(Ncount)

rvals = np.array(rvals)
Nvals = np.array(Nvals)

plt.figure()

plt.plot(r/u.pc, binaryA.Norb_to_merger(r), label=r"$m_2 = 10^{3}\,M_\odot$")
plt.plot(r/u.pc, binaryB.Norb_to_merger(r), label=r"$m_2 = 10^{3.5}\,M_\odot$")
plt.plot(r/u.pc, binaryC.Norb_to_merger(r), label=r"$m_2 = 10^{4}\,M_\odot$")

plt.plot(rvals/u.pc, Nvals[::-1] - Nvals[0], color='C2', linestyle=':',lw=3,alpha=0.5)

plt.xlabel(r'$r$ [pc]')
plt.ylabel(r'$N_\mathrm{orb}$ to merger')

plt.xlim(1e-6, 2e-5)
plt.ylim(1, 1e7)

plt.xscale('log')
plt.yscale('log')

plt.yticks(np.geomspace(1, 1e7, 8))
plt.grid(axis='y')

plt.axvline(binaryA.r_isco, linestyle='--', color='grey')



plt.legend(loc='best')

plt.savefig("../plots/Norbs_to_merger.pdf", bbox_inches='tight')

plt.show()
