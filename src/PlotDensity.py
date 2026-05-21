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

import utilities
import binaries
from MCimri import *

#Specify the binary system
m1 = 4e6*u.Msun
m2 = (10**3.5)*u.Msun

binaryC = binaries.CircularBinary(m1, m2)
r_isco = binaryC.r_isco

a_i = 20*r_isco
SpikeDF = df.GeneralizedNFWSpike(m1, rho_6=1*u.Msun/u.pc**3, gamma_sp=7/3, r_t=20*a_i, alpha=2)

#a_i = 12.26*r_isco
rlist = np.loadtxt(f"../data/rlist_logM2_{np.log10(m2/u.Msun):.2f}.txt")
rho_ana = np.vectorize(SpikeDF.rho_ini)(rlist) #Analytic expression for the density

#rholist_i = reconstruct_density_full(rlist, Es_i, Ls_i, weights, m1) #Density reconstructed from orbits
rho_sim = np.loadtxt(f"../data/DensityRatio_logM2_{np.log10(m2/u.Msun):.2f}_Norb_950000.txt")
rho_sim_v2 = np.loadtxt(f"../data/DensityRatio_logM2_{np.log10(m2/u.Msun):.2f}_v2_Norb_950000.txt")

#---------------

plt.figure()

plt.semilogx(rlist/u.pc, rho_sim, label='v1')
plt.semilogx(rlist/u.pc, rho_sim_v2, label='v2')
plt.axhline(1.0, linestyle='-', color='grey', alpha=0.6, zorder=0)

#plt.axvline(a_i/u.pc, linestyle='--', color='grey')
#plt.axvline(r_orb/u.pc, linestyle='--', color='C1')

#plt.xlim(1e-9, 1e-5)
plt.xlim(1e-2*a_i/u.pc, 1e2*a_i/u.pc)
plt.ylim(0, 2)

plt.gca().axvspan(
    1e-15,               # sufficiently far left
    binaryC.r_isco/u.pc,
    facecolor='grey',
    alpha=0.3,
    hatch='///'
)
#plt.axvline(binaryC.r_isco/u.pc, linestyle='--', color='k')

plt.xscale('log')

plt.xlabel(r'$r$ [pc]')
plt.ylabel(r'$\rho_\mathrm{DM}(r)/\rho_i(r)$')

plt.legend(loc='best', fontsize=12)

#plt.text(0.04, 0.95, plotlabel, ha='left', va='top', fontsize=12, transform=plt.gca().transAxes)

m1str =  utilities.to_scientific(m1/u.Msun)
m2str = utilities.to_scientific(m2/u.Msun)

plt.title(r"$(m_1, m_2) = (" + m1str + ", " + m2str + ")\,M_\odot$")
plt.legend(loc='upper right')


#---------------------------



rho_sim_i   = np.loadtxt(f"../data/DensityRatio_logM2_{np.log10(m2/u.Msun):.2f}_v2_Norb_700000.txt")
rho_sim_end = np.loadtxt(f"../data/DensityRatio_logM2_{np.log10(m2/u.Msun):.2f}_v2_end.txt")

#---------------

plt.figure()

#plt.semilogx(rlist/u.pc, rho_sim_i, label='Initial')
plt.semilogx(rlist/u.pc, rho_sim_end, color='C3')
#plt.semilogx(rlist/u.pc, utilities.moving_avg(rho_sim_end, 10), alpha=0.7)
#plt.semilogx(rlist/u.pc, rho_sim_end/rho_sim_i, label='Ratio')
plt.axhline(1.0, linestyle='-', color='grey', alpha=0.6, zorder=0)

#plt.axvline(a_i/u.pc, linestyle='--', color='grey')
#plt.axvline(r_orb/u.pc, linestyle='--', color='C1')

#plt.xlim(1e-9, 1e-5)
#plt.xlim(1e-2*a_i/u.pc, 1e2*a_i/u.pc)
plt.xlim(1e-6, 1.5e-5)
plt.ylim(0.3, 1.5)

plt.gca().axvspan(
    1e-15,               # sufficiently far left
    binaryC.r_isco/u.pc,
    facecolor='grey',
    alpha=0.3,
    hatch='///'
)
#plt.axvline(binaryC.r_isco/u.pc, linestyle='--', color='k')

plt.xscale('log')

plt.xlabel(r'$r$ [pc]')
plt.ylabel(r'$\rho_\mathrm{DM}(r)/\rho_i(r)$')

#plt.legend(loc='best', fontsize=12)

#plt.text(0.04, 0.95, plotlabel, ha='left', va='top', fontsize=12, transform=plt.gca().transAxes)

m1str =  utilities.to_scientific(m1/u.Msun)
m2str = utilities.to_scientific(m2/u.Msun)

plt.title(r"$(m_1, m_2) = (" + m1str + ", " + m2str + ")\,M_\odot$")
plt.legend(loc='upper right')

plt.savefig("../plots/FinalDensity_logM2_3.5.pdf", bbox_inches='tight')

plt.show()