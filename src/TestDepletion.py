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
import MCimri
import orbits

#Specify the binary system
#m1 = 10.0*u.Msun
#m2 = 0.15*u.Msun


#Specify the binary system
m1 = 4e6*u.Msun
m2 = (10**3.5)*u.Msun
a_over_risco = 12.26

binaryC = binaries.CircularBinary(m1, m2)
r_isco = binaryC.r_isco

a_i = a_over_risco*r_isco

q  = m2/m1

#####################
##### FLAGS #########
#####################
INCLUDE_DF = True
INCLUDE_3BODY = True
DYNAMIC = False

#Define the spike and sample the energies of N_particles particles (or rather, orbits), from P(E) = g(E)*d(E)
#---------------------------
N_particles = 10000
SpikeDF = df.GeneralizedNFWSpike(m1, rho_6=1*u.Msun/u.pc**3, gamma_sp=7/3, r_t=20*a_i, alpha=2)
r_min = 0.1*r_isco
r_max = 10000*a_i

#Es_i = SpikeDF.draw_E(r_max=r_max, N = N_particles)
Es_i, weights = orbits.draw_biased_E(SpikeDF, r_min, r_max, r0 = a_i, N = N_particles)

L_circ = u.G_N*m1/np.sqrt(2*Es_i)


Ls_i = L_circ*(np.random.rand(N_particles))**0.5


Lz_i = Ls_i*(2*np.random.rand(N_particles) - 1)


m_part = SpikeDF.M_DM_ini(r_max)/N_particles

#weights = m_part*np.ones(N_particles)
weights *= SpikeDF.M_DM_ini(r_max)/np.sum(weights)
#print(np.sum(weights)/SpikeDF.M_DM_ini(r_max))

#Define a grid for radii and calculate densities
#----------------------------------------------
rlist = np.geomspace(0.1*r_isco, 1000*a_i, 1000)
rho_ana = np.vectorize(SpikeDF.rho_ini)(rlist) #Analytic expression for the density
rholist_i = orbits.reconstruct_density_full(rlist, Es_i, Ls_i, weights, m1) #Density reconstructed from orbits

#----------------------------------------
plt.figure()

plt.semilogx(rlist/u.pc, rholist_i/rho_ana, label='MC Reconstruction')

x_new, y_new = utilities.block_avg(rlist/u.pc,rholist_i/rho_ana, 10)
plt.semilogx(x_new, y_new)

plt.axhline(1.0, linestyle='-', color='grey', alpha=0.6, zorder=0)

plt.axvline(a_i/u.pc, linestyle='--', color='grey')
#plt.axvline(r_orb/u.pc, linestyle='--', color='C1')

#plt.xlim(1e-9, 1e-5)
plt.xlim(1e-2*r_isco/u.pc, 1.5*np.max(rlist)/u.pc)
plt.ylim(0, 2)

plt.gca().axvspan(1e-15, binaryC.r_isco/u.pc, facecolor='grey', alpha=0.3, hatch='///')
#plt.axvline(binaryC.r_isco/u.pc, linestyle='--', color='k')

plt.xscale('log')

plt.xlabel(r'$r$ [pc]')
plt.ylabel(r'$\rho_\mathrm{DM}(r)/\rho_i(r)$')


#Simulate over Norb orbits and reconstruct density
#---------------------------------------------
Norb = 1000

dE, dL, dLz = MCimri.calculate_dE(Es_i, Ls_i, Lz_i, binaryC, a_i, include_DF=True, include_3body=False)

Es = 1.0*Es_i
Ls = 1.0*Ls_i
Lz = 1.0*Lz_i

r_orb = 1.0*a_i
t = 0

dN = 1

for i in tqdm(range(Norb)):
    L_circ = u.G_N*m1/np.sqrt(2*np.abs(Es))
    inds = (Es > 0) & (Ls < L_circ)
    
    
    L_out = ((Es > 0) & (Ls > L_circ))
    if (np.sum(L_out) > 0):
        if (i%100 == 0):
            print(np.sum(L_out))
            
    if (DYNAMIC):
        t += binaryC.T_orb(r_orb)
        r_orb = binaryC.r_of_t(t, a_i)
            
    if (i%dN == 0):
        dE, dL, dLz = MCimri.calculate_dE(Es[inds], Ls[inds], Lz[inds], binaryC, r_orb, mult = dN, include_DF=INCLUDE_DF, include_3body=INCLUDE_3BODY)
    
        Es[inds] += dE
        Ls[inds] += dL
        Lz[inds] += dLz
    
    if (r_orb < r_isco):
        break

rholist_full = orbits.reconstruct_density_full(rlist, Es, Ls, weights, m1)

plt.figure()

plt.semilogx(rlist/u.pc, rholist_full/rho_ana, label='MC Reconstruction')
x_new, y_new = utilities.block_avg(rlist/u.pc,rholist_full/rho_ana, 10)
plt.semilogx(x_new, y_new)


plt.semilogx(rlist/u.pc, rholist_full/rholist_i, label='MC Reconstruction', color='grey')
x_new, y_new = utilities.block_avg(rlist/u.pc,rholist_full/rholist_i, 10)
plt.semilogx(x_new, y_new, color='k')

plt.axhline(1.0, linestyle='-', color='grey', alpha=0.6, zorder=0)

plt.axvline(a_i/u.pc, linestyle='--', color='grey')
plt.axvline(r_orb/u.pc, linestyle='--', color='C1')

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

plt.legend(loc='upper right', fontsize=12)

#plt.text(0.04, 0.95, plotlabel, ha='left', va='top', fontsize=12, transform=plt.gca().transAxes)

m1str =  utilities.to_scientific(m1/u.Msun)
m2str = utilities.to_scientific(m2/u.Msun)
plt.title(r"$(m_1, m_2) = (" + m1str + ", " + m2str + ")\,M_\odot$; " + str(int(Norb)) + " orbits")
plt.legend(loc='upper right')

#plt.savefig("../plots/InspiralDepletion.pdf", bbox_inches='tight')

plt.show()