import numpy as np
from numpy import cos, sin

#from tqdm import tqdm
tqdm = lambda x: x


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

from pathlib import Path
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-rank", type=int)
parser.add_argument("-logm2", type=float, default = 4)
parser.add_argument("-xM", type=float, default = 100)

args = parser.parse_args()
#m1 = args.m1*u.Msun
#m2 = args.m2*u.Msun

rank = int(args.rank) + 48
print("rank:", rank)

#Specify the binary system
m1 = 4e6*u.Msun
m2 = (10**args.logm2)*u.Msun
a_over_rM    = args.xM
N_particles = 2000
dN = 1

SGSK_MODE = False
SAVE_ORBITS = False

a_over_risco = a_over_rM/6

binaryC = binaries.CircularBinary(m1, m2)
r_isco = binaryC.r_isco

a_i = a_over_risco*r_isco

#ID = tools.generate_hash()

dN_str = ""
if (dN > 1):
    dN_str = f"dN_{str(int(dN))}_"

fstr = f"logM2_{np.log10(m2/u.Msun):.2f}_NDM_{str(int(N_particles))}_rM_{str(int(np.round(a_over_rM)))}_{dN_str}" + str(int(rank))
datapath = "../data/" + fstr + "/"
plotpath = "../plots/" + fstr + "/"

Path(datapath).mkdir(parents=True, exist_ok=True)
Path(plotpath).mkdir(parents=True, exist_ok=True)

def make_plot(N):
    
    rholist_full = orbits.reconstruct_density_full(rlist, Es, Ls, weights, m1)
    
    plt.figure()

    plt.semilogx(rlist/u.pc, rholist_full/rholist_i, alpha=0.5)
    x_new, y_new = utilities.block_avg(rlist/u.pc,rholist_full/rholist_i, 10)
    plt.semilogx(x_new, y_new, label='MC Reconstruction')
    
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
    plt.title(r"$(m_1, m_2) = (" + m1str + ", " + m2str + ")\,M_\odot$; " + str(int(N)) + " orbits")
    plt.legend(loc='upper right')

    plt.savefig(plotpath + f"InspiralDepletion_{fstr}_Norb_{int(N)}.pdf", bbox_inches='tight')

def save_density(N):
    rholist_full = orbits.reconstruct_density_full(rlist, Es, Ls, weights, m1)
    hdrtxt = "Columns: r [pc], rho [Msun/pc**3], rho/rho_i"
    np.savetxt(datapath + f"Density_{fstr}_Norb_" + str(int(N)) + ".txt", np.column_stack((rlist/u.pc, rholist_full/(u.Msun/u.pc**3), rholist_full/rholist_i)), header=hdrtxt)

def save_orbits(N):
    hdrtxt = "Columns: E [(km/s)^2], L [pc (km/s)], Lz [pc (km/s)], w [Msun]"
    outdata = np.column_stack((Es/(u.km/u.s)**2, Ls/(u.pc*u.km/u.s), Lz/(u.pc*u.km/u.s), weights/u.Msun))
    np.savetxt(datapath + f"Orbits_{fstr}_Norb_" + str(int(N)) + ".txt", outdata, header=hdrtxt)


#####################
##### FLAGS #########
#####################
INCLUDE_DF = True
INCLUDE_3BODY = True
DYNAMIC = True

#Define the spike and sample the energies of N_particles particles (or rather, orbits), from P(E) = g(E)*d(E)
#---------------------------
SpikeDF = df.GeneralizedNFWSpike(m1, rho_6=1*u.Msun/u.pc**3, gamma_sp=7/3, r_t=20*a_i, alpha=2)
r_min = 0.1*r_isco
r_max = 10000*a_i

if (SGSK_MODE):
    ecc = 0.67
    gamma = 7/3
    
    _u = np.random.rand(N_particles)
    #rp_min = r_isco
    #rp_max = 330*r_isco
    
    rp_min = r_min
    rp_max = r_max
    rps = rp_min*(rp_max/rp_min)**_u

    Es_i = np.ones(N_particles)*(u.G_N*m1)*(1-ecc)/(2*rps)
    L_circ = u.G_N*m1/np.sqrt(2*Es_i)
    Ls_i = L_circ*np.sqrt(1 - ecc**2)
    
    weights = rps**(-(gamma - 3))
    
else:
    Es_i = SpikeDF.draw_E(r_max=r_max, N = N_particles)
    L_circ = u.G_N*m1/np.sqrt(2*Es_i)
    Ls_i = L_circ*np.sqrt(np.random.rand(N_particles))

Lz_i = Ls_i*(2*np.random.rand(N_particles) - 1)

m_part = SpikeDF.M_DM_ini(r_max)/N_particles
weights = m_part*np.ones(N_particles)


#Define a grid for radii and calculate densities
#----------------------------------------------
rlist = np.geomspace(0.1*r_isco, 1000*a_i, 1000)
rho_ana = np.vectorize(SpikeDF.rho_ini)(rlist) #Analytic expression for the density
rholist_i = orbits.reconstruct_density_full(rlist, Es_i, Ls_i, weights, m1) #Density reconstructed from orbits
#np.savetxt(f"../data/rlist_{fstr}.txt", rlist)

#Simulate over Norb orbits and reconstruct density
#---------------------------------------------
N_to_merge = binaryC.Norb_to_merger(a_i)
print("Number of orbits to merger:", N_to_merge)

Norb = int(1.1*N_to_merge)
offset = 0

N_out = 50_000

Es = 1.0*Es_i
Ls = 1.0*Ls_i
Lz = 1.0*Lz_i

r_orb = 1.0*a_i
t = 0

for i in tqdm(range(Norb)):
    
    if (i%N_out == 0):
        print(i, r_orb/a_i)
        save_density(i+offset)
        if (SAVE_ORBITS): save_orbits(i+offset)
        make_plot(i+offset)
    
    L_circ = u.G_N*m1/np.sqrt(2*np.abs(Es))
    inds = (Es > 0) & (Ls < L_circ)
    L_out = ((Es > 0) & (Ls > L_circ))
    if (np.sum(L_out) > 0):
        if (i%100 == 0):
            print(np.sum(L_out))
            
    if (i%dN == 0):
        dE, dL, dLz = MCimri.calculate_dE(Es[inds], Ls[inds], Lz[inds], binaryC, r_orb, mult = dN, include_DF=INCLUDE_DF, include_3body=INCLUDE_3BODY)
    
        Es[inds] += dE
        Ls[inds] += dL
        Lz[inds] += dLz
        
    if (DYNAMIC):
        t += binaryC.T_orb(r_orb)
        r_orb = binaryC.r_of_t(t, a_i)
        
    if (r_orb < r_isco):
        save_density(i+offset)
        if (SAVE_ORBITS): save_orbits(i+offset)
        make_plot(i+offset)
        break

#plt.show()
#----------------

