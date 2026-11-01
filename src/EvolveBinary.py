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


# Physical Parameters
# -------------------
rho_6 = 5.45e15*u.Msun/u.pc**3
gamma_sp = 7/3

#Minimum impact parameter
b_min = 0.0

x_trunc = 20 #x_trunc = r_t/a_i

# Simulation Parameters
# ---------------------
Norb = -1 #Fix Norb = -1 for running to merger (fixed output)

#Options for running to merger
Nout_min = 10_000
Nsnaps_target = 200

# Flags
# -----
DYNAMIC = True
SAVE_ALL_ORBITS = False
MAKE_PLOTS = False
SAVE_PROFILES = True


#####################


#Command line arguments
#----------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-rank", type=int)
parser.add_argument("-logm1", type=float, default = np.log10(4e6))
parser.add_argument("-logm2", type=float, default = 4)
parser.add_argument("-mode", type=str, default="Full")
parser.add_argument("-IDstr", type=str, default="")
parser.add_argument("-dN", type=int, default = 1)
parser.add_argument("-N_particles", type=int, default = 2000)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-rS', type = float)
group.add_argument('-pc', type = float)


args = parser.parse_args()

rank = int(args.rank)
print("rank:", rank)

dN = args.dN
N_particles = args.N_particles

#Flags
#-------------------------
mode = args.mode

INCLUDE_DF = True
INCLUDE_3BODY = True
EJECTING_ONLY = False

if (mode == "EjOnly"):
    EJECTING_ONLY = True
elif (mode == "NoStir"):
    INCLUDE_3BODY = False
elif (mode == "NoInner"):
    MCimri.DO_INNER = False

#Specify the binary system
#-------------------------
m1 = (10**args.logm1)*u.Msun
m2 = (10**args.logm2)*u.Msun

q = m2/m1

binary = binaries.CircularBinary(m1, m2)
r_isco = binary.r_isco

rS = r_isco/3

if (args.rS is not None):
    a_i = args.rS*rS
    rstr = f"r_{str(int(np.round(args.rS)))}_rS"
else:
    a_i = args.pc*u.pc
    rstr = f"r_{args.pc:.2e}_pc"

L_lc = 4*u.G_N*m1/u.c


# Set up file system
#-------------------

dN_str = ""
if (dN > 1):
    dN_str = f"_dN_{str(int(dN))}"

IDstr = args.IDstr
if (IDstr):
    if (IDstr[0] != "_"):
        IDstr = "_" + IDstr

froot = f"logM1_{np.log10(m1/u.Msun):.2f}_logM2_{np.log10(m2/u.Msun):.2f}_{mode}_{rstr}{dN_str}{IDstr}"
fstr = froot + "_" + str(int(rank))
datapath = "../data/" + froot + "/" + fstr + "/"
plotpath = "../plots/" + froot + "/" + fstr + "/"

Path(datapath).mkdir(parents=True, exist_ok=True)
if (MAKE_PLOTS):
    Path(plotpath).mkdir(parents=True, exist_ok=True)

def make_plot(N):

    if not (MAKE_PLOTS == True):
        return -1
    
    rholist_full = orbits.reconstruct_density_full(rlist, Es, Ls, weights, m1)
    rholist_full_free = orbits.reconstruct_density_full(rlist, Es, Ls, weights_lc, m1)
    
    plt.figure()

    with np.errstate(divide='ignore', invalid='ignore'):
        plt.semilogx(rlist/u.pc, rholist_full/rholist_i, alpha=0.5)
        x_new, y_new = utilities.block_avg(rlist/u.pc,rholist_full/rholist_i, 10)
        plt.semilogx(x_new, y_new, label='MC Reconstruction')
    
        plt.semilogx(rlist/u.pc, rholist_full_free/rholist_i_free, alpha=0.5)
        x_new, y_new = utilities.block_avg(rlist/u.pc,rholist_full_free/rholist_i_free, 10)
        plt.semilogx(x_new, y_new, label='MC (uncaptured)')
    
    plt.axhline(1.0, linestyle='-', color='grey', alpha=0.6, zorder=0)

    plt.axvline(a_i/u.pc, linestyle='--', color='grey')
    plt.axvline(r_orb/u.pc, linestyle='--', color='C1')

    #plt.xlim(1e-9, 1e-5)
    plt.xlim(1e-2*a_i/u.pc, 1e2*a_i/u.pc)
    plt.ylim(0, 2)

    plt.gca().axvspan(
        1e-15,               # sufficiently far left
        binary.r_isco/u.pc,
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
    rholist_full_free = orbits.reconstruct_density_full(rlist, Es, Ls, weights_lc, m1)
    hdrtxt = "Columns: r [pc], rho [Msun/pc**3], rho/rho_i, rho (uncaptured) [Msun/pc**3], rho (uncaptured)/rho_i"
    
    with np.errstate(divide='ignore', invalid='ignore'):
        np.savetxt(datapath + f"Density_{fstr}_Norb_" + str(int(N)) + ".txt.gz", np.column_stack((rlist/u.pc, rholist_full/(u.Msun/u.pc**3), rholist_full/rholist_i, rholist_full_free/(u.Msun/u.pc**3), rholist_full_free/rholist_i_free)), header=hdrtxt, fmt='%.5e')


def save_orbits(N):
    hdrtxt = "Columns: E [(km/s)^2], L [pc (km/s)], Lz [pc (km/s)], w [Msun], w (uncaptured) [Msun]"
    outdata = np.column_stack((Es/(u.km/u.s)**2, Ls/(u.pc*u.km/u.s), Lz/(u.pc*u.km/u.s), weights/u.Msun, weights_lc/u.Msun))
    np.savetxt(datapath + f"Orbits_{fstr}_Norb_" + str(int(N)) + ".txt.gz", outdata, header=hdrtxt)    




#Define the spike and sample the energies of N_particles particles (or rather, orbits), from P(E) = g(E)*d(E)
#---------------------------
SpikeDF = df.GeneralizedNFWSpike(m1, rho_6=rho_6, gamma_sp=gamma_sp, r_t=x_trunc*a_i, alpha=2)

r_min = 1*rS
r_max = 10000*a_i

#Draw (E, L, Lz)
Es_i = SpikeDF.draw_E(r_max=r_max, N = N_particles)
L_circ = u.G_N*m1/np.sqrt(2*Es_i)
Ls_i = L_circ*np.sqrt(np.random.rand(N_particles))
Lz_i = Ls_i*(2*np.random.rand(N_particles) - 1)

#Fix weights
m_part = SpikeDF.M_DM_ini(r_max)/N_particles
weights = m_part*np.ones(N_particles)

#Set weights of initially captured particles to zero
weights_lc = 1.0*weights
L_cut = L_lc*np.sqrt(np.clip(1 - 4*Es_i/u.c**2, 0, None))
inds = (Ls_i < L_cut) | (Es_i > u.G_N*m1/(2*rS))
weights_lc[inds] *= 0

#Define a grid for radii and calculate densities
#----------------------------------------------
rlist = np.geomspace(1*rS, 1000*a_i, 1000)
rho_ana = np.vectorize(SpikeDF.rho_ini)(rlist) #Analytic expression for the density
rholist_i = orbits.reconstruct_density_full(rlist, Es_i, Ls_i, weights, m1) #Density reconstructed from orbits
rholist_i_free = orbits.reconstruct_density_full(rlist, Es_i, Ls_i, weights_lc, m1)


#Simulate over Norb orbits and reconstruct density
#---------------------------------------------
N_to_merge = binary.Norb_to_merger(a_i)
print("Number of orbits to merger:", N_to_merge)

if (Norb < 0):
    DYNAMIC_OUTSTEPS = False
    Norb = int(1.1*N_to_merge)
    N_out = int(max(Nout_min, ((Norb/Nsnaps_target) // Nout_min) * Nout_min))
else:
    DYNAMIC_OUTSTEPS = True
    N_out = 1
        

Es = 1.0*Es_i
Ls = 1.0*Ls_i
Lz = 1.0*Lz_i

r_orb = 1.0*a_i
t = 0

E_uncap = 0.0

ts = utilities.TimeSeries(binary.m1)

#Save initial orbits
save_orbits(0)

for i in tqdm(range(Norb)):

    if (DYNAMIC_OUTSTEPS):
        if (i > 20):
            N_out = 10
        if (i > 200):
            N_out = 100
        if (i > 2000):
            N_out = 1000
        if (i > 20_000):
            N_out = 10_000
        if (i > 200_000):
            N_out = 100_000

    N_remain = N_to_merge - i
    if (N_remain < 50_000):
        dN = 1
        N_out = 5_000
    if (N_remain < 10_000):
        N_out = 1_000
    if (N_remain < 1_000):
        N_out = 100
    if (N_remain < 100):
        N_out = 10
    if (N_remain < 10):
        N_out = 1
            
    if (i%25_000 == 0):
        print("Number of orbits completed:", i)

    if (i%N_out == 0):
        #print("Number of orbits, r/r_i:", i, r_orb/a_i)
        
        ts.add(i, Es, Ls, weights, weights_lc, r_orb, E_uncap)

        if ((i == 0) or (SAVE_PROFILES == True)):
            save_density(i)
        make_plot(i)
        if (SAVE_ALL_ORBITS): save_orbits(i)
    
    L_circ = u.G_N*m1/np.sqrt(2*np.abs(Es))
    inds = (Es > 0) & (Ls < L_circ)
    L_out = ((Es > 0) & (Ls > L_circ))
    if (np.sum(L_out) > 0):
        if (i%100 == 0):
            print(np.sum(L_out))
            
    if (i%dN == 0):
        dE, dL, dLz = MCimri.calculate_dE(Es[inds], Ls[inds], Lz[inds], binary, r_orb, mult = dN, include_DF=INCLUDE_DF, include_3body=INCLUDE_3BODY, b_min = b_min)

        _dE  = np.nan_to_num(dE)
        _dL  = np.nan_to_num(dL)
        _dLz = np.nan_to_num(dLz)

        E_uncap += np.sum(_dE*weights_lc[inds])
        
        # Work on subset
        Es_sub = Es[inds]
        Ls_sub = Ls[inds]
        Lz_sub = Lz[inds]

        # Trial values
        Es_trial = Es_sub + _dE

        if (EJECTING_ONLY):
            mask = Es_trial < 0
        else:
            mask = np.abs(Es_trial) >= 0
        #print("Ejected particles:", np.sum(mask)/N_particles)

        if (np.sum(mask) > 0):
            # Apply updates ONLY inside subset
            Es_sub[mask] += _dE[mask]
            Ls_sub[mask] += _dL[mask]
            Lz_sub[mask] += _dLz[mask]

            # Write back once
            Es[inds] = Es_sub
            Ls[inds] = Ls_sub
            Lz[inds] = Lz_sub
        
    #Set weights of captured particles to zero
    L_cut = L_lc*np.sqrt(np.clip(1 - 4*Es/u.c**2, 0, None))
    inds = (Ls < L_cut) | (Es > u.G_N*m1/(2*rS))
    weights_lc[inds] *= 0.0
        
    if (DYNAMIC):
        t += binary.T_orb(r_orb)
        r_orb = binary.r_of_t(t, a_i)
        
    if (r_orb < r_isco):
        
        ts.add(i, Es, Ls, weights, weights_lc, r_orb, E_uncap)
        
        save_density(i)
        make_plot(i)
        if (SAVE_ALL_ORBITS): save_orbits(i)
        save_orbits(-1)
        
        ts.save(datapath, fstr)
        
        break

if (r_orb > r_isco):
    ts.add(i, Es, Ls, weights, weights_lc, r_orb, E_uncap)

    save_density(i)
    make_plot(i)
    if (SAVE_ALL_ORBITS): save_orbits(i)
    save_orbits(-1)
    
    ts.save(datapath, fstr)
    
#plt.show()
#----------------

