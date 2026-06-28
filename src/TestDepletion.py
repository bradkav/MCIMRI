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

from pathlib import Path

import utilities
import binaries
import MCimri
import orbits

#Specify the binary system
#m1 = 10.0*u.Msun
#m2 = 0.15*u.Msun
#m1 = 7.46*u.Msun
#m2 = 0.18*u.Msun
#rsun = 695700*u.km
#a_i = 2.54*rsun
#a_i = 1e-7*u.pc



#Specify the binary system
m1 = 4e6*u.Msun
m2 = (10**4)*u.Msun
a_over_risco = 12.26

binaryC = binaries.CircularBinary(m1, m2)
r_isco = binaryC.r_isco

a_i = a_over_risco*r_isco

q  = m2/m1

#IDstr = "SystemA_res"
#IDstr = "M1_10_M2_0.15_doinner_Nout_50"
IDstr = "Test"
datapath = "../data/" + IDstr + "/"
plotpath = "../plots/" + IDstr + "/"

MAKE_PLOTS = False

Path(datapath).mkdir(parents=True, exist_ok=True)
Path(plotpath).mkdir(parents=True, exist_ok=True)

#####################
##### FLAGS #########
#####################
INCLUDE_DF = True
INCLUDE_3BODY = True
DYNAMIC = False

#Define the spike and sample the energies of N_particles particles (or rather, orbits), from P(E) = g(E)*d(E)
#---------------------------
N_particles = 25000
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

thetas = np.linspace(0, np.pi, 201)

rho_th_i = orbits.reconstruct_density_full_th(a_i, thetas, Es_i, Ls_i, Lz_i, weights, m1)

plt.figure()
plt.plot(thetas, rho_th_i)

def make_plot(outstr):
    
    outstr = str(outstr)
    
    rholist_full = orbits.reconstruct_density_full(rlist, Es, Ls, weights, m1)
    
    plt.figure()

    plt.semilogx(rlist/u.pc, rholist_full/rholist_i, alpha=0.5)
    x_new, y_new = utilities.density_avg(rlist/u.pc,rholist_full/rholist_i, 10)
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
    plt.title(r"$(m_1, m_2) = (" + m1str + ", " + m2str + ")\,M_\odot$; " + outstr + " orbits")
    plt.legend(loc='upper right')

    plt.savefig(plotpath + f"Depletion_{IDstr}_Norb_{outstr}.pdf", bbox_inches='tight')
    plt.close()

#Test
#------------------------

#MCimri.calculate_dE(Es_i, Ls_i, Lz_i, binaryC, a_i, mult = 1, include_DF=INCLUDE_DF, include_3body=INCLUDE_3BODY)

#assert 1 == 0


#Simulate over Norb orbits and reconstruct density
#---------------------------------------------
Norb = 20000
N_out = 4000

Es = 1.0*Es_i
Ls = 1.0*Ls_i
Lz = 1.0*Lz_i



r_orb = 1.0*a_i
t = 0

dN = 1

rho_c = []
N_list = []
E_tot = []

for i in tqdm(range(Norb)):
    #if (i > 1001):
    #    N_out = 250
    
    if (i%N_out == 0):
        N_list.append(i)
        rho_c.append(orbits.reconstruct_density_full(np.array([r_orb,]), Es, Ls, weights, m1)[0])
        rholist_full = orbits.reconstruct_density_full(rlist, Es, Ls, weights, m1)
        print(i, r_orb/a_i)

        hdrtxt = "Columns: r [pc], rho [Msun/pc**3], rho/rho_i"
        np.savetxt(datapath + f"Density_{IDstr}_Norb_" + str(int(i)) + ".txt", np.column_stack((rlist/u.pc, rholist_full/(u.Msun/u.pc**3), rholist_full/rholist_i)), header=hdrtxt)
        
        rho_th = orbits.reconstruct_density_full_th(a_i, thetas, Es, Ls, Lz, weights, m1)
        np.savetxt(datapath + f"Density_theta_{IDstr}_Norb_" + str(int(i)) + ".txt", np.column_stack((thetas, rho_th/(u.Msun/u.pc**3))))
        plt.plot(thetas, rho_th, label=str(int(i)) + " orbits")
        
        if (MAKE_PLOTS): make_plot(i)
        
        E_tot.append(np.sum(Es*weights))
        
    
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
    
        Es[inds] += np.nan_to_num(dE)
        Ls[inds] += np.nan_to_num(dL)
        Lz[inds] += np.nan_to_num(dLz)
    
    if (r_orb < r_isco):
        break

rho_c.append(orbits.reconstruct_density_full(np.array([r_orb,]), Es, Ls, weights, m1)[0])
rholist_full = orbits.reconstruct_density_full(rlist, Es, Ls, weights, m1)
N_list.append(Norb)
E_tot.append(np.sum(Es*weights))

N_list = np.array(N_list)
rho_c = np.array(rho_c)
E_tot = np.array(E_tot)
np.savetxt(datapath + f"rho_c_MC_{IDstr}_Norb_" + str(int(Norb)) + ".txt", np.column_stack((N_list, rho_c/(u.Msun/u.pc**3))))
np.savetxt(datapath + f"Etot_MC_{IDstr}_Norb_" + str(int(Norb)) + ".txt", np.column_stack((N_list, E_tot/(u.Msun*(u.km/u.s)**2))))

rho_th_f = orbits.reconstruct_density_full_th(a_i, thetas, Es, Ls, Lz, weights, m1)
np.savetxt(datapath + f"Density_theta_{IDstr}_Norb_" + str(int(Norb)) + ".txt", np.column_stack((thetas, rho_th_f/(u.Msun/u.pc**3))))


plt.plot(thetas, rho_th_f, label = str(int(Norb)) + " orbits")

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\rho(r_2, \theta)$ [Arb. units]")

plt.legend(loc='best', fontsize=10)

plt.savefig("../plots/Density_angular.pdf", bbox_inches='tight')

plt.figure()
vr_sq = 2*(orbits.psi(a_i, m1) - Es_i) - Ls_i**2/a_i**2
inds = vr_sq > 0

bins = np.linspace(-1, 1, 25)
plt.hist(Lz_i[inds]/Ls_i[inds],bins, alpha=0.8, label="Initial Distribution")

vr_sq = 2*(orbits.psi(a_i, m1) - Es) - Ls**2/a_i**2
inds = vr_sq > 0

bins = np.linspace(-1, 1, 25)
plt.hist( Lz[inds]/Ls[inds], bins,  alpha=0.8, label="Final Distribution")

plt.legend(loc='best')

plt.xlabel(r"$L_z/L$")
plt.ylabel(r"Counts")

plt.savefig("../plots/Lz_distribution.pdf", bbox_inches='tight')

plt.show()

if (MAKE_PLOTS): 
    plt.figure()
    plt.semilogy(N_list, rho_c/rho_c[0])
    plt.xlabel(r"$N_\mathrm{orb}$")
    plt.ylabel(r"$\rho(r_2)/\rho_i(r_2)$")
    plt.savefig(plotpath + f"rho_c_{IDstr}_Norb_" + str(int(Norb)) + ".pdf", bbox_inches='tight')
    plt.close()

hdrtxt = "Columns: r [pc], rho [Msun/pc**3], rho/rho_i"
np.savetxt(datapath + f"Density_{IDstr}_Norb_" + str(int(Norb)) + ".txt", np.column_stack((rlist/u.pc, rholist_full/(u.Msun/u.pc**3), rholist_full/rholist_i)), header=hdrtxt)
if (MAKE_PLOTS): make_plot(Norb)