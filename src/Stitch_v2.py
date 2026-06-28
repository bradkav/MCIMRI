import numpy as np
from numpy import cos, sin

from tqdm import tqdm

from scipy.interpolate import interp1d

from NbodyIMRI import units as u
from NbodyIMRI import distributionfunctions as df
import binaries
import orbits
import utilities

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-logm1", type=float, default = np.log10(4e6))
parser.add_argument("-logm2", type=float, default = 4)
parser.add_argument("-r_pc", type=float, default = 1e-9)

args = parser.parse_args()
m1 = (10**args.logm1)*u.Msun
m2 = (10**args.logm2)*u.Msun
a_i = args.r_pc*u.pc

rstr = f"r_{a_i/u.pc:.2e}_pc"

fstr_base = f"logM1_{np.log10(m1/u.Msun):.2f}_logM2_{np.log10(m2/u.Msun):.2f}_{rstr}"
print(fstr_base)

ext = ".txt.gz"

binaryC = binaries.CircularBinary(m1, m2)
r_isco = binaryC.r_isco

#a_i = rM_val*r_isco/6
#SpikeDF = df.GeneralizedNFWSpike(m1, rho_6=1*u.Msun/u.pc**3, gamma_sp=7/3, r_t=20*a_i, alpha=2)

N_jobs = 32
for i in range(N_jobs):
    fstr = f"{fstr_base}_" + str(int(i))
    datapath = "../data/" + fstr + "/"

    files = glob.glob(datapath + "Density_" + fstr + "_Norb_*" + ext)
    if not files:
        print("No .txt.gz files found, trying with .txt extension...")
        ext = ".txt"
        files = glob.glob(datapath + "Density_" + fstr + "_Norb_*" + ext)
    
    largest_file = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    data = np.loadtxt(largest_file)
    if i == 0:
        print(largest_file)
        rlist = data[:,0]*u.pc
        rho_full = data[:,1]*u.Msun/u.pc**3
        rho_ratio = data[:,2]
        rho_full_free = data[:,3]*u.Msun/u.pc**3
        rho_ratio_free = data[:,4]
    else:
        rho_full += data[:,1]*u.Msun/u.pc**3
        rho_ratio += data[:,2]
        rho_full_free += data[:,3]*u.Msun/u.pc**3
        rho_ratio_free += data[:,4]

rho_full /= N_jobs
rho_ratio /= N_jobs

rho_full_free /= N_jobs
rho_ratio_free /= N_jobs

hdrtxt = "Columns: r [pc], rho [Msun/pc**3], rho/rho_i, rho (uncaptured) [Msun/pc**3], rho/rho_i (uncaptured)"
np.savetxt(f"../data/Density_{fstr_base}_Norb_final.txt", np.column_stack((rlist/u.pc, rho_full/(u.Msun/u.pc**3), rho_ratio, rho_full_free/(u.Msun/u.pc**3), rho_ratio_free)), header=hdrtxt)

    
#---------------

plt.figure()

plt.semilogx(rlist/u.pc, rho_ratio, alpha=0.5)
x_new, y_new = utilities.density_avg(rlist/u.pc,rho_ratio, 10)
plt.semilogx(x_new, y_new)

plt.semilogx(rlist/u.pc, rho_ratio_free, alpha=0.5)
x_new, y_new = utilities.density_avg(rlist/u.pc,rho_ratio_free, 10)
plt.semilogx(x_new, y_new)

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
#plt.legend(loc='upper right')

plt.savefig(f"../plots/FinalDensity_logM2_{fstr_base}.pdf", bbox_inches='tight')

print("DONE!")

#plt.show()
