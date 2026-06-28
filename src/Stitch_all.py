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
import re

#Command line arguments
#----------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-rank", type=int)
parser.add_argument("-logm1", type=float, default = np.log10(4e6))
parser.add_argument("-logm2", type=float, default = 4)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-rS', type = float)
group.add_argument('-pc', type = float)


args = parser.parse_args()


#Specify the binary system
#-------------------------
m1 = (10**args.logm1)*u.Msun
m2 = (10**args.logm2)*u.Msun

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

dN = 1

dN_str = ""
if (dN > 1):
    dN_str = f"_dN_{str(int(dN))}"


froot = f"logM1_{np.log10(m1/u.Msun):.2f}_logM2_{np.log10(m2/u.Msun):.2f}_{rstr}{dN_str}"

N_jobs = 3

plotpath = "../plots/" + froot + "/"

#--------------------------------------
print("Stitching rho(r) files...")
for i in range(N_jobs):
    fstr = froot + "_" + str(int(i))
    datapath = "../data/" + froot + "/" + fstr + "/"

    ext = ".txt.gz"
    fname = "Density_r_" + fstr + ext
    
    data = np.loadtxt(datapath + fname)

    if i == 0:
        Nlist = data[:,0]
        rlist = data[:,1]*u.pc
        
        rho_full = data[:,2]*u.Msun/u.pc**3
        rho_full_free = data[:,3]*u.Msun/u.pc**3
    else:
        rho_full += data[:,2]*u.Msun/u.pc**3
        rho_full_free += data[:,3]*u.Msun/u.pc**3

rho_full /= N_jobs
rho_full_free /= N_jobs

hdrtxt = "Columns: N_orbs, r_2 [pc], rho(r_2), no capture [Msun/pc**3], rho(r_2), including capture [Msun/pc**3]"
np.savetxt("../data/" + froot + "/"  + f"Density_r_{froot}_all.txt", np.column_stack((Nlist, rlist/u.pc, rho_full/(u.Msun/u.pc**3), rho_full_free/(u.Msun/u.pc**3))), header=hdrtxt)

plt.figure()

plt.loglog(Nlist, rho_full/(u.Msun/u.pc**3))
plt.loglog(Nlist, rho_full_free/(u.Msun/u.pc**3), linestyle='--')

plt.savefig(f"../plots/" + froot + f"/Density_r_{froot}.pdf", bbox_inches='tight')


#--------------------------------------
print("Stitching E_total files...")
for i in range(N_jobs):
    fstr = froot + "_" + str(int(i))
    datapath = "../data/" + froot + "/" + fstr + "/"
    

    ext = ".txt.gz"
    fname = "Etot_" + fstr + ext
    
    data = np.loadtxt(datapath + fname)

    if i == 0:
        Nlist = data[:,0]
        
        Etot = data[:,1]
        Etot_free = data[:,2]
    else:
        Etot += data[:,1]
        Etot_free += data[:,2]

Etot /= N_jobs
Etot_free /= N_jobs

hdrtxt = "Columns: N_orbs, Total DM energy (no capture) [Msun (km/s)^2], Total DM energy (including capture) [Msun (km/s)^2]"
np.savetxt("../data/" + froot + "/" + f"Etot_{froot}_all.txt", np.column_stack((Nlist, rho_full, rho_full_free)), header=hdrtxt)



plt.figure()

plt.loglog(Nlist, Etot)
plt.loglog(Nlist, Etot_free, linestyle='--')

plt.savefig(f"../plots/" + froot + f"/Etot_{froot}.pdf", bbox_inches='tight')



#--------------------------------------
print("Stitching density profile files...")

j = 0
RUNNING = True

while RUNNING:
    print(j)
    for i in range(N_jobs):
        fstr = f"{froot}_" + str(int(i))
        datapath = "../data/" + froot + "/" + fstr + "/"

        ext = ".txt.gz"
        files = glob.glob(datapath + "Density_" + fstr + "_Norb_*" + ext)
        if not files:
            print("No .txt.gz files found, trying with .txt extension...")
            ext = ".txt"
            files = glob.glob(datapath + "Density_" + fstr + "_Norb_*" + ext)   
 
        pairs = sorted(
            [(f, int(re.search(r"Norb_(\d+)", f).group(1))) for f in files],
                key=lambda x: x[1]
            )

        files_sorted = [f for f, _ in pairs]
        numbers_sorted = [n for _, n in pairs]

        N_snaps = len(numbers_sorted)
   
        file_j = files_sorted[j]
        n_j = numbers_sorted[j]

        print(file_j)
        data = np.loadtxt(file_j)
    
        if i == 0:
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
    np.savetxt("../data/" + froot + "/" + f"Density_{froot}_all_Norb_{str(n_j)}.txt", np.column_stack((rlist/u.pc, rho_full/(u.Msun/u.pc**3), rho_ratio, rho_full_free/(u.Msun/u.pc**3), rho_ratio_free)), header=hdrtxt)

    j += 1
    if (j >= (N_snaps)):
        RUNNING = False
        
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
    binary.r_isco/u.pc,
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

plt.savefig(f"../plots/" + froot + f"/FinalDensity_{froot}.pdf", bbox_inches='tight')

print("DONE!")

#plt.show()
