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
#parser.add_argument("-m1", type=float, default = 4e6)
parser.add_argument("-logm2", type=float, default = 4)
parser.add_argument("-rM", type=float, default = 100)

args = parser.parse_args()
m2 = (10**args.logm2)*u.Msun
rM_val = args.rM


#Specify the binary system
m1 = 4e6*u.Msun

fstr_base = f"logM2_{np.log10(m2/u.Msun):.2f}_NDM_2000_rM_{str(int(rM_val))}_lc_allorbits"
print(fstr_base)

ext = ".txt.gz"

binaryC = binaries.CircularBinary(m1, m2)
r_isco = binaryC.r_isco

a_i = rM_val*r_isco/6
#SpikeDF = df.GeneralizedNFWSpike(m1, rho_6=1*u.Msun/u.pc**3, gamma_sp=7/3, r_t=20*a_i, alpha=2)

N_jobs = 32
for i in range(N_jobs):
    fstr = f"{fstr_base}_" + str(int(i))
    datapath = "../data/" + fstr + "/"

    files = glob.glob(datapath + "Orbits_" + fstr + "_Norb_*" + ext)
    if not files:
        print("No .txt.gz files found, trying with .txt extension...")
        ext = ".txt"
        files = glob.glob(datapath + "Orbits_" + fstr + "_Norb_*" + ext)
    
    largest_file = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    data = np.loadtxt(largest_file)
    if i == 0:
        Es = data[:,0]
        Ls = data[:,1]
        Lz = data[:,2]
        print(largest_file)
    else:
        Es = np.append(Es, data[:,0])
        Ls = np.append(Ls, data[:,1])
        Lz = np.append(Lz, data[:,2])


np.savetxt(f"../data/Orbits_{fstr_base}_Norb_final.txt", np.column_stack((Es, Ls, Lz)))
