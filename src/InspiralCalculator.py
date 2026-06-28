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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m1", type=float, default = 4e6)
parser.add_argument("-m2", type=float, default = 1e4)
#parser.add_argument("-x", type=float, default = 16.67)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-rS', type = float)
group.add_argument('-pc', type = float)


args = parser.parse_args()
m1 = args.m1*u.Msun
m2 = args.m2*u.Msun



#Specify the binary system

binary = binaries.CircularBinary(m1, m2)
rS = binary.r_isco/3

if (args.rS is not None):
    r = args.rS*rS
else:
    r = args.pc*u.pc
    
T_i = binary.T_orb(r)

print(f"(m1, m2) = ({m1/u.Msun:.2e}, {m2/u.Msun:.2e}) Msun")
print(f"r = {r/rS:.2e} rS = {r/binary.r_isco:.2e} r_isco = {r/u.pc:.2e} pc")
print(f"T_orb_i = {T_i/u.s} s")

#N_per_min = 26_000  #1000 particles per sim
N_per_min  = 18_500  #2000 particles per sim

N = binary.Norb_to_merger(r)

print(f"Number of orbits to merger: {N:.2e}")
print(f"Simulation time required [hr]: {(N/N_per_min)/60}")
