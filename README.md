# MCIMRI

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.21822507-blue)](https://doi.org/10.5281/zenodo.21822507) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`MCIMRI` is a code for studying the co-evolution of Dark Matter (DM) spikes and intermediate mass ratio inspirals (IMRIs). The code uses a phase space Monte Carlo approach to evolve the orbits of DM particles, while self-consistently evolving the binary trajectory under gravitational wave energy losses and dynamical friction. 

### Examples

The code comes with a range of tools and further documentation will be provided soon. However, to evolve a system, you can use the script: 
```bash
python3 EvolveBinary_v2.py
```
The script takes the following flags:
- `-logm1`: The log10 of the primary mass in Msun
- `-logm2`: The log10 of the secondary mass in Msun
- `-rS`, `-pc`: The initial binary separation. Only one of these two flags can be set, with the first (`-rS`) being the separation in units of Schwarzschild Radii of the primary and the second (`-pc`) being the separation in pc
- `-rank`: This is a useful flag (integer) for labelling different runs in parallel. If using only one run, this can be set to 0. 

Further internal options can be set directly in the `EvolveBinary_v2.py` script and in the `MCimri.py` module, which contains the core of the Monte Carlo evolution of the DM spike. 
