# MCIMRI

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.21822507-blue)](https://doi.org/10.5281/zenodo.21822507) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`MCIMRI` is a code for studying the co-evolution of Dark Matter (DM) spikes and intermediate mass ratio inspirals (IMRIs). The code uses a phase space Monte Carlo approach to evolve the orbits of DM particles, while self-consistently evolving the binary trajectory under gravitational wave energy losses and dynamical friction. 

### Code

The code requires [`NbodyIMRI`](https://github.com/bradkav/NbodyIMRI) to be installed.

The relevant code is in the `src` directory:
- `MCimri.py`. The core of the code for calculating the energy kicks of particles at each orbit of the binary.
- `binaries.py`. Code for defining and calculating the properties of binaries (orbital periods etc).
- `orbits.py`. Code for drawing Keplerian orbits from a given set of orbital parameters.
- `utilities.py`. A range of tools, such as performing averages, certain vector operations, and storing output of the simulations.
- `InspiralCalculator.py`. Code to calculate the number of orbits to merger for a given binary and initial separation.

### Examples

The code comes with a range of tools and further documentation will be provided soon. However, to evolve a system, you can use the script: 
```bash
python3 EvolveBinary.py
```
The script takes the following flags:
- `-logm1`: The log10 of the primary mass in Msun
- `-logm2`: The log10 of the secondary mass in Msun
- `-rS`, `-pc`: The initial binary separation. Only one of these two flags can be set, with the first (`-rS`) being the separation in units of Schwarzschild Radii of the primary and the second (`-pc`) being the separation in pc
- `-rank`: This is a useful flag (integer) for labelling different runs in parallel. If using only one run, this can be set to 0. 
- `-IDstr`: Identifier string which will be appended to the output files.
- `-N_particles`: Number of DM particles to be simulated in the run. 
- `-mode`: The simulation can be run with the options `Full` (which includes dynamical friction and stirring effects, the default), `NoInner` (which includes dynamical friction and stirring effects only outside the binary), or `NoStir` (which includes only dynamical friction). 

Further internal options can be set directly in the `EvolveBinary.py` script and in the `MCimri.py` module, which contains the core of the Monte Carlo evolution of the DM spike. 
