[![ASCL record](https://img.shields.io/badge/ASCL-2109.011-blue.svg?colorB=262255)](https://ascl.net/2109.011)
[![PyPI version](https://badge.fury.io/py/rubble.svg)](https://badge.fury.io/py/rubble)   
[![Paper](https://img.shields.io/static/v1?label=MethodPaper1&message=DOI:10.1093/mnras/stab3677&color=blue&link=https://doi.org/10.1093/mnras/stab3677)](https://doi.org/10.1093/mnras/stab3677)   
[![Paper](https://img.shields.io/static/v1?label=MethodPaper2&message=DOI:10.1093/mnras/stae581&color=blue&link=https://doi.org/10.1093/mnras/stae581)](https://doi.org/10.1093/mnras/stae581)

# Rubble: An implicit code for simulating the local evolution of solid size distributions in protoplanetary disks

`Rubble` ([Li, Chen & Lin, 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.5246L), [2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.529..893L)) implicitly models the local evolution of dust distributions in size, mass, and surface density by solving the Smoluchowski equation (also known as the coagulation-fragmentation equation) under given disk conditions. 

The code robustness has been validated by a suite of numerical benchmarks against known analytical and empirical results. `Rubble` is also able to model prescribed physical processes such as bouncing, modulated mass transfer, regulated dust loss/supply, and probabilistic collisional outcomes based on velocity distributions, etc.  A thermal evolution module has been later included to self-consistently update opacity and temperature, and models silicate evaporation and condensation using the Clausius–Clapeyron relation. The package also includes a toolkit for analyzing and visualizing results produced by `Rubble`.

`Rubble` is built on `PyTorch`, enabling GPU-accelerated computation and significant performance gains over traditional CPU-based linear algebra backends.

# Installation

You may install `Rubble` by this command:

```bash
pip install -U rubble
```

Or, you may try the most updated `Rubble` by this command:

```bash
pip install -U -e git+git://github.com/astroboylrx/Rubble
```

It will automatically install all the required modules.

# Usage

Three demo Jupyter Notebooks are provided under the `docs/JupyterNotebooks` folder:

- **Demo1** — Analytical kernels (constant, linear, product) and coagulation/fragmentation in protoplanetary disk environments, benchmarked against known analytical and empirical results.
- **Demo2** — Collision physics including coagulation, bouncing, and fragmentation (destructive, erosion, and mass transfer), with or without velocity distributions.
- **Demo3** — Evaporation and condensation of silicates via the Clausius–Clapeyron relation, with timestep convergence tests at different temperature regimes.

You may also try `help(Rubble)` to read the raw documentation.