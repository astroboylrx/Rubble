[![PyPI version](https://badge.fury.io/py/rubble.svg)](https://badge.fury.io/py/rubble)
[![Paper](https://img.shields.io/static/v1?label=Submitted&message=MethodPaper&color=blue&link=https://lavinia.as.arizona.edu/~rixin/misc/ms_locked.html#Mei5You4Mi4Ma!)](https://lavinia.as.arizona.edu/~rixin/misc/ms_locked.html#Mei5You4Mi4Ma!)

# Rubble: An implicit code for simulating the local evolution of solid size distributions in protoplanetary disks

`Rubble` implicitly models the local evolution of dust distributions in size, mass, and surface density by solving the Smoluchowski equation (also known as the coagulation-fragmentation equation) under given disk conditions. The code robustness has been validated by a suite of numerical benchmarks against known analytical and empirical results. `Rubble` is also able to model prescribed physical processes such as bouncing, modulated mass transfer, regulated dust loss/supply, and probabilistic collisional outcomes based on velocity distributions, etc. The package also includes a toolkit for analyzing and visualizing results produced by `Rubble`.

## Installation

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

For now, please check out the Jupyter Notebooks under the `doc/JupyterNotebooks` folder. More examples and documentation will be added soon.

You may also try `help(Rubble)` to read the raw documentation.