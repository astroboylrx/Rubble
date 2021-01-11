[![PyPI version](https://badge.fury.io/py/rubble.svg)](https://badge.fury.io/py/rubble) [![Anaconda version](https://anaconda.org/astroboylrx/rubble/badges/version.svg)
](https://anaconda.org/astroboylrx/rubble)

# Rubble: An implicit code for simulating the local evolution of solid size distributions in protoplanetary disks

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

Also, if you are using Intel processors, it is strongly recommended to use [Intel Distribution for Python](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html) (or at least `intel-numpy` and `intel-scipy`), which will speed up the calculations for one or two order of magnitudes. It is relatively easier if you are already using `Conda` or `Anaconda` following [this instructions](https://software.intel.com/content/www/us/en/develop/articles/using-intel-distribution-for-python-with-anaconda.html). After that, you can activate the environment and install `Rubble` by

```bash
conda install -c astroboylrx rubble
```



# Usage

For now, please check out the Jupyter Notebooks under the `doc/JupyterNotebooks` folder. More examples will be added soon.

You may also try `help(Rubble)` to read the naive documentation.