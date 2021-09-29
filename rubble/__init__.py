""" Rubble: An implicit code for simulating the local evolution of solid size distributions in protoplanetary disks

    Rubble implicitly models the local evolution of solid distributions in size, mass, and surface density by solving
    the Smoluchowski equation (Smoluchowski, M. V. 1916; also often referred as the coagulation equation,
    or the coagulation-fragmentation equation) under given disk conditions. The code robustness has been
    validated by a suite of numerical benchmarks against known analytical and empirical results.
    Rubble is also able to model prescribed physical processes such as bouncing, modulated mass transfer,
    regulated dust loss/supply, and probabilistic collisional outcomes based on velocity distributions, etc.
    The package also includes a toolkit for analyzing and visualizing results produced by Rubble.
"""

__version__ = "0.2.1"
__author__ = "Rixin Li"
__email__ = "rixinli.astro@gmail.com"
__all__ = ["help_info", "Rubble", "RubbleData"]

from .rubble import Rubble
from .rubble_data import RubbleData

def help_info():
    """ Print Basic Help Info """

    print("""
    **************************************************************************
    * Rubble: An implicit code for simulating the local evolution of 
    *         solid size distributions in protoplanetary disks
    * 
    * Rubble implicitly models the local evolution of solid distribution 
    * in size, mass, and surface density by solving the Smoluchowski equation
    * (Smoluchowski, M. V. 1916; also often referred as the coagulation
    * equation, or the coagulation-fragmentation equation) under given disk
    * conditions. The code robustness has been validated by a suite of
    * numerical benchmarks against known analytical and empirical results. 
    * Rubble is also able to model prescribed physical processes such as
    * bouncing, modulated mass transfer, regulated dust loss/supply, and 
    * probabilistic collisional outcomes based on velocity distributions, etc.
    * This package also includes a toolkit for analyzing and visualizing
    * results produced by Rubble.
    * 
    * Author: Rixin Li
    * Current Version: 0.2.1
    * Note: This package is still under active development and we welcome
    *       any comments and suggestions.
    **************************************************************************
    """)