# BiorbdOptim
An optimization framework that links CasADi, ipopt and biorbd for Optimal Control Problem 

## Status

| | |
|---|---|
| Continuous integration | [![Build Status](https://travis-ci.org/pyomeca/BiorbdOptim.svg?branch=master)](https://travis-ci.org/pyomeca/BiorbdOptim) |
|  Code coverage | [![codecov](https://codecov.io/gh/pyomeca/BiorbdOptim/branch/master/graph/badge.svg)](https://codecov.io/gh/pyomeca/BiorbdOptim) |

# How to install 
The way to install BiorbdOptim on your computer is to installing from the sources. 

## Installing from the sources 
You simply have to download the source, navigate to the root folder and (assuming your conda environment is loaded if needed) type the following command :
```bash 
python setup.py install
```
Please note that this method will not install the dependencies for you, therefore you will have to install them manually. Moreover, the setuptools dependencies must be installed prior to the installation in order for it to work.

## Dependencies
BiorbdOptim relies on several libraries. The most obvious one is BIORBD, but pyomeca is also requires and some others.

There are two ways to install Biorbd, to choose the better way for you please refer to the ReadMe file here : https://github.com/pyomeca/biorbd.

Moreover, BiorbdViz is optional but can be useful to visualize your simulations. To install BiorbdViz please refer to the ReadMe file here : https://github.com/pyomeca/biorbd-viz.

The first hand dependencies (meaning that some dependencies may require other libraries themselves) are: rbdl-casadi (https://github.com/pyomeca/rbdl-casadi), pandas (https://pandas.pydata.org/), numpy (https://numpy.org/), scipy (https://scipy.org/), matplotlib (https://matplotlib.org/), vtk (https://vtk.org/), PyQt (https://www.riverbankcomputing.com/software/pyqt), pyomeca (https://github.com/pyomeca/pyomeca), tinyxml(http://www.grinninglizard.com/tinyxmldocs/index.html) and Ipopt (https://github.com/coin-or/Ipopt) . All these can manually be install using (assuming the anaconda environment is loaded if needed) `pip3` command or the Anaconda's following command.
```bash
conda install rbdl-casadi pandas numpy scipy matplotlib vtk pyqt pyomeca tinyxml ipopt -cconda-forge
```
Please note that some libraries in this command may have been installed  during the installation of Biorbd and BiorbdViz. 
