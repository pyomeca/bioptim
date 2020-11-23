# `bioptim`
An optimization framework that links CasADi, ipopt and biorbd for Optimal Control Problem 

## Status

| | |
|---|---|
| Continuous integration | [![Build Status](https://travis-ci.org/pyomeca/bioptim.svg?branch=master)](https://travis-ci.org/pyomeca/bioptim) |
|  Code coverage | [![codecov](https://codecov.io/gh/pyomeca/bioptim/branch/master/graph/badge.svg)](https://codecov.io/gh/pyomeca/bioptim) |

# How to install 
The way to install `bioptim` on your computer is to install from the sources. 

## Installing from the sources 
You simply have to download the source, navigate to the root folder and (assuming your conda environment is loaded if needed) type the following command :
```bash 
python setup.py install
```
WARNING: Since the last refactor, this method won't work anymore, you will have to run `bioptim` from the folder or to install it manually.
Please note that this method will not install the dependencies for you, therefore you will have to install them manually. Moreover, the setuptools dependencies must be installed prior to the installation in order for it to work.

## Dependencies
`bioptim` relies on several libraries. The most obvious one is `biorbd`, but pyomeca is also requires and some others.

There are two ways to install Biorbd, to choose the better way for you please refer to the ReadMe file here : https://github.com/pyomeca/biorbd.

Moreover, `bioviz` is optional but can be useful to visualize your simulations. To install `bioviz` please refer to the ReadMe file here : https://github.com/pyomeca/bioviz.

The first hand dependencies (meaning that some dependencies may require other libraries themselves) are: rbdl-casadi (https://github.com/pyomeca/rbdl-casadi) with the casadi backend (https://web.casadi.org/), pandas (https://pandas.pydata.org/), numpy (https://numpy.org/), scipy (https://scipy.org/), matplotlib (https://matplotlib.org/), vtk (https://vtk.org/), PyQt (https://www.riverbankcomputing.com/software/pyqt), pyomeca (https://github.com/pyomeca/pyomeca), tinyxml (http://www.grinninglizard.com/tinyxmldocs/index.html) and Ipopt (https://github.com/coin-or/Ipopt). All these can manually be install using (assuming the anaconda environment is loaded if needed) `pip3` command or the Anaconda's following command.
```bash
conda install casadi rbdl=*=*casadi* biorbd=*=*casadi* [bioviz=*=*casadi*] -cconda-forge
```


# How to use
TODO

## Objective function
TODO + Lagrange objective functions are integrated using rectangle method.

## Online plotting
TODO + It is expected to slow down the optimization by about 15%

## Citing

If you use `biorb-optim`, we would be grateful if you could cite it as follows:

@misc{Michaud2018bioptim,
    author = {Michaud, Benjamin and Begon, Mickael},
    title = {bioptim: An optimal control framework for biomechanical analyses using biorbd},
    howpublished={Web page},
    url = {https://github.com/pyomeca/bioptim},
    year = {2020}
}
