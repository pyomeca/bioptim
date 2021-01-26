# `bioptim`
`Bioptim` is an optimal control program (OCP) framework for biomechanics. 
It is based on the efficient [biorbd](https://github.com/pyomeca/biorbd) biomechanics library and befits from the powerful algorithmic diff provided by [CasADi](https://web.casadi.org/).
It interfaces the robust [Ipopt](https://github.com/coin-or/Ipopt) and fast [ACADOS](https://github.com/acados/acados) solvers to suit all your needs for an OCP in biomechanics. 

## Status

| | |
|---|---|
| License | <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a> |
| Continuous integration | [![Build Status](https://travis-ci.org/pyomeca/bioptim.svg?branch=master)](https://travis-ci.org/pyomeca/bioptim) |
| Code coverage | [![codecov](https://codecov.io/gh/pyomeca/bioptim/branch/master/graph/badge.svg?token=NK1V6QE2CK)](https://codecov.io/gh/pyomeca/bioptim) |

# How to install 
The prefered way to install for the lay user is using anaconda. 
Another way, more designed for the core programmers is from the sources. 
When it is theoritically possible to use `bioptim` from Windows, it is highly discouraged since it will required to manually compile all the dependecies. 
A great alternative for the Windows users is *Ubuntu on Windows*.

## Installing from Anaconda (For Linux and Mac)
The easiest way to install `bioptim` is to download the binaries from [Anaconda](https://anaconda.org/) repositories. 
The project is hosted on the conda-forge channel (https://anaconda.org/conda-forge/bioptim).

After having installed properly an anaconda client [my suggestion would be Miniconda (https://conda.io/miniconda.html)] and loaded the desired environment to install `bioptim` in, just type the following command:
```bash
conda install -c conda-forge bioptim
```
This will download and install all the dependencies and install `bioptim`. 
And that is it! 
You can already enjoy bioptiming!

The current status of `bioptim` on conda-forge is
| License | Name | Downloads | Version | Platforms |
| --- | --- | --- | --- | --- |
|   <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a> | [![Conda Recipe](https://img.shields.io/badge/recipe-bioptim-green.svg)](https://anaconda.org/conda-forge/bioptim) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/bioptim.svg)](https://anaconda.org/conda-forge/bioptim) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/bioptim.svg)](https://anaconda.org/conda-forge/bioptim) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/bioptim.svg)](https://anaconda.org/conda-forge/bioptim) |


## Installing from the sources (For Windows, Linux and Mac)
Installing from the source is basically as easy as installing from Anaconda, with the difference that you will be required to download and install the dependencies by hand (see section below). 

Once you have downloaded `bioptim`, you navigate to the root folder and (assuming your conda environment is loaded if needed), you can type the following command:
```bash 
python setup.py install
```
Assuming everything went well, that is it! 
You can already enjoy bioptiming!

Please note that Windows is shown here as possible OS. 
As stated before, when this is theoretically possible, it will require that you compile `CasADi`, `RBDL` and `biorbd` by hand since the Anaconda packages are not built for Windows.
This is therefore highly discouraged. 

## Dependencies
`bioptim` relies on several libraries. 
The most obvious one is `biorbd` suite (including indeed `biorbd` and `bioviz`), but some extra more are required.
Due to the amount of different dependencies, it would be tedecious to show how to install them all here. 
The user is therefore invited to read the relevant documentation. 

Here is a list of all direct dependencies (meaning that some dependencies may require other libraries themselves):
- [Python](https://www.python.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [packaging](https://packaging.python.org/)
- [setuptools](https://pypi.org/project/setuptools/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [pyomeca](https://github.com/pyomeca/pyomeca)
- [CasADi](https://web.casadi.org/)
- [rbdl-casadi](https://github.com/pyomeca/rbdl-casadi) compiled with the CasADi backend
- [tinyxml](http://www.grinninglizard.com/tinyxmldocs/index.html)
- [biorbd](https://github.com/pyomeca/biorbd)
- [vtk](https://vtk.org/)
- [PyQt](https://www.riverbankcomputing.com/software/pyqt)
- [bioviz](https://github.com/pyomeca/bioviz)
- [Ipopt](https://github.com/coin-or/Ipopt)
- [ACADOS](https://github.com/acados/acados)

All these (except for ACADOS) can manually be installed using (assuming the anaconda environment is loaded if needed) `pip3` command or the Anaconda's following command.
```bash
conda install casadi rbdl=*=*casadi* biorbd=*=*casadi* [bioviz=*=*casadi*] -cconda-forge
```

Since there isn't any `Anaconda` nor `pip3` package of ACADOS, a convenient installer is provided with `bioptim`. 
The installer can be found and run at `[ROOT_BIOPTIM]/external/acados_install.sh`.
However, the installer requires an `Anaconda` environment.
If you have an `Anaconda` environment loaded, the installer should find itself where to install. 
If you want to install elsewhere, you can provide the script with a first argument which is the `$CONDA_PREFIX`. 
The second argument that can be passed to the script is the `$BLASFEO_TARGET`. 
If you don't know what it is, it is probably better to keep the default. 
Please note that depending on your computer architecture, ACADOS may or may not work properly.


# Getting started
The easiest way to lear `bioptim` is to dive into.
So let's do that and build our first optimal control program together.
Please note that this tutorial is design to recreate the `examples/getting_started/pendulum.py` file where a pendulum is asked to start in a downward position and to end, balanced, in an upward position while only being able to actively move sideways.

## The import
We won't spend time explaining the import, since every one of them will be explained in details later, and is pretty straightforward anyway.
```python
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ShowResult,
    ObjectiveFcn,
    Objective,
)
```

## Building the ocp
First of all, let's load a bioMod file using `biorbd`:
```python
biorbd_model = biorbd.Model("pendulum.bioMod")
```
It is convenient since it will provide interesting functions such as the number of degrees of freedom (`biorbd_model.nbQ()`). 
Please note that a copy of `pendulum.bioMod` is available at the end of the *Getting started* section.
In brief, the pendulum consists of two degrees of freedom (sideway movement and rotation) with the center of mass near the head.

The dynamics of the pendulum, as for a lot of biomechanics one as well, is to drive it by the generalized forces. 
That is forces and moments directly applied to the degrees of freedom as if virtual motors were to power them.
This dynamics is called in `bioptim` torque driven. 
In a torque driven dynamics, the states are the positions (also called generalized coordinates, q) and the velocities (also called the generalized velocities, qdot) and the controls are the joint torques (also called generalized forces, tau). 
Let's define such a dynamics:
```python
dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
```

The pendulum is required to start in a downward position (0 rad) and finish in upward position (3.14 rad) with no velocity at start and end nodes.
To define that, it would be nice to first define boundary constraints on the position (q) and velocities (qdot) that match those in the bioMod file and to apply them at the very beginning, the very end and all the intermediate nodes as well.
QAndQDotBounds waits for a biorbd model and returns a structure with the minimal and maximal bounds for all the deegres of freedom and velocities on three columns corresponding to the starting node, the intermediate nodes and the final nodes, respectively.
How convenient!
```python
x_bounds = QAndQDotBounds(biorbd_model)
```
Then, override the first and last column to be 0, that is the sideway and rotation to be null for both the position and the velocities
```python
x_bounds[:, [0, -1]] = 0
```
Finally, override once again the final node for the rotation so it is upside down.
```python
x_bounds[1, -1] = 3.14
```
At that point, you may want to have a look at the `x_bounds.min` and `x_bounds.max` matrice to convince yourself that the initial and final position and velocities are prescribed and that all the intermediate points are free up to a certain minimal and maximal values. 

Up to that point, there is nothing preventing the solver to simply use the virtual motor of the rotation to rotate the pendulum upward (like clock hands) to get to the upside down rotation. 
What makes this example interesting is that we can prevent this by defining minimal and maximal bounds on the control (the maximal forces that these motors have)
```
u_bounds = Bounds([-100, 0], [100, 0])
```
Like this, the sideway force ranges from -100 Newton to 100 Newton, but the rotation force ranges from 0 N/m to 0 N/m.
Again, `u_bound` is defined for the first, the intermediate and the final nodes, but this time, we don't want to specify anything particular for the first and final nodes, so we can leave them as is. 

You says optimization says cost function.
Even though, it is possible to define an OCP without objective, it is not so much recommanded, and let's face it... much less fun!
So the goal (or the cost function) of the pendulum is to perform its task while using the minimum forces as possible. 
Therefore, an objective function that minimizes the generalized forces is defined:
```python
objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)
```

At that point, it is possible to solves the program.
Still, helping the solver is usually a good idea, so let's give Ipopt a starting point to investigate.
The initial guess that we can provide are those for the states (`x_init`, here q and qdot) and for the controls (`u_init`, here tau). 
So let's define both of them quickly
```python
x_init = InitialGuess([0, 0, 0, 0])
u_init = InitialGuess([0, 0])
```
Please note that `x_init` is twice the size of `u_init` because it contains the two degrees of freedom from the generalized coordinates (q) and the two from the generalized velocities (qdot), while `u_init` only contains the generalized forces (tau)

We now have everything to create the ocp!
For that we have to decide how much time the pendulum has to get up there (`phase_time`) and how many shooting point are defined for the multishoot (`n_shooting`).
Thereafter, you just have to send everything to the `OptimalControlProgram` class and let `bioptim` prepares everything for you.
For simplicity sake, I copy all the piece of code previously visited in the building the ocp section here:
```python
ocp = OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting=25,
        phase_time=3,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )
```

## Solving the ocp
It is now time to see `Ipopt` in action! 
To solve the ocp, you simply have to call the `solve()` method of the `ocp` class
```python
sol = ocp.solve(show_online_optim=True)
```
If you feel fancy, you can even activate the only opimization graphs!
However, for such easy problem, `Ipopt` won't leave you the time to appreciate the update realtime updates of the graph...
That's it!

## Watching the results
If you want to have a look at the animated data, `bioptim` has an interface to `bioviz` which is designed to vizualize bioMod files.
For that, simple call the `animate()` method of a `ShowData` class as such:
```python
ShowResult(ocp, sol).animate()
```

If you did not fancy the only graphs, but would enjoy them anyway, you can call the same class with the method `graphs()`:
```python
ShowResult(ocp, sol).graphs()
```

And that is all! 
You have completed your first optimal control program with `bioptim`! 

## The expected files
If you did not completely follow (or were too lazy to!) you will find in this section the complete file described in the Getting started section.
You will find that the file is a bit different from the `example/getting_started/pendulum.py`, but it is merely differences on the surface.

### The pendulum.py file
```python
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ShowResult,
    ObjectiveFcn,
    Objective,
)

biorbd_model = biorbd.Model("pendulum.bioMod")
dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
x_bounds = QAndQDotBounds(biorbd_model)
x_bounds[:, [0, -1]] = 0
x_bounds[1, -1] = 3.14
u_bounds = Bounds([-100, 0], [100, 0])
objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)
x_init = InitialGuess([0, 0, 0, 0])
u_init = InitialGuess([0, 0])

ocp = OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting=25,
        phase_time=3,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )
    
sol = ocp.solve(show_online_optim=True)
ShowResult(ocp, sol).animate()
```
## The pendulum.bioMod file
Here is a simple pendulum that can be interpreted by `biorbd`. 
For more information on how to build a bioMod file, one can read the doc of [biorbd](https://github.com/pyomeca/biorbd).

```c
version 4

// Seg1
segment Seg1
    translations	y
    rotations	x
    ranges  -1 5
            -2*pi 2*pi
    mass 1
    inertia
        1 0 0
        0 1 0
        0 0 0.1
    com 0.1 0.1 -1
    mesh 0.0   0.0   0.0
    mesh 0.0  -0.0  -0.9
    mesh 0.0   0.0   0.0
    mesh 0.0   0.2  -0.9
    mesh 0.0   0.0   0.0
    mesh 0.2   0.2  -0.9
    mesh 0.0   0.0   0.0
    mesh 0.2   0.0  -0.9
    mesh 0.0   0.0   0.0
    mesh 0.0  -0.0  -1.1
    mesh 0.0   0.2  -1.1
    mesh 0.0   0.2  -0.9
    mesh 0.0  -0.0  -0.9
    mesh 0.0  -0.0  -1.1
    mesh 0.2  -0.0  -1.1
    mesh 0.2   0.2  -1.1
    mesh 0.0   0.2  -1.1
    mesh 0.2   0.2  -1.1
    mesh 0.2   0.2  -0.9
    mesh 0.0   0.2  -0.9
    mesh 0.2   0.2  -0.9
    mesh 0.2  -0.0  -0.9
    mesh 0.0  -0.0  -0.9
    mesh 0.2  -0.0  -0.9
    mesh 0.2  -0.0  -1.1
endsegment

    // Marker 1
    marker marker_1
        parent Seg1
        position 0 0 0
    endmarker

    // Marker 2
    marker marker_2
        parent Seg1
        position 0.1 0.1 -1
    endmarker
```

# A more in depth look at `bioptim`

## Objective function
TODO + Lagrange objective functions are integrated using rectangle method.

## Online plotting
TODO + It is expected to slow down the optimization by about 15%

# Citing

If you use `bioptim`, we would be grateful if you could cite it as follows:

@misc{Michaud2020bioptim,
    author = {Michaud, Benjamin and Bailly, Francois and Begon, Mickael et al.},
    title = {bioptim, a Python interface for Musculoskeletal Optimal Control in Biomechanics},
    howpublished={Web page},
    url = {https://github.com/pyomeca/bioptim},
    year = {2020}
}
