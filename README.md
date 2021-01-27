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
In a torque driven dynamics, the states are the positions (also called generalized coordinates, *q*) and the velocities (also called the generalized velocities, *qdot*) and the controls are the joint torques (also called generalized forces, *tau*). 
Let's define such a dynamics:
```python
dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
```

The pendulum is required to start in a downward position (0 rad) and finish in upward position (3.14 rad) with no velocity at start and end nodes.
To define that, it would be nice to first define boundary constraints on the position (*q*) and velocities (*qdot*) that match those in the bioMod file and to apply them at the very beginning, the very end and all the intermediate nodes as well.
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
The initial guess that we can provide are those for the states (`x_init`, here *q* and *qdot*) and for the controls (`u_init`, here *tau*). 
So let's define both of them quickly
```python
x_init = InitialGuess([0, 0, 0, 0])
u_init = InitialGuess([0, 0])
```
Please note that `x_init` is twice the size of `u_init` because it contains the two degrees of freedom from the generalized coordinates (*q*) and the two from the generalized velocities (*qdot*), while `u_init` only contains the generalized forces (*tau*)

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

## Show the results
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

## The full files
If you did not completely follow (or were too lazy to!) you will find in this section the complete files described in the Getting started section.
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
### The pendulum.bioMod file
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

# A more in depth look at the `bioptim` API

In this section, we are going to have an in depth look at all the classes one can use to interact with the bioptim API. 
All the classes covered here, can be imported using the command:
```python
from bioptim import ClassName
```

## The Dynamics
By essence, an optimal control program (ocp) links two types of variables: the states (x) and the controls (u). 
Conceptually, the controls could be seen as the driving forces of the system, what makes the system do something, while the states are the consequences of these driving forces. 
In the case of the biomechanics, the states are usually the generalized coordinates (*q*) and velocities (*qdot*), that is the pose of the musculoskelatal model and the speed the joint moves. 
On the other hand, the controls can be the generalized forces, that is the joint torques, but can also be the muscle excitations, for instance.
The key is there are dynamic equations that link them such that: dx/dt = f(x, u, p), where p can be additional parameters that act on the system, but are not time dependent.

The following section investigate how to instruct `bioptim` of the dynamic equations the system should follow.

 
### Class: Dynamics
This class is the main class to define a dynamics. 
It therefore contains all the information necessary to configure (that is determining which variables are states or controls) and performs the dynamics. 
It is what is expected by the `OptimalControlProgram` for its `dynamics_type` parameter. 

The user can minimally define a Dynamics as follow: `dyn = Dynamics(DynamicsFcn)`.
The `DynamicsFcn` are the one presented in the corresponding section below. 

#### The options
The full signature of Dynamics is as follow:
```python
Dynamics(dynamics_type, configure: Callable, dynamic_function: Callable, phase: int)
```
The `dynamics_type` is the selected `DynamicsFcn`. 
It automatically define both `configure` and `dynamic_function`. 
If a function is sent instead, this function is interpreted as `configure` and the DynamicsFcn is assumed to be `DynamicsFcn.CUSTOM`
If one is interested in changing the behaviour of a particular `DynamicsFcn`, they can refer to the Custom dynamics functions right below. 

The `phase` is the index of the phase the dynamics apply to. 
This is usually taken care by the `add()` method of `DynamicsList`, but it can be useful when declaring the dynamics out of order.

#### Custom dynamic functions
If an advanced user wants to defined their own dynamic function, they can define the configuration and/or the dynamics. 

The configuration is what tells `bioptim` which variables are states and which are control.
The user is expected to provide a function handler with the follow signature: `custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram)`.
In this function the user is expected to call the relevant `Problem` class methods: 
- `configure_q(nlp, as_states: bool, as_controls: bool)`
- `configure_qdot(nlp, as_states: bool, as_controls: bool)`
- `configure_q_qdot(nlp, as_states: bool, as_controls: bool)`
- `configure_tau(nlp, as_states: bool, as_controls: bool)`
- `configure_muscles(nlp, as_states: bool, as_controls: bool)`
where `as_states` add the variable to the states vector and `as_controls` to the controls vector.
Please note that this is not necessary mutually exclusive.
Finally, the user is expected to configure the dynamic by calling `Problem.configure_dynamics_function(ocp, nlp, custom_dynamics)`

Defining the dynamic function must be done when one provides a custom configuration, but can also be defined by providing a function handler to the `dynamic_function` parameter for `Dynamics`. 
The signature of this custom dynamic function is as follow: `custom_dynamic(states: MX, controls: MX, parameters: MX, nlp: NonLinearProgram`.
This function is expected to return a tuple[MX] of the derivative of the states. 
Some method defined in the class `DynamicsFunctions` can be useful, but will not be covered here since it is initially designed for internal use.
Please note that MX type is a CasADi type.
Anyone who wants to define custom dynamics should be at least familiar with this type beforehand. 


### Class: DynamicsList
A DynamicsList is by essence simply a list of Dynamics. 
The `add()` method can be called exacly as if one was calling the `Dynamics` constructor. 
If the `add()` method is used more than one, the `phase` parameter is automatically incremented. 

So a minimal use is as follow:
```python
dyn_list = DynamicsList
dyn_list.add(DynamicsFcn)
```

### Enum: DynamicsFcn
The DynamicsFcn Enum is the configuration and declaration of all the already available dynamics in `bioptim`. 
Since this is an Enum, it is possible to use tab key on the keyboard to dynamically list them all, assuming you IDE allows for it. 

Please note that one can change the dynamic function associated to any of the configuration by providing a custom dynamics_function. 
For more information on this, please refer to the Dynamics and DynamicsList section right before. 

#### TORQUE_DRIVEN
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as *tau*. 
The derivative of *q* is trivially *qdot*.
The derivative of *qdot* is given by the biorbd function: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 
If external forces are provided, they are added to the ForwardDynamics function. 

#### TORQUE_DRIVEN_WITH_CONTACT
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as *tau*. 
The derivative of *q* is trivially *qdot*.
The derivative of *qdot* is given by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `qddot = biorbd_model.ForwardDynamicsConstraintsDirect(q, qdot, tau)`.

#### TORQUE_ACTIVATIONS_DRIVEN
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the level of activation of *tau*. 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the activation by the `biorbd` function: `tau = biorbd_model.torque(torque_act, q, qdot)`.
Then, the derivative of *qdot* is given by the `biorbd` function: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 

#### TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the level of activation of *tau*. 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the activation by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `tau = biorbd_model.torque(torque_act, q, qdot)`.
Then, the derivative of *qdot* is given by the `biorbd` function: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 

#### MUSCLE_ACTIVATIONS_DRIVEN
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the muscle activations. 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the muscle activation converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `biorbd_model.muscularJointTorque(muscles_states, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 

#### MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the *tau* and the muscle activations (*a*). 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the sum of *tau* to the muscle activation converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `biorbd_model.muscularJointTorque(a, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 

#### MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the *tau* and the muscle activations (*a*). 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the sum of *tau* to the *a* converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `biorbd_model.muscularJointTorque(a, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 

#### MUSCLE_EXCITATIONS_DRIVEN
The torque driven defines the states (x) as *q*, *qdot* and muscle activations (*a*) and the controls (u) as the *EMG*. 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from *a* converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `biorbd_model.muscularJointTorque(muscles_states, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 
The derivative of *a* is computed by the `biorbd` function: `adot = model.activationDot(emg, a)`

#### MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN
The torque driven defines the states (x) as *q*, *qdot* and muscle activations (*a*) and the controls (u) as the *tau* and the *EMG*. 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the sum of *tau* to *a* converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `biorbd_model.muscularJointTorque(muscles_states, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 
The derivative of *a* is computed by the `biorbd` function: `adot = model.activationDot(emg, a)`

#### MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT
The torque driven defines the states (x) as *q*, *qdot* and muscle activations (*a*) and the controls (u) as the *tau* and the *EMG*. 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the sum of *tau* to *a* converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `biorbd_model.muscularJointTorque(muscles_states, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `qddot = biorbd_model.ForwardDynamics(q, qdot, tau)`. 
The derivative of *a* is computed by the `biorbd` function: `adot = model.activationDot(emg, a)`

#### CUSTOM
This leaves the user to define both the configuration (what are the states and controls) and to define the dynamic function. 
CUSTOM should not be called by the user, but the user should pass the configure_function directly. 
You can have a look at Dynamics and DynamicsList sections for more information about how to configure and define custom dynamics.


## The Bounds
The bounds provide a class that has minimal and maximal values for a variable.
It is, for instance, use for the inequality constraints that limits the maximal and minimal values the states (x) and the controls (u) can have.
In that sense, it is what is expected by the `OptimalControlProgram` for its `u_bounds` and `x_bounds` parameters. 
It can however be used for much more.

### Class: Bounds
The Bounds class is the main class to define bounds.
The constructor can be call by sending two boundary matrices (min, max) as such: `bounds = Bounds(min_bounds, max_bounds)`. 
Or by providing a previously declared bounds: `bounds = Bounds(bounds=another_bounds)`.
The `min_bounds` and `max_bounds` matrices must have the dimensions that fits the chosen `InterpolationType`, the default type being `InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT`, which is 3 columns.
If the interpolation type is CUSTOM, then the bounds are function handlers of signature: 
```python
custom_bound(current_shooting_point: int, n_elements: int, n_shooting: int)
```
where current_shooting_point is the current point to return, n_elements is the number of expected lines and n_shooting is the number of total shooting point (that is if current_shooting_point == n_shooting, this is the end of the phase)

The main methods the user will be interested in is the `min` property that returns the minimal bounds and the `max` property that returns the maximal bounds. 
Unless it is a custom function, `min` and `max` are numpy.ndarray and can be directly modified to change the boundaries. 
It is also possible to change `min` and `max` simultaneously by directly slicing the bounds as if it was a numpy.array, effectively defining an equality constraint: for instance `bounds[:, 0] = 0`. 
Finally, the `concatenate(another_bounds: Bounds)` method can be called to vertically concatenate multiple bounds.

### Class: BoundsList
A BoundsList is by essence simply a list of Bounds. 
The `add()` method can be called exacly as if one was calling the `Bounds` constructor. 
If the `add()` method is used more than one, the `phase` parameter is automatically incremented. 

So a minimal use is as follow:
```python
bounds_list = BoundsList
bounds_list.add(min_bounds, max_bounds)
```

### Class: QAndQDotBounds 
The QAndQDotBounds is simply a Bounds that uses a biorbd_model to define the miminal and maximal bounds for the generalized coordinates (*q*) and velocities (*qdot*). 
It is particularly useful when declaring the states bounds for *q* and *qdot*. 
Anything that was presented for Bounds, also apply to QAndQDotBounds


## The Initial conditions
The initial conditions the solver should start from, that is initial values of the states (x) and the controls (u).
In that sense, it is what is expected by the `OptimalControlProgram` for its `u_init` and `x_init` parameters. 

### Class InitialGuess

The InitialGuess class is the main class to define initial guesses.
The constructor can be call by sending one initial guess matrix (init) as such: `bounds = InitialGuess(init)`. 
The `init` matrix must have the dimensions that fits the chosen `InterpolationType`, the default type being `InterpolationType.CONSTANT`, which is 1 column.
If the interpolation type is CUSTOM, then the InitialGuess is a function handler of signature: 
```python
custom_bound(current_shooting_point: int, n_elements: int, n_shooting: int)
```
where current_shooting_point is the current point to return, n_elements is the number of expected lines and n_shooting is the number of total shooting point (that is if current_shooting_point == n_shooting, this is the end of the phase)

The main methods the user will be interested in is the `init` property that returns the initial guess. 
Unless it is a custom function, `init` is a numpy.ndarray and can be directly modified to change the initial guess. 
Finally, the `concatenate(another_initial_guess: InitialGuess)` method can be called to vertically concatenate multiple initial guesses.

### Class InitialGuessList
A InitialGuessList is by essence simply a list of InitialGuess. 
The `add()` method can be called exacly as if one was calling the `InitialGuess` constructor. 
If the `add()` method is used more than one, the `phase` parameter is automatically incremented. 

So a minimal use is as follow:
```python
init_list = InitialGuessList
init_list.add(init)
```


## The Constraints

### Class: ConstraintFcn

### Class: Constraint

### Class: ConstraintList

### Class: StateTransitionFcn

### Class: StateTransitionList

### Enum: Node



## The objective functions
TODO + Lagrange objective functions are integrated using rectangle method.

### Class: ObjectiveFcn

### Class: Objective, ObjectiveList

### Enum: Node



## The Parameters

### Class: ParameterList



## The OCP

### Class: OptimalControlProgram

### Class: NonLinearProgram

### Enum: OdeSolver
### Enum: Solver
### Enum: ControlType

### Enum: InterpolationType
The type of interpolation something is. 
- CONSTANT needs only one column, since it does not change over time
- CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT needs three columns. The first and last columns correspond to the first and last node, while the remaining interdiate node are treated as CONSTANT from the middle column
- LINEAR, needs two columns. It corresponds to the first and last node and is linearly interpolated in between.
- EACH_FRAME, needs as many as nodes. It is not an interpolation per se, but it allows the user to specify all the nodes individually.
- SPLINE, needs five columns. It performs a cubic spline to interpolate
- CUSTOM, user defined interpolation function



## The Results

### Class: Data

### Class: ShowResult
TODO + It is expected to slow down the optimization by about 15%

### Class: CustomPlot

### Class: ObjectivePrinter

### Class: Simulate

### Enum: Mapping

### Enum: PlotType











# Citing

If you use `bioptim`, we would be grateful if you could cite it as follows:

@misc{Michaud2020bioptim,
    author = {Michaud, Benjamin and Bailly, Francois and Begon, Mickael et al.},
    title = {bioptim, a Python interface for Musculoskeletal Optimal Control in Biomechanics},
    howpublished={Web page},
    url = {https://github.com/pyomeca/bioptim},
    year = {2020}
}
