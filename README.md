<p align="center">
    <img
      src="https://github.com/pyomeca/biorbd_design/blob/main/logo_png/bioptim_full.png"
      alt="logo"
    />
</p>

`Bioptim` is an optimal control program (OCP) framework for biomechanics. 
It is based on the efficient [biorbd](https://github.com/pyomeca/biorbd) biomechanics library and benefits from the powerful algorithmic diff provided by [CasADi](https://web.casadi.org/).
It interfaces the robust [`Ipopt`](https://github.com/coin-or/Ipopt) and the fast [`Acados`](https://github.com/acados/acados) solvers to suit all your needs for solving OCP in biomechanics. 

## Status

| Type | Status |
|---|---|
| License | <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a> |
| Continuous integration | [![Build status](https://github.com/pyomeca/bioptim/actions/workflows/run_tests.yml/badge.svg)](https://github.com/pyomeca/bioptim/actions) |
| Code coverage | [![codecov](https://codecov.io/gh/pyomeca/bioptim/branch/master/graph/badge.svg?token=NK1V6QE2CK)](https://codecov.io/gh/pyomeca/bioptim) |
| DOI | [![DOI](https://zenodo.org/badge/251615517.svg)](https://zenodo.org/badge/latestdoi/251615517) |

The current status of `bioptim` on conda-forge is

| Name | Downloads | Version | Platforms | MyBinder |
| --- | --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-bioptim-green.svg)](https://anaconda.org/conda-forge/bioptim) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/bioptim.svg)](https://anaconda.org/conda-forge/bioptim) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/bioptim.svg)](https://anaconda.org/conda-forge/bioptim) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/bioptim.svg)](https://anaconda.org/conda-forge/bioptim) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyomeca/bioptim-tutorial/HEAD?urlpath=lab) |

# Try bioptim
Anyone can play with bioptim with a working (but slightly limited in terms of graphics) MyBinder by clicking the following badge

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyomeca/bioptim-tutorial/HEAD?urlpath=lab)

As a tour guide that uses this binder, you can watch the `bioptim` workshop that we gave at the CMBBE conference on September 2021 by following this link:
[https://youtu.be/z7fhKoW1y60](https://youtu.be/z7fhKoW1y60)

# Table of Contents 





[Testing bioptim](#try-bioptim)
<details>
<summary><a href="#how-to-install">How to install</a></summary>

- [From anaconda](#installing-from-anaconda-for-windows-linux-and-mac)
- [From the sources](#installing-from-the-sources-for-linux-mac-and-windows)
- [Installation complete](#installation-complete)

</details>


[Defining our optimal control problems](#defining-our-optimal-control-problems)

<details>
<summary><a href="#a-first-practical-example">A first practical example</a></summary>

- [The import](#the-import)
- [Building the ocp](#building-the-ocp)
- [Solving the ocp](#solving-the-ocp)
- [Show the results](#show-the-results)
- [The full example files](#the-full-example-files)
- [Solving using multi-start](#solving-using-multi-start)

</details>



<details>
  <summary><a href="#a-more-in-depth-look-at-the-bioptim-api">A more in depth look at the `bioptim` API</a></summary>

  - <details>
    <summary><a href="#the-ocp">The OCP</a></summary>
  
    - [OptimalControlProgram](#class-optimalcontrolprogram)
    - [NonLinearProgram](#class-nonlinearprogram)

    </details>

  - <details>
    <summary><a href="#the-dynamics">The dynamics</a></summary>

    - [Dynamics](#class-dynamics)
    - [DynamicsList](#class-dynamicslist)
    - [DynamicsFcn](#class-dynamicsfcn)

    </details>

  - <details>
    <summary><a href="#the-bounds">The bounds</a></summary>

    - [Bounds](#class-bounds)
    - [BoundsList](#class-boundslist)

    </details>

  - <details>
    <summary><a href="#the-initial-conditions">The initial conditions</a></summary>

    - [InitialGuess](#class-initialguess)
    - [InitialGuessList](#class-initialguesslist)

    </details>

  - <details>
    <summary><a href="#the-variable-scaling">The variable scaling</a></summary>

    - [VariableScaling](#class-VariableScaling)
    - [VariableScalingList](#class-VariableScalinglist)

    </details>

  - <details>
    <summary><a href="#the-constraints">The constraints</a></summary>

    - [Constraint](#class-constraint)
    - [ConstraintList](#class-constraintlist)
    - [ConstraintFcn](#class-constraintfcn)

    </details>

  - <details>
    <summary><a href="#the-objective-functions">The objective functions</a></summary>

    - [Objective](#class-objective)
    - [ObjectiveList](#class-objectivelist)
    - [ObjectiveFcn](#class-objectivefcn)

    </details>

  - <details>
    <summary><a href="#the-parameters">The parameters</a></summary>

    - [ParameterList](#class-parameterlist)

    </details>

  - <details>
    <summary><a href="#the-multinode-constraints">The multinode constraints</a></summary>

    - [BinodeConstraintList](#class-binodeconstraintlist)
    - [BinodeConstraintFcn](#class-binodeconstraintfcn)

    </details>

  - <details>
    <summary><a href="#the-phase-transitions">The phase transitions</a></summary>

    - [PhaseTransitionList](#class-phasetransitionlist)
    - [PhaseTransitionFcn](#class-phasetransitionfcn)

    </details>

  - <details>
    <summary><a href="#the-results">The results</a></summary>

    - [Data manipulation](#data-manipulation)
    - [Data visualization](#data-visualization)

    </details>

  - <details>
    <summary><a href="#the-extra-stuff-and-the-enum">The extra stuff and the Enum</a></summary>

    - [The mappings](#the-mappings)
    - [Node](#enum-node)
    - [OdeSolver](#class-odesolver)
    - [Solver](#enum-solver)
    - [ControlType](#enum-controltype)
    - [PlotType](#enum-plottype)
    - [InterpolationType](#enum-interpolationtype)
    - [Shooting](#enum-shooting)
    - [CostType](#enum-costtype)
    - [SolutionIntegrator](#enum-solutionintegrator)
    - [QuadratureRule](#enum-quadraturerule)
    - [RigidBodyDynamics](#enum-rigidbodydynamics)
    - [SoftContactDynamics](#enum-softcontactdynamics)
    - [DefectType](#enum-defecttype)

    </details>

  </details>








        
<details>
<summary><a href="#examples">Examples</a></summary>

- [Run examples](#run-examples)
- [Getting started](#getting-started)
- [Muscle driven OCP](#muscle-driven-ocp)
- [Muscle driven with contact](#muscle-driven-with-contact)
- [Optimal time OCP](#optimal-time-ocp)
- [Symmetrical torque driven OCP](#symmetrical-torque-driven-ocp)
- [Torque driven OCP](#torque-driven-ocp)
- [Track](#track)
- [Moving estimation horizon](#moving-estimation-horizon)
- [Acados](#acados)
- [Inverse optimal control](#inverse-optimal-control)

</details>


<details>
<summary><a href="#performance">Performance</a></summary>

- [use_sx](#use_sx)
- [n_threads](#n_threads)
- [assume_phase_dynamics](#assume_phase_dynamics)
- [expand](#expand)

</details>


<details>
<summary><a href="#troubleshooting">Troubleshooting</a></summary>

- [freezing compute](#freezing-compute)
- [free variables](#free-variables)
- [non-converging problem](#non-converging-problem)

</details>



[Citing](#Citing)


# How to install 
The preferred way to install for the lay user is using anaconda. 
Another way, more designed for the core programmers, is from the sources. 
While it is theoretically possible to use `bioptim` from Windows, it is highly discouraged since it will require manually compiling all the dependencies. 
A great alternative for Windows users is *Ubuntu* on *Windows supporting Linux*.

## Installing from Anaconda (For Windows, Linux, and Mac)
The easiest way to install `bioptim` is to download the binaries from [Anaconda](https://anaconda.org/) repositories. 
The project is hosted on the conda-forge channel (https://anaconda.org/conda-forge/bioptim).

After having appropriately installed an anaconda client [my suggestion would be Miniconda (https://conda.io/miniconda.html)] and loaded the desired environment to install `bioptim` in, just type the following command:
```bash
conda install -c conda-forge bioptim
```
This will download and install all the dependencies and install `bioptim`. 
And that is it! 
You can already enjoy bioptiming!

## Installing from the sources (For Linux, Mac, and Windows)
Installing from the sources is as easy as installing from Anaconda, with the difference that you will be required to download and install the dependencies by hand (see the section below). 

### Dependencies
`bioptim` relies on several libraries. 
The most obvious one is the `biorbd` suite (including indeed `biorbd` and `bioviz`), but extra libraries are required.
Due to the different dependencies, it would be tedious to show how to install them all here. 
The user is therefore invited to read the relevant documentation. 

Here is a list of all direct dependencies (meaning that some dependencies may require other libraries themselves):  
[Python](https://www.python.org/) | [numpy](https://numpy.org/) | [scipy](https://scipy.org/) | [packaging](https://packaging.python.org/) | [setuptools](https://pypi.org/project/setuptools/)
| [matplotlib](https://matplotlib.org/) | [pandas](https://pandas.pydata.org/) | [pyomeca](https://github.com/pyomeca/pyomeca) | [CasADi](https://web.casadi.org/) | [rbdl-casadi](https://github.com/pyomeca/rbdl-casadi) compiled with the CasADi backend | [tinyxml](http://www.grinninglizard.com/tinyxmldocs/index.html) | [biorbd](https://github.com/pyomeca/biorbd) | [vtk](https://vtk.org/) | [PyQt](https://www.riverbankcomputing.com/software/pyqt) | [bioviz](https://github.com/pyomeca/bioviz) | [graphviz](https://graphviz.org/) | [`Ipopt`](https://github.com/coin-or/Ipopt) | [`Acados`](https://github.com/acados/acados) | [pyqtgraph](https://www.pyqtgraph.org/) | [pygmo](https://esa.github.io/pygmo2/) (only for inverse optimal control)  
and optionally: [The linear solvers from the HSL Mathematical Software Library](http://www.hsl.rl.ac.uk/index.html)

#### Linux - Installing dependencies with conda
All these (except for ̀`Acados` and the HSL lib) can easily be installed using (assuming the anaconda3 environment is loaded if needed) the `pip3` command or the Anaconda's following command:
```bash
conda install biorbd bioviz python-graphviz -cconda-forge
```
Since there is no `Anaconda` nor `pip3` package of `Acados`, a convenient installer is provided with `bioptim`. 
The installer can be found and run at `[ROOT_BIOPTIM]/external/acados_install_linux.sh`.
However, the installer requires an `Anaconda3` environment.
If you have an `Anaconda3` environment loaded, the installer should find itself where to install it. 
If you want to install it elsewhere, you can provide the script with a first argument which is the `$CONDA_PREFIX`. 
The second argument that can be passed to the script is the `$BLASFEO_TARGET`. 
If you don't know what it is, it is probably better to keep the default. 
Please note that depending on your computer architecture, `Acados` may or may not work correctly.

#### Mac - Installing dependencies with conda
Equivalently for MacOSX:
```bash
conda install casadi 'rbdl' 'biorbd' 'bioviz' python-graphviz -cconda-forge
```
Since there is no `Anaconda` nor `pip3` package of `Acados`, a convenient installer is provided with `bioptim`.
The `Acados` installation script is `[ROOT_BIOPTIM]/external/acados_install_mac.sh`.
However, the installer requires an `Anaconda3` environment.
If you have an `Anaconda3` environment loaded, the installer should find itself where to install it. 
If you want to install it elsewhere, you can provide the script with a first argument, the `$CONDA_PREFIX`. 
The second argument that can be passed to the script is the `$BLASFEO_TARGET`. 
If you don't know what it is, it is probably better to keep the default. 
Please note that depending on your computer architecture, `Acados` may or may not work correctly.

#### Windows - Installing dependencies with conda
Equivalently for Windows:
```bash
conda install casadi 'rbdl' 'biorbd' 'bioviz' python-graphviz -cconda-forge
```
There is no `Anaconda` nor `pip3` package of `Acados`.
To use the `Acados` solver on Windows, one must compile it themselves.

#### The case of HSL solvers
HSL is a collection of state-of-the-art packages for large-scale scientific computation. 
Among its best-known packages are those for the solution of sparse linear systems (`ma27`, `ma57`, etc.), compatible with ̀`Ipopt`.
HSL packages are [available](http://www.hsl.rl.ac.uk/download/coinhsl-archive-linux-x86_64/2014.01.17/) at no cost for academic research and teaching. 
Once you obtain the HSL dynamic library (precompiled `libhsl.so` for Linux, to be compiled `libhsl.dylib` for MacOSX, `libhsl.dll` for Windows), you just have to place it in your `Anaconda3` environment into the `lib/` folder.
You can now use all the options of `bioptim`, including the HSL linear solvers with `Ipopt`.
We recommend using `ma57` as a default linear solver by calling as such:
```python
solver = Solver.IPOPT()
solver.set_linear_solver("ma57")
ocp.solve(solver)
```
## Installation complete
Once `bioptim` is downloaded, navigate to the root folder and (assuming your conda environment is loaded if needed), you can type the following command:
```bash 
python setup.py install
```
Assuming everything went well, that is it! 
You can already enjoy bioptimizing!

# Defining our optimal control problems
Here we will detail our implementation of optimal control problems and some definitions.
The mathematical transcription of the OCP is as follows:
![](OCP_equation.jpg)
The optimization variables are the states (x = variables that represent the state of the system at each node and that 
are subject to continuity constraints), controls (u = decision variables defined at each node that drive the system),
algebraic states (s = optimization variables that are defined at each node but that are not subject to the 
built-in continuity constraints), and parameters (p = optimization variables defined once per phase).
The state continuity constraints implementation may vary depending on the transcription of the problem (implicit vs explicit, direct multiple shooting vs direct collocations).

The cost function can include Mayer terms (function evaluated at one node) and Lagrange terms (functions integrated over the duration of the phase).
The optimization variables can be subject to equality and/or inequality constraints.

# A first practical example
The easiest way to learn `bioptim` is to dive into it.
So let us do that and build our first optimal control program together.
Please note that this tutorial is designed to recreate the `examples/getting_started/pendulum.py` file where a pendulum is asked to start in a downward position and end, balanced, in an upward position while only being able to move sideways actively.

## The import
We will not spend time explaining the import since every one of them will be explained in detail later, and it is pretty straightforward anyway.
```python
from bioptim import (
  BiorbdModel,
  OptimalControlProgram,
  DynamicsFcn,
  Dynamics,
  BoundsList,
  InitialGuessList,
  ObjectiveFcn,
  Objective,
)
```

## Building the ocp
First of all, let us load a bioMod file using `biorbd`:
```python
bio_model = BiorbdModel("pendulum.bioMod")
```
It is convenient since it will provide interesting functions such as the number of degrees of freedom (`bio_model.nb_q`).
Please note that a `pendulum.bioMod` copy is available at the end of the *Getting started* section.
In brief, the pendulum consists of two degrees of freedom (sideways movement and rotation), with the center of mass near the head.

The dynamics of the pendulum, as for many biomechanical dynamics, is driven by the generalized forces. 
Generalized forces are forces or moments directly applied to the degrees of freedom as if virtual motors were driven them.
In `bioptim`, this dynamic is called torque driven. 
In a torque driven dynamics, the states are the positions (also called generalized coordinates, *q*) and the velocities (also called the generalized velocities, *qdot*), whereas the controls are the joint torques (also called generalized forces, *tau*). 
Let us define such dynamics:
```python
dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
```

The pendulum is required to start in a downward position (0 rad) and to finish in an upward position (3.14 rad) with no velocity at the start and end nodes.
To define that, it would be nice first to define boundary constraints on the position (*q*) and velocities (*qdot*) that match those in the bioMod file and to apply them at the very beginning, the very end, and all the intermediate nodes as well.
In this case, the state with index 0 is translation y, and index 1 refers to rotation about x. 
Finally, the index 2 and 3 are  the velocity of translation y and rotation about x,respectively.

bounds_from_ranges uses the ranges from a biorbd model and returns a structure with the minimal and maximal bounds for all the degrees of freedom and velocities on three columns corresponding to the starting, intermediate, and final nodes, respectively.
How convenient!
```python
x_bounds = BoundsList()
x_bounds["q"] = bio_model.bounds_from_ranges("q")
x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
```
The first dimension of x_bounds is the degrees of freedom (*q*) `and` their velocities (*qdot*) that match those `in` the bioMod `file`. The time `is` discretized `in` nodes which `is` the second dimension declared `in` x_bounds.
If you have more than one phase, we would have x_bound[*phase*][*q `and` qdot*, *nodes*]
In the first place, we want the first `and` last column (which `is` equivalent to nodes 0 `and` -1) to be 0, i.e., the translations `and` rotations to be null `for` both the position `and` so the velocities.
```python
x_bounds["q"][:, [0, -1]] = 0
x_bounds["qdot"][:, [0, -1]] = 0
```
Finally, override once again the final node for the rotation so it is upside down.
```python
x_bounds["q"][1, -1] = 3.14
```
At that point, you may want to have a look at the `x_bounds["q"].min` and `x_bounds["q"].max` matrices to convince yourself that the initial and final positions are prescribed and that all the intermediate points are free up to certain minimal and maximal values. 

Up to that point, nothing prevents the solver from simply using the virtual motor of the rotation to rotate the pendulum upward (like clock hands) to get to the upside-down rotation. 
What makes this example interesting is that we can prevent this by defining minimal and maximal bounds on the control (the maximal forces that these motors have)
```
u_bounds = BoundsList()
u_bounds["tau"] = [-100, 0], [100, 0]
```
Like this, the sideways force ranges from -100 Newton to 100 Newton, but the rotation force ranges from 0 N/m to 0 N/m.
Again, `u_bounds` is defined for the first, the intermediate, and the final nodes, but this time, we do not want to specify anything particular for the first and final nodes, so we can leave them as is. 

If you wondering where are defined *q*, *qdot* and *tau*, it is in the configuration of `DynamicsFcn.TORQUE_DRIVEN`. If you define a custom dynamics, then the variable's name should match those you define yourself.

Who says optimization says cost function.
Even though, it is possible to define an OCP without objective, it is not so much recommended, and let us face it... much less fun!
So the pendulum's goal (or the cost function) is to perform its task while using the minimum forces possible. 
Therefore, an objective function that minimizes the generalized forces is defined:
```python
objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)
```

At that point, it is possible to solve the program.
Still, helping the solver is usually a good idea, so let us give ̀`Ipopt` a starting point to investigate.
The initial guess that we can provide is those for the states (`x_init`, here *q* and *qdot*) and for the controls (`u_init`, here *tau*). 
So let us define both of them quickly
```python
x_init = InitialGuessList()
x_init["q"] = [0, 0]
x_init["qdot"] = [0, 0]

u_init = InitialGuessList()
u_init["tau"] = [0, 0]
```
Please note that initial guess is optional. The default value if a value is not provided is zero.

On the same train of thought, if we want to help the solver even more, we can also define a variable scaling for the states (`x_scaling`, here *q* and *qdot*) and for the controls (`u_scaling`, here *tau*). *Note that the scaling should be declared in the order in which the variables appear. 
We encourage you to choose a variable scaling the same order of magnitude to the expected optimal values.
```python
x_scaling = VariableScalingList()
x_scaling.add("q", scaling=[1, 3])
x_scaling.add("qdot", scaling=[85, 85])
   
u_scaling = VariableScalingList()
u_scaling.add("tau", scaling=[900, 1])
```
We now have everything to create the ocp!
For that, we have to decide how much time the pendulum has to get up there (`phase_time`) and how many shooting points are defined for the multishoot (`n_shooting`).
Thereafter, you have to send everything to the `OptimalControlProgram` class and let `bioptim` prepare everything for you.
For simplicity's sake, I copied all the pieces of code previously visited in the building of the ocp section here:
```python
ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting=25,
        phase_time=3,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        assume_phase_dynamics=True,
    )
```
The argument `assume_phase_dynamics` should be set to `True` if we assume the dynamics are the same within each phase of the ocp problem. 
This argument increases the speed to mount the problem; it should be considered each time you build an Optimal Control Program.
The default value is `False`, meaning we consider the dynamic equations are different for each shooting node (e.g., when applying a different external force at each shooting node).


## Checking the ocp
Now you can check if the ocp is well-defined for the initial values.
This checking will help see if your constraints and objectives are okay.
To visualize it, you can use
```python
ocp.check_conditioning()
```
This call will print two different plots!

The first shows the Jacobian matrix of constraints and the norm of each Hessian matrix of constraints.
There is one matrix for each phase.
The first half of the plot can be used to verify if some constraints are redundant. It simply compares the rank of the Jacobian with the number of constraints for each phase.
The second half of the plot can be used to verify if the equality constraints are linear.

The second plot window shows the hessian of the objective for each phase. It calculates if the problem can be convex by checking if the matrix is positive semi-definite.
It also calculates the condition number for each phase thanks to the eigenvalues.

If everything is okay, let us solve the ocp !

## Solving the ocp
It is now time to see `Ipopt` in action! 
To solve the ocp, you simply have to call the `solve()` method of the `ocp` class
```python
solver = Solver.IPOPT(show_online_optim=True)
sol = ocp.solve(solver)
```
If you feel fancy, you can even activate the online optimization graphs!
However, for such an easy problem, `Ipopt` will not leave you the time to appreciate the real-time updates of the graph...
For a more complicated problem, you may also wish to visualize the objectives and constraints during the optimization 
(useful when debugging, because who codes the right thing the first time). You can do it by calling
```python
ocp.add_plot_penalty(CostType.OBJECTIVES)
ocp.add_plot_penalty(CostType.CONSTRAINTS)
```
or alternatively asks for both at once using
```python
ocp.add_plot_penalty(CostType.ALL)
```
That's it!

## Show the results
If you want to look at the animated data, `bioptim` has an interface to `bioviz` designed to visualize bioMod files.
For that, simply call the `animate()` method of the solution:
```python
sol.animate()
```

If you did not fancy the online graphs but would enjoy them anyway, you can call the method `graphs()`:
```python
sol.graphs()
```

If you are interested in the results of individual objective functions and constraints, you can print them using the 
`print_cost()` or access them using the `detailed_cost_values()`:

```python
# sol.detailed_cost  # Invoke this for adding the details of the objectives to sol for later manipulations
sol.print_cost()  # For printing their values in the console
```

And that is all! 
You have completed your first optimal control program with `bioptim`! 

## Solving using multi-start
Due to the gradient descent methods, we can affirm that the optimal solution is a local minimum. However, it is impossible to know if a global minimum was found. For highly non-linear problems, there might exist a wide range of local 
optima. Solving the same problem with different initial guesses can be helpful to find the best local minimum or to 
compare the different optimal kinematics. It is possible to multi-start the problem by creating a multi-start object 
with `MultiStart()` and running it with its method `run()`.
An example of how to use multi-start is given in examples/getting_started/multi-start.py.


## The complete example files
If you did not completely follow (or were too lazy to!) you will find  the complete files described in the Getting started section here.
You will find that the file is a bit different from the `example/getting_started/pendulum.py`, but it is merely different on the surface.

### The pendulum.py file
```python
import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    BoundsList,    
    InitialGuessList,
    ObjectiveFcn,
    Objective,
)

bio_model = BiorbdModel("pendulum.bioMod")
dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

# Bounds are optional (default -inf -> inf)
x_bounds = BoundsList()
x_bounds["q"] = bio_model.bounds_from_ranges("q")
x_bounds["q"][:, [0, -1]] = 0
x_bounds["q"][1, -1] = 3.14
x_bounds["dot"] = bio_model.bounds_from_ranges("qdot")
x_bounds["qdot"][:, [0, -1]] = 0

u_bounds = BoundsList()
u_bounds["tau"] = [-100, 0], [100, 0]

objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)

# Initial guess is optional (default = 0)
x_init = InitialGuessList()
x_init["q"] = [0, 0]
x_init["qdot"] = [0, 0]
u_init = InitialGuessList()
u_init = [0, 0]

ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting=25,
        phase_time=3,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        assume_phase_dynamics=True,
    )
    
sol = ocp.solve(show_online_optim=True)
sol.print_cost()
sol.animate()
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

# A more in-depth look at the `bioptim` API

In this section, we will have an in-depth look at all the classes one can use to interact with the bioptim API. 
All the classes covered here can be imported using the command:
```python
from bioptim import ClassName
```

## The OCP
An optimal control program is an optimization that uses control variables to drive some state variables.
There are mainly two types of transcription methods: the `direct collocation` and the `direct multiple shooting`.
`Bioptim` is based on the latter. 
To summarize, it defines a large optimization problem by discretizing the control and the state variables into a predetermined number of intervals, the beginning of the interval being the shooting points.
By defining strict constraints between the end of an interval and the beginning of the next one, it can ensure proper dynamics of the system, ???->while having good insight to solve the problem using gradient descending algorithms.

### Class: OptimalControlProgram
This is the main class that holds an ocp. 
Most of the attributes and methods are for internal use; therefore the API user should not care much about them.
Once an OptimalControlProgram is constructed, it is usually ready to be solved.

The full signature of the `OptimalControlProgram` can be scary at first, but should become clear soon.
Here it is:

```python
OptimalControlProgram(
    bio_model: [list, BioModel],
    dynamics: [Dynamics, DynamicsList],
    n_shooting: [int, list],
    phase_time: [float, list], 
    x_bounds: BoundsList,
    u_bounds: BoundsList,
    x_init: InitialGuessList
    u_init: InitialGuessList,
    objective_functions: [Objective, ObjectiveList],
    constraints: [Constraint, ConstraintList],
    parameters: ParameterList,
    external_forces: list[list[Any | np.ndarray]],
    ode_solver: OdeSolver,
    control_type: [ControlType, list],
    all_generalized_mapping: BiMapping,
    q_mapping: BiMapping,
    qdot_mapping: BiMapping,
    tau_mapping: BiMapping,
    plot_mappings: Mapping,
    phase_transitions: PhaseTransitionList,
    n_threads: int,
    use_sx: bool,
    assume_phase_dynamics=False,
)
```
Of these, only the first four are mandatory.  
`bio_model` is the model loaded with classes such as BiorbdModel, MultiBiorbdModel, or a custom class. 
In the case of a multiphase optimization, one model per phase should be passed in a list.  
`dynamics` is the system's dynamics during each phase (see The dynamics section).  
`n_shooting` is the number of shooting points of the direct multiple shooting (method) for each phase.  
`phase_time` is the final time of each phase. If the time is free, this is the initial guess.  
`x_bounds` is the minimal and maximal value the states can have (see The bounds section)  .  
`u_bounds` is the minimal and maximal value the controls can have (see The bounds section).  
`x_init` is the initial guess for the states variables (see The initial conditions section).  
`u_init` is the initial guess for the controls variables (see The initial conditions section).  
`x_scaling` is the scaling applied to the states variables (see The variable scaling section).  
`xdot_scaling` is the scaling applied to the state derivative variables (see The variable scaling section).  
`u_scaling` is the scaling applied to the controls variables (see The variable scaling section).  
`objective_functions` is the objective function set of the ocp (see The objective functions section).  
`constraints` is the constraint set of the ocp (see The constraints section).  
`parameters` is the parameter set of the ocp (see The parameters section).  
`external_forces` are the external forces acting on the center of mass of the bodies. 
It is a list (one element for each phase) of np.ndarray of shape (6, i, n), where the 6 components are [Mx, My, Mz, Fx, Fy, Fz], for the ith force platform (defined by the externalforceindex) for each node n.  
`ode_solver` is the ode solver used to solve the dynamic equations.  
`control_type` is the type of discretization of the controls (usually CONSTANT) (see ControlType section).  
`all_generalized_mapping` is used to reduce the number of degrees of freedom by linking them (see The mappings section).
This one applies the same mapping to the generalized coordinates (*q*), velocities (*qdot*), and forces (*tau*).
`q_mapping` the mapping applied to *q*.  
`qdot_mapping` the mapping applied to *q_dot*.  
`tau_mapping` the mapping applied to *tau*.  
`plot_mappings` is to force some plots to be linked together.  
`n_threads` is to solve the optimization using multiple threads. 
This number is the number of threads to use.  
`use_sx` is if the CasADi graph should be constructed in SX. 
SX will tend to solve much faster than MX graphs, however they necessitate a huge amount of RAM.

Please note that a common ocp will usually define only these parameters:

```python
ocp = OptimalControlProgram(
    bio_model: [list, BioModel],
    dynamics: [Dynamics, DynamicsList],
    n_shooting: [int, list],
    phase_time: [float, list],
    x_init: InitialGuessList
    u_init: InitialGuessList, 
    x_bounds: BoundsList,
    u_bounds: BoundsList,
    objective_functions: [Objective, ObjectiveList],
    constraints: [Constraint, ConstraintList],
    n_threads: int,
)
```

The main methods one will be interested in are:
```python
ocp.update_objectives()
ocp.update_constraints()
ocp.update_parameters()
ocp.update_bounds()
ocp.update_initial_guess()
```
These allow to modify the ocp after being defined. 
It is advantageous when solving the ocp for the first time, then adjusting some parameters and reoptimizing afterward.

Moreover, the method 
```python
solution = ocp.solve(Solver)
```
is called to solve the ocp (the solution structure is discussed later). 
The `Solver` class can be used to select the nonlinear solver to solve the ocp:

- IPOPT
- ACADOS
- SQP method

Note that options can be passed to the solver parameter.
One can refer to their respective solver's documentation to know which options exist.
The `show_online_optim` parameter can be set to `True` so the graphs nicely update during the optimization.
It is expected to slow down the optimization a bit.

Finally, one can save and load previously optimized values by using
```python
ocp.save(solution, file_path)
ocp, solution = OptimalControlProgram.load(file_path)
```
IMPORTANT NOTICE: Please note that saved solution depends on the `bioptim` version used to create the .bo file, and retro-compatibility is NOT enforced.
In other words, an optimized solution from a previous version will probably NOT load on a newer `bioptim` version.
To save the solution in a way independent of the version of `bioptim`, one may use the `stand_alone` flag to `True`.

Finally, the `add_plot(name, update_function)` method can create new dynamics plots.
The name is simply the name of the figure.
If one with the same name already exists, the axes are merged.
The update_function is a function handler with signature: `update_function(states: np.ndarray, constrols: np.ndarray: parameters: np.ndarray) -> np.ndarray`.
It is expected to return a np.ndarray((n, 1)), where `n` is the number of elements to plot. 
The `axes_idx` parameter can be added to parse the data in a more exotic manner.
For instance, on a three-axes figure, if one wanted to plot the first value on the third axes and the second value on the first axes and nothing on the second, the `axes_idx=[2, 0]` would do the trick.
The interested user can have a look at the `examples/getting_started/custom_plotting.py` example.

### Class: NonLinearProgram
The NonLinearProgram is, by essence, the phase of an ocp. 
The user is expected not to change anything from this class but can retrieve valuable information from it.

One main use of nlp is to get a reference to the bio_model for the current phase: `nlp.model`.
Another essential value stored in nlp is the shape of the states and controls: `nlp.shape`, which is a dictionary where the keys are the names of the elements (for instance, *q* for the generalized coordinates)

It would be tedious, and probably not much useful, to list all the elements of nlp here.   
The interested user is invited to look at the docstrings for this class to get a detailed overview of it.

### Class: VariationalOptimalControlProgram
The `VariationalOptimalControlProgram` class inherits from `OptimalControlProgram` and is used to solve optimal control
problems using the variational approach. A variational integrator does the integration. The formulation being completely different from the other approaches, it needed its own class. The parameters are the same as in
`OptimalControlProgram` apart from the following changes:
- `bio_model` must be a `VariationalBiorbdModel`
- The phases have not been implemented yet; hence, only `final_time` must be specified, and it must be a float.
- There are no velocities in the variational approach, so you must only specify the `q_init` and not the `q_bounds`
instead of `x_init` and `x_bounds`.
- You can specify an initial guess for the velocities at the first node and the last node using `qdot_init` and
`qdot_bounds` and the keys must be `"qdot_start"` and `"qdot_end"`. These velocities are implemented as parameters of
the OCP, you can access them with `sol.parameters["qdot_start"]` and `sol.parameters["qdot_end"]` at the end of the
optimization.

## The model

Bioptim is designed to work with any model, as long as it inherits from the class `bioptim.Model`. Models built with `biorbd` are already compatible with `bioptim`.
They can be used as is or modified to add new features.

### Class: BiorbdModel

The `BiorbdModel` class implements a BioModel of the biorbd dynamics library. Some methods may not be interfaced yet; it is accessible through:
```python
bio_model = BiorbdModel("path/to/model.bioMod")
bio_model.marker_names  # for example returns the marker names
# if the methods is not interfaced, it can be accessed through
bio_model.model.markerNames()
```

### Class: MultiBiorbdModel

The `MultiBiorbdModel` class implements BioModel of multiple models of biorbd dynamics library. Some methods may not be interfaced yet; it is accessible through:
```python
bio_model = MultiBiorbdModel(("path/to/model.bioMod", "path/to/other/model.bioMod"))
```

### Class: HolonomicBiorbdModel
The `HolonomicBiorbdModel` class implements a BioModel of the biorbd dynamics library. Since the class inherits
from `BiorbdModel`, all the methods of `BiorbdModel` are available. You can define the
degrees of freedom (DoF) that are independent (that define the movement) and the ones that are dependent (that are
defined by the independent DoF and the holonomic constraint(s)). You can add some holonomic constraints to the model.
For this, you can use one of the functions of `HolonomicConstraintFcn` or add a custom one. You can refer to the
examples in `bioptim/examples/holonomic_constraints` to see how to use it.
Some methods may not be interfaced yet; it is accessible through:


```python
bio_model = HolonomicBiorbdModel("path/to/model.bioMod")
holonomic_constraints = HolonomicConstraintsList()
holonomic_constraints.add("holonomic_constraints", HolonomicConstraintsFcn.function, **kwargs)
bio_model.set_holonomic_configuration(holonomic_constraints, independent_joint_index, dependent_joint_index)
```
Two dynamics are implemented in the differential algebraic equations handling constraints at the acceleration level in
constrained_forward_dynamics(...). Moreover, the other was inspired by Robotran, which uses index reduction methods to satisfy
the constraints: partitioned_forward_dynamics(...)
### Class VariationalBiorbdModel
The `VariationalBiorbdModel` class implements a BioModel of the biorbd dynamics library. It is used in Discrete
Mechanic and Optimal Control (DMOC) and Discrete Mechanics and Optimal Control in Constrained Systems (DMOCC).
Since the class inherits from `HolonomicBiorbdModel`, all the `HolonomicBiorbdModel` and `BiorbdModel` methods are
available. This class is used in `VariationalOptimalControlProgram`. You can refer to the examples in
`bioptim/examples/discrete_mechanics_and_optimal_control` to see how to use it.
Some methods may not be interfaced yet; it is accessible through:

```python
bio_model = VariationalBiorbdModel("path/to/model.bioMod")
holonomic_constraints = HolonomicConstraintsList()
holonomic_constraints.add("holonomic_constraints", HolonomicConstraintsFcn.function, **kwargs)
bio_model.set_holonomic_configuration(holonomic_constraints)
VariationalOptimalControlProgram(bio_model, ...)
```

### Class: CustomModel

The `BioModel` class is the base class for BiorbdModel and any custom models.
The methods are abstracted and must be implemented in the child class,
or at least raise a `NotImplementedError` if they are not implemented. For example:
```python
from bioptim import Model

class MyModel(CustomModel, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        ...

    def name_dof(self):
        return ["dof1", "dof2", "dof3"]

    def marker_names(self):
        raise NotImplementedError
```

see the example [examples/custom_model/](https://github.com/pyomeca/bioptim/tree/master/bioptim/examples/custom_model) for more details.

## The dynamics
By essence, an optimal control program (ocp) links two types of variables: the states (x) and the controls (u). 
Conceptually, the controls are the driving inputs of the system, which participate in changing the system states. 
In the case of biomechanics, the states (*x*) are usually the generalized coordinates (*q*) and velocities (*qdot*), i.e., the pose of the musculoskeletal model and the joint velocities. 
On the other hand, the controls (*u*) can be the generalized forces, i.e., the joint torques, but can also be the muscle excitations, for instance.
States and controls are linked through Ordinary differential equations: dx/dt = f(x, u, p), where p can be additional parameters that act on the system but are not time-dependent.

The following section investigates how to instruct `bioptim` of the dynamic equations the system should follow.

 
### Class: Dynamics
This class is the main class to define a dynamics. 
It, therefore, contains all the information necessary to configure (i.e., determining which variables are states or controls) and perform the dynamics. 
When constructing an `OptimalControlProgram()`, Dynamics is the expected class for the `dynamics` parameter. 

The user can minimally define a Dynamics as: `dyn = Dynamics(DynamicsFcn)`.
The `DynamicsFcn` is the one presented in the corresponding section below. 

#### The options
The full signature of Dynamics is as follows:
```python
Dynamics(dynamics_type, configure: Callable, dynamic_function: Callable, phase: int)
```
The `dynamics_type` is the selected `DynamicsFcn`. 
It automatically defines both `configure` and `dynamic_function`. 
If a function is sent instead, this function is interpreted as `configure` and the DynamicsFcn is assumed to be `DynamicsFcn.CUSTOM`
If one is interested in changing the behavior of a particular `DynamicsFcn`, they can refer to the Custom dynamics functions right below. 

The `phase` is the index of the phase the dynamics applies to. 
The `add()` method of `DynamicsList` usually takes care of this, but it can be useful when declaring the dynamics out of order.

#### Custom dynamic functions
If an advanced user wants to define their own dynamic function, they can define the configuration and/or the dynamics. 

The configuration is what tells `bioptim` which variables are states and which are control.
The user is expected to provide a function handler with the following signature: `custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram)`.
In this function, the user is expected to call the relevant `ConfigureProblem` class methods: 
- `configure_q(nlp, as_states: bool, as_controls: bool)`
- `configure_qdot(nlp, as_states: bool, as_controls: bool)`
- `configure_q_qdot(nlp, as_states: bool, as_controls: bool)`
- `configure_tau(nlp, as_states: bool, as_controls: bool)`
- `configure_residual_tau(nlp, as_states: bool, as_controls: bool)`
- `configure_muscles(nlp, as_states: bool, as_controls: bool)`
where `as_states` add the variable to the states vector and `as_controls` to the controls vector.
Please note that this is not necessarily mutually exclusive.
Finally, the user is expected to configure the dynamic by calling `ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)`

Defining the dynamic function must be done when one provides a custom configuration, but it can also be defined by providing a function handler to the `dynamic_function` parameter for `Dynamics`. 
The signature of this custom dynamic function is as follows: `custom_dynamic(states: MX, controls: MX, parameters: MX, nlp: NonLinearProgram`.
This function is expected to return a tuple[MX] of the derivative of the states. 
Some methods defined in the class `DynamicsFunctions` can be useful, but will not be covered here since it is initially designed for internal use.
Please note that MX type is a CasADi type.
Anyone who wants to define custom dynamics should be at least familiar with this type beforehand. 


### Class: DynamicsList
A DynamicsList is simply a list of Dynamics. 
The `add()` method can be called exactly as if one was calling the `Dynamics` constructor. 
If the `add()` method is used more than one, the `phase` parameter is automatically incremented. 

So a minimal use is as follows:
```python
dyn_list = DynamicsList()
dyn_list.add(DynamicsFcn)
```

### Class: DynamicsFcn
The `DynamicsFcn` class is the configuration and declaration of all the already available dynamics in `bioptim`. 
Since this is an Enum, it is possible to use tab key on the keyboard to dynamically list them all, depending on the capabilities of your IDE. 

Please note that one can change the dynamic function associated to any of the configuration by providing a custom dynamics_function. 
For more information on this, please refer to the Dynamics and DynamicsList section right before. 

#### TORQUE_DRIVEN 
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as *tau*. 
The derivative of *q* is trivially *qdot*.
The derivative of *qdot* is given by the biorbd function: `qddot = bio_model.ForwardDynamics(q, qdot, tau)`.
If external forces are provided, they are added to the ForwardDynamics function. Possible options:
- **with_contact = True:** The derivative of *qdot* is given by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `qddot = bio_model.ForwardDynamicsConstraintsDirect(q, qdot, tau)`.
- **with_passive_torque = True:** The passive torque is taken into account in the *tau*.
- **with_ligament = True:** The tau generated by the ligament is taken into account in the *tau*.

#### TORQUE_DERIVATIVE_DRIVEN
The torque derivative driven defines the states (x) as *q*, *qdot*, *tau* and the controls (u) as *taudot*. 
The derivative of *q* is trivially *qdot*.
The derivative of *qdot* is given by the biorbd function: `qddot = bio_model.ForwardDynamics(q, qdot, tau)`.
The derivative of *tau* is trivially *taudot*. 
If external forces are provided, they are added to the ForwardDynamics function. Possible options:
- **with_contact = True:** The derivative of *qdot* is given by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `qddot = bio_model.ForwardDynamicsConstraintsDirect(q, qdot, tau)`.
- **with_passive_torque = True:** The passive torque is taken into account in the *tau*.
- **with_ligament = True:** The tau generated by the ligament is taken into account in the *tau*.

#### TORQUE_ACTIVATIONS_DRIVEN
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the level of activation of *tau*. 
The derivative of *q* is trivially *qdot*.
The actual *tau* is computed from the activation by the `biorbd` function: `tau = bio_model.torque(torque_act, q, qdot)`.
Then, the derivative of *qdot* is given by the `biorbd` function: `qddot = bio_model.ForwardDynamics(q, qdot, tau)`.

Please note, this dynamics is expected to be very slow to converge, if it ever does. 
One is therefore encourage using TORQUE_DRIVEN instead, and to add the TORQUE_MAX_FROM_ACTUATORS constraint.
This has been shown to be more efficient and allows defining minimum torque. Possible options:
- **with_contact = True:** The derivative of *qdot* is given by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `qddot = bio_model.ForwardDynamicsConstraintsDirect(q, qdot, tau)`.
- **with_passive_torque = True:** The passive torque is taken into account in the *tau*.
- **with_ligament = True:** The tau generated by the ligament is taken into account in the *tau*.
- **with_residual_torque = True:** The residual torque is taken into account in the *tau*.

#### JOINTS_ACCELERATION_DRIVEN
The joints acceleration driven defines the states (x) as *q* and *qdot* and the controls (u) as *qddot_joints*. The derivative of *q* is trivially *qdot*.
The joints' acceleration *qddot_joints* is the acceleration of the actual joints of the `biorb_model` without its root's joints.
The model's root's joints acceleration *qddot_root* are computed by the `biorbd` function: `qddot_root = boirbd_model.ForwardDynamicsFreeFloatingBase(q, qdot, qddot_joints)`.
The derivative of *qdot* is the vertical stack of *qddot_root* and *qddot_joints*.

This dynamic is suitable for bodies in free fall.

#### MUSCLE_DRIVEN
The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the muscle activations. 
The derivative of *q* is trivially *qdot*. Possible options:
The actual *tau* is computed from the muscle activation converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `bio_model.muscularJointTorque(muscles_states, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function: `qddot = bio_model.ForwardDynamics(q, qdot, tau)`.
- **with_contact = True:** The actual *tau* is computed from the sum of *tau* to the *a* converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `bio_model.muscularJointTorque(a, q, qdot)`.
The derivative of *qdot* is given by the `biorbd` function that includes non-acceleration contact point defined in the bioMod: `qddot = bio_model.ForwardDynamics(q, qdot, tau)`.
- **with_passive_torque = True:** The passive torque is taken into account in the *tau*.
- **with_ligament = True:** The tau generated by the ligament is taken into account in the *tau*.
- **with_residual_torque = True:** The torque driven defines the states (x) as *q* and *qdot* and the controls (u) as the *tau* and the muscle activations (*a*). 
The actual *tau* is computed from the sum of *tau* to the muscle activation converted in muscle forces and thereafter converted to *tau* by the `biorbd` function: `bio_model.muscularJointTorque(a, q, qdot)`.
- **with_excitations = True:** The torque driven defines the states (x) as *q*, *qdot* and muscle activations (*a*) and the controls (u) as the *tau* and the *EMG*.
The derivative of *a* is computed by the `biorbd` function: `adot = model.activationDot(emg, a)`


#### HOLOMOMIC_TORQUE_DRIVEN
This dynamics have been implemented to be used with `HolonomicBiorbdModel`. It is a torque driven only applied on the independent
degrees of freedom.



#### CUSTOM
This leaves the user to define both the configuration (what are the states and controls) and to define the dynamic function. 
CUSTOM should not be called by the user, but the user should pass the configure_function directly. 
You can have a look at Dynamics and DynamicsList sections for more information about how to configure and define custom dynamics.


## The bounds
The bounds provide a class that has minimal and maximal values for a variable.
It is, for instance, useful for the inequality constraints that limit the maximal and minimal values of the states (x) and the controls (u) .
In that sense, it is what is expected by the `OptimalControlProgram` for its `u_bounds` and `x_bounds` parameters. 
It can however be used for much more.
If not provided for one variable, then it is -infinity to +infinity for that particular variable.

### Class: BoundsList
The BoundsList class is the main class to define bounds.
The constructor can be called by sending two boundary matrices (min, max) as such: `bounds["name"] = min_bounds, max_bounds`. 
Or by providing a previously declared bounds: `bounds.add("name", another_bounds)`.
The add nomenclature can also be used with the min and max, but must be specified as such: `bounds.add("name", min_bound=min_bounds, max_bound=max_bounds)`.
The `min_bounds` and `max_bounds` matrices must have dimensions that fit the chosen `InterpolationType`, the default type being `InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT`, which is 3 columns.

Please note that to change any option, you must use the `.add` nomenclature

The full signature of BoundsList.add is as follows:
```python
BoundsList.add("name", bounds, min_bounds, max_bound, interpolation_type, phase)
```
The first parameters are presented before.
The `phase` is the index of the phase the bounds apply to. If you add twice the same element on the same phase, the first is then overrided.

If the interpolation type is CUSTOM, then the bounds are function handlers of signature: 
```python
custom_bound(current_shooting_point: int, n_elements: int, n_shooting: int)
```
where current_shooting_point is the current point to return, n_elements is the number of expected lines and n_shooting is the number of total shooting point (that is if current_shooting_point == n_shooting, this is the end of the phase)

The main methods the user will be interested in is the `min` property that returns the minimal bounds and the `max` property that returns the maximal bounds. 
Unless it is a custom function, `min` and `max` are numpy.ndarray and can be directly modified to change the boundaries. 
It is also possible to change `min` and `max` simultaneously by directly slicing the bounds as if it was a numpy.array, effectively defining an equality constraint: for instance `bounds["name"][:, 0] = 0`.
Please note that if more than one phase is present in the bounds, then you must specify on which phase it should apply like so: `bounds[phase_index]["name"]...`

## The initial conditions
The initial conditions the solver should start from, i.e., initial values of the states (x) and the controls (u).
In that sense, it is what is expected by the `OptimalControlProgram` for its `u_init` and `x_init` parameters. 
If not specified for one variable, then it is set to zero for that particular variable. 

### Class InitialGuessList

The InitialGuessList class is the main class to define initial guesses.
The `.add` can be called by sending one initial guess matrix (init) as such: `init["name"] = init`. 
The `init` matrix must have the dimensions that fits the chosen `InterpolationType`, the default type being `InterpolationType.CONSTANT`, which is 1 column.

The full signature of `InitialGuessList.add` is as follows:
```python
InitialGuessList.add("name", initial_guess, interpolation_type, phase)
```
The first parameters are presented before.
The `phase` is the index of the phase the initial guess applies to.

If the interpolation type is CUSTOM, then the InitialGuess is a function handler of signature: 
```python
custom_init(current_shooting_point: int, n_elements: int, n_shooting: int)
```
where current_shooting_point is the current point to return, n_elements is the number of expected lines and n_shooting is the number of total shooting point (that is if current_shooting_point == n_shooting, this is the end of the phase)

The main methods the user will be interested in is the `init` property that returns the initial guess. 
Unless it is a custom function, `init` is a numpy.ndarray and can be directly modified to change the initial guess. 

If someone wants to add noise to the initial guess, you can provide the following:
```python
init = init.add_noise(
    bounds: BoundsList, 
    magnitude: list | int | float | np.ndarray,
    magnitude_type: MagnitudeType, n_shooting: int, 
    bound_push: list | int | float, 
    seed: int
    )
```
The bounds must contain all the keys defined in the init list.
The parameters, except `MagnitudeType` must be specified for each phase unless you want the same value for every phases.


## The variable scaling
The scaling applied to the optimization variables, it is what is expected by the `OptimalControlProgram` for its `x_scaling`, `xdot_scaling` and `u_init` parameters. 

### Class VariableScalingList
A VariableScalingList is a list of VariableScaling. 
The `add()` method can be called exactly as if one was calling the `VariableScaling` constructor.  

So a minimal use is as follows:
```python
scaling = VariableScalingList()
scaling.add("q", scaling=[1, 1])
```

## The constraints
The constraints are hard penalties of the optimization program.
That means the solution won't be considered optimal unless all the constraint set is fully respected.
The constraints come in two format: equality and inequality. 

### Class: Constraint
The Constraint provides a class that prepares a constraint, so it can be added to the constraint set by `bioptim`.
When constructing an `OptimalControlProgram()`, Constraint is the expected class for the `constraint` parameter. 
It is also possible to later change the constraint by calling the method `update_constraints(the_constraint)` of the `OptimalControlProgram`

The Constraint class is the main class to define constraints.
The constructor can be called with the type of the constraint and the node to apply it to, as such: `constraint = Constraint(ConstraintFcn, node=Node.END)`. 
By default, the constraint will be an equality constraint equals to 0. 
To change this behaviour, one can add the parameters `min_bound` and `max_bound` to change the bounds to their desired values. 

The full signature of Constraint is as follows:
```python
Constraint(ConstraintFcn, node: node, index: list, phase: int, list_index: int, target: np.ndarray **extra_param)
```
The first parameters are presented before.
The `list` is the list of elements to keep. 
For instance, if one defines a TRACK_STATE constraint with `index=0`, then only the first state is tracked.
The default value is all the elements.
The `phase` is the index of the phase the constraint should apply to.
If it is not sent, phase=0 is assumed.
The `list_index` is the ith element of a list for a particular phase
This is usually taken care by the `add()` method of `ConstraintList`, but it can be useful when declaring the constraints out of order, or when overriding previously declared constraints using `update_constraints`.
The `target` is a value subtracted to the constraint value. 
It is useful to define tracking problems.
The dimensions of the target must be of [index, node]

The `ConstraintFcn` class provides a list of some predefined constraint functions. 
Since this is an Enum, it is possible to use tab key on the keyboard to dynamically list them all, assuming you IDE allows for it. 
It is possible however to define a custom constraint by sending a function handler in place of the `ConstraintFcn`.
The signature of this custom function is: `custom_function(pn: PenaltyController, **extra_params)`
The PenaltyController contains all the required information to act on the states and controls at all the nodes defined by `node`, while `**extra_params` are all the extra parameters sent to the `Constraint` constructor. 
The function is expected to return an MX vector of the constraint to be inside `min_bound` and `max_bound`. 
Please note that MX type is a CasADi type.
Anyone who wants to define custom constraint should be at least familiar with this type beforehand. 

### Class: ConstraintList
A ConstraintList is simply a list of Constraints. 
The `add()` method can be called exactly as calling the `Constraint` constructor. 
If the `add()` method is used more than once, the `list_index` parameter is automatically incremented for the prescribed `phase`.
If no `phase` is prescribed by the user, the first phase is assumed. 

So a minimal use is as follows:
```python
constraint_list = ConstraintList()
constraint_list.add(constraint)
```

### Class: ConstraintFcn
The `ConstraintFcn` class is the declaration of all the already available constraints in `bioptim`. 
Since this is an Enum, it is possible to use the tab key on the keyboard to dynamically list them all, depending on the capabilities of your IDE. The existing contraint functions in alphabetical order:
- **NON_SLIPPING**  &mdash; Adds a constraint of static friction at contact points constraining for small tangential forces.  
This constraint assumes that the normal forces is positive (that is having an additional TRACK_CONTACT_FORCES with `max_bound=np.inf`). The extra parameters `tangential_component_idx: int`, `normal_component_idx: int`, and `static_friction_coefficient: float` must be passed to the `Constraint` constructor.
- **PROPORTIONAL_CONTROL** &mdash; Links one control to another, such that `u[first_dof] - first_dof_intercept = coef * (u[second_dof] - second_dof_intercept)`. The extra parameters `first_dof: int` and `second_dof: int` must be passed to the `Constraint` constructor.
- **PROPORTIONAL_STATE** &mdash; Links one state to another, such that `x[first_dof] - first_dof_intercept = coef * (x[second_dof] - second_dof_intercept)`. The extra parameters `first_dof: int` and `second_dof: int` must be passed to the `Constraint` constructor.
- **SUPERIMPOSE_MARKERS** &mdash; Matches one marker with another one. The extra parameters `first_marker_idx: int` and `second_marker_idx: int` informs which markers are to be superimposed.
- **TIME_CONSTRAINT**  &mdash; Adds the time to the optimization variable set. It will leave the time free within the given boundaries.
- **TORQUE_MAX_FROM_ACTUATORS**  &mdash; Adds a constraint of maximal torque to the generalized forces controls such that the maximal *tau* are computed from the `biorbd` method `bio_model.torque_max(q, qdot)`. This is an efficient alternative to torque activation dynamics.  The extra parameter `min_torque` can be passed to ensure that the model is never too weak.
- **TRACK_ALL_CONTROLS**  &mdash; Tracks all the control variables toward a target.
- **TRACK_ANGULAR_MOMENTUM**  &mdash; Constraints the angular momentum in the global reference frame toward a target. The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be sent to specify the axes along which the momentum should be tracked.
- **TRACK_COM_POSITION**  &mdash; Constraints the center of mass toward a target. The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be sent to specify the axes along which the center of mass should be tracked.
- **TRACK_COM_VELOCITY**  &mdash; Constraints the center of mass velocity toward a target. The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be provided to specify the axes along which the velocity should be tracked.
- **TRACK_CONTACT_FORCES**  &mdash; Tracks the non-acceleration point reaction forces toward a target.
- **TRACK_LINEAR_MOMENTUM**  &mdash; Constraints the linear momentum toward a target. The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be sent to specify the axes along which the momentum should be tracked.
- **TRACK_MARKER_WITH_SEGMENT_AXIS**  &mdash; Tracks a marker using a segment, that is aligning an axis toward the marker. The extra parameters `marker_idx: int`, `segment_idx: int`, and `axis: Axis` must be passed to the `Constraint` constructor
- **TRACK_MARKERS_VELOCITY** &mdash; Tracks the skin marker velocities toward a target.
- **TRACK_MARKERS** &mdash; Tracks the skin markers toward a target. The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be provided to specify the axes along which the markers should be tracked.
- **TRACK_MUSCLES_CONTROL**  &mdash; Tracks the muscles (part of the control variables) toward a target.
- **TRACK_SEGMENT_WITH_CUSTOM_RT**  &mdash;Links a segment with an RT (for instance, an Inertial Measurement Unit). It does so by computing the homogenous transformation between the segment and the RT and then converting this to Euler angles. The extra parameters `segment_idx: int` and `rt_idx: int` must be passed to the `Constraint` constructor.
- **TRACK_STATE** &mdash; Tracks the state's variable toward a target.
- **TRACK_TORQUE**  &mdash; Tracks the generalized forces (part of the control variables) toward a target.
- **CUSTOM**  &mdash; The user should not directly send CUSTOM, but the user should pass the custom_constraint function directly. You can look at Constraint and ConstraintList sections for more information about how to define custom constraints.


## The objective functions
The objective functions are soft penalties of the optimization program.
In other words, the solution tries to minimize the value as much as possible but will not complain if the objective remains high.
The objective functions come in two formats: Lagrange and Mayer. 

The Lagrange objective functions are integrated over the whole phase (actually over the selected nodes, usually Node.ALL). 
One should note that integration is not given by the dynamics function but by the rectangle approximation over a node.

The Mayer objective functions are values at a single node, usually the Node.LAST. 

### Class: Objective
The Objective provides a class that prepares an objective function so that it can be added to the objective set by `bioptim`.
When constructing an `OptimalControlProgram()`, Objective is the expected class for the `objective_functions` parameter. 
It is also possible to later change the objective functions by calling the method `update_objectives(the_objective_function)` of the `OptimalControlProgram`

The Objective class is the main class to define objectives.
The constructor can be called with the type of the objective and the node to apply it to, as such: `objective = Objective(ObjectiveFcn, node=Node.END)`. 
Please note that `ObjectiveFcn` should either be a `ObjectiveFcn.Lagrange` or `ObjectiveFcn.Mayer`.

The full signature of Objective is as follows:
```python
Objective(ObjectiveFcn, node: Node, index: list, phase: int, list_index: int, quadratic: bool, target: np.ndarray, weight: float, **extra_param)
```
The first parameters are presented before.
The `list` is the list of elements to keep. 
When defining a MINIMIZE_STATE objective_function with `index=0`,  only the first state is minimized.
The default value is all the elements.
The `phase` is the index of the phase the objective function should apply to.
If it is not sent, phase=0 is assumed.
The `list_index` is the ith element of a list for a particular phase
This is usually taken care by the `add()` method of `ObjectiveList`, but it can be useful when declaring the objectives out of order or when overriding previously declared objectives using `update_objectives`.
`quadratic` defines if the objective function should be squared. 
This is particularly useful when minimizing toward 0 instead of minus infinity.
The `target` is a value subtracted from the objective value. 
It is relevant to define tracking problems.
The dimensions of the target must be of [index, node].
Finally, `weight` is the weighting that should be applied to the objective. 
The higher the weight is, the more important the objective is compared to the other objective functions.

The `ObjectiveFcn` class provides a list of some predefined objective functions. 
Since `ObjectiveFcn.Lagrange` and `ObjectiveFcn.Mayer` are Enum, it is possible to use tab key on the keyboard to dynamically list them all, assuming you IDE allows for it. 
It is possible, however, to define a custom objective function by sending a function handler in place of the `ObjectiveFcn`.
In this case, an additional parameter must be sent to the `Objective` constructor:  the `custom_type` with either `ObjectiveFcn.Lagrange` or `ObjectiveFcn.Mayer`.
The signature of the custom function is: `custom_function(pn: PenaltyController, **extra_params)`
The PenaltyController contains all the required information to act on the states and controls at all the nodes defined by `node`, while `**extra_params` are all the extra parameters sent to the `Objective` constructor. 
The function is expected to return an MX vector of the objective function. 
Please note that MX type is a CasADi type.
Anyone who wants to define custom objective functions should be at least familiar with this type beforehand. 

### Class: ObjectiveList
An ObjectiveList is a list of Objective. 
The `add()` method can be called exactly as calling the `Objective` constructor. 
If the `add()` method is used more than once, the `list_index` parameter is automatically incremented for the prescribed `phase`.
If no `phase` is prescribed by the user, the first phase is assumed. 

So a minimal use is as follows:
```python
objective_list = ObjectiveList()
objective_list.add(objective)
```

### Class: ObjectiveFcn
Here a list of objective function with its type (Lagrange and/or Mayer) in alphabetical order: 
- **MINIMIZE_ALL_CONTROLS** (Lagrange) &mdash; Minimizes all the control variables toward zero (or a target).
- **MINIMIZE_ANGULAR_MOMENTUM** (Lagrange and Mayer)  &mdash; Minimizes the angular momentum in the global reference frame toward zero (or a target). The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be provided to specify the axes along which the momentum should be minimized.
- **MINIMIZE_COM_ACCELERATION** (Lagrange and Mayer)  &mdash; Minimizes the center of mass acceleration towards zero (or a target). The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be provided to specify the axes along which the acceleration should be minimized.
- **MINIMIZE_COM_POSITION** (Lagrange and Mayer)  &mdash; Minimizes the center of mass position toward zero (or a target). The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be sent to specify the axes along which the center of mass should be minimized.
- **MINIMIZE_COM_VELOCITY**  (Lagrange and Mayer)  &mdash; Minimizes the center of mass velocity towards zero (or a target). The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be provided to specify the axes along which the velocity should be minimized.
- **MINIMIZE_CONTACT_FORCES** (Lagrange) &mdash; Minimizes the non-acceleration points of the reaction forces toward zero (or a target).
- **MINIMIZE_LINEAR_MOMENTUM** (Lagrange and Mayer)  &mdash; Minimizes the linear momentum towards zero (or a target). The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be provided to specify the axes along which the momentum should be minimized.
- **MINIMIZE_MARKERS_DISPLACEMENT** (Lagrange) &mdash; Minimizes the difference between a state at a node and the same state at the next node, effectively minimizing the velocity. The extra parameter `coordinates_system_idx` can be specified to compute the marker position in that coordinate system. Otherwise, it is computed in the global reference frame. 
- **MINIMIZE_MARKERS_VELOCITY or MINIMIZE_MARKERS_ACCELERATION** (Lagrange and Mayer) &mdash; Minimizes the marker velocities or accelerations toward zero (or a target).
- **MINIMIZE_MARKERS** (Lagrange and Mayer) &mdash; Minimizes the position of the markers toward zero (or a target). The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be sent to specify the axes along which the markers should be minimized.
- **MINIMIZE_MUSCLES_CONTROL** (Lagrange) &mdash;  Minimizes the muscles' controls (part of the control variables) toward zero (or a target).
- **MINIMIZE_PREDICTED_COM_HEIGHT** (Mayer)  &mdash; Minimizes the maximal height of the center of mass, predicted from the parabolic equation, assuming vertical axis is Z (2): CoM_dot[2]**2 / (2 * -g) + CoM[2]. To maximize a jump, one can use this function at the end of the push-off phase and declare a weight of -1.
- **MINIMIZE_SOFT_CONTACT_FORCES** (Lagrange) &mdash; Minimizes the external forces induced by soft contacts (or a target).
- **MINIMIZE_STATE_DERIVATIVE** (Lagrange) &mdash; Minimizes the difference between a state at a node and the same state at the next node, i.e., minimizes the generalized state derivative.
- **MINIMIZE_STATE** (Lagrange and Mayer) &mdash; Minimizes the state variable towards zero (or a target).
- **MINIMIZE_TIME** (Lagrange and Mayer) &mdash; Adds the time to the optimization variable set. It will minimize the time toward minus infinity or a target. If the Mayer term is used, `min_bound` and `max_bound` can also be defined.
- **MINIMIZE_TORQUE_DERIVATIVE** (Lagrange) &mdash; Minimizes the difference between a *tau* at a node and the same *tau* at the next node, i.e., minimizes the generalized forces derivative.
- **MINIMIZE_TORQUE** (Lagrange) &mdash; Minimizes the generalized forces (part of the control variables) toward zero (or a target).
- **PROPORTIONAL_CONTROL** (Lagrange) &mdash; Minimizes the difference between one control and another, such that `u[first_dof] - first_dof_intercept = coef * (u[second_dof] - second_dof_intercept)`. The extra parameters `first_dof: int` and `second_dof: int` must be passed to the `Objective` constructor.
- **PROPORTIONAL_STATE** (Lagrange and Mayer) &mdash; Minimizes the difference between one state and another, such that `x[first_dof] - first_dof_intercept = coef * (x[second_dof] - second_dof_intercept)`. The extra parameters `first_dof: int` and `second_dof: int` must be passed to the `Objective` constructor.
- **SUPERIMPOSE_MARKERS** (Lagrange and Mayer) &mdash; Tracks one marker with another one. The extra parameters `first_marker_idx: int` and `second_marker_idx: int` informs which markers are to be superimposed
- **TRACK_ALL_CONTROLS (Lagrange)** &mdash; Tracks all the control variables toward a target.
- **TRACK_CONTACT_FORCES** (Lagrange) &mdash; Tracks the non-acceleration points of the reaction forces toward a target.
- **TRACK_MARKER_WITH_SEGMENT_AXIS** (Lagrange and Mayer) &mdash; Minimizes the distance between a marker and an axis of a segment, that is aligning an axis toward the marker. The extra parameters `marker_idx: int`, `segment_idx: int` and `axis: Axis` must be passed to the `Objective` constructor
- **TRACK_MARKERS_VELOCITY or TRACK_MARKERS_ACCELERATION** (Lagrange and Mayer) &mdash;  Tracks the marker velocities or accelerations toward a target.
- **TRACK_MARKERS** (Lagrange and Mayer) &mdash; Tracks the skin markers towards a target. The extra parameter `axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z)` can be sent to specify the axes along which the markers should be tracked.
- **TRACK_MUSCLES_CONTROL** (Lagrange) &mdash; Tracks the muscles' controls (part of the control variables) toward a target.
- **TRACK_SEGMENT_WITH_CUSTOM_RT** (Lagrange and Mayer)  &mdash; Minimizes the distance between a segment and an RT (for instance, an Inertial Measurement Unit). It does so by computing the homogenous transformation between the segment and the RT and then converting this to Euler angles. The extra parameters `segment_idx: int` and `rt_idx: int` must be passed to the `Objective` constructor.
- **TRACK_SOFT_CONTACT_FORCES**  (Lagrange)  &mdash; Tracks the external forces induced by soft contacts toward a target.
- **TRACK_STATE**  (Lagrange and Mayer) &mdash; Tracks the state variable toward a target.
- **TRACK_TORQUE** (Lagrange &mdash; Tracks the generalized forces (part of the control variables) toward a target.
- **CUSTOM** (Lagrange and Mayer)  &mdash; The user should not directly send CUSTOM, but pass the custom_objective function directly. 
You can look at Objective and ObjectiveList sections for more information about defining custom objective function.


## The parameters
Parameters are time-independent variables (e.g., a muscle maximal isometric force, the value of gravity ). that affect the dynamics of the whole system. 
Due to the variety of parameters, it was impossible to provide predefined parameters but the time. 
Therefore, all the parameters are custom-made.

### Class: ParameterList
The ParameterList provides a class that prepares the parameters, so it can be added to the parameter set to optimize by `bioptim`.
When constructing an `OptimalControlProgram()`, ParameterList is the expected class for the `parameters` parameter. 
It is also possible to later change the parameters by calling the method `update_parameters(the_parameter_list)` of the `OptimalControlProgram`

The ParameterList class is the main class to define parameters.
Please note that, unlike other lists, `Parameter` is not accessible. This is for simplicity reasons, as it would complicate the API quite a bit to permit it.
Therefore, one should not call the Parameter constructor directly. 

Here is the full signature of the `add()` method of the `ParameterList`:
```python
ParameterList.add(parameter_name: str, function: Callable, initial_guess: InitialGuess, bounds: Bounds, size: int, phase: int, **extra_parameters)
```
The `parameter_name` is the parameter's name (reference for the output data as well).
The `function` is the function that modifies the biorbd model, it will be called just prior to applying the dynamics. 
The signature of the custom function is: `custom_function(BioModel, MX, **extra_parameters)`, where BiorbdModel is the model to apply the parameter to, the MX is the value the parameter will take, and the `**extra_parameters` are those sent to the add() method.
This function is expected to modify the bio_model, and not return anything.
Please note that MX type is a CasADi type.
Anyone who wants to define custom parameters should be at least familiar with this type beforehand.
The `initial_guess` is the initial value of the parameter.
The `bounds` are the maximal and minimal values of the parameter.
The `size` is the number of elements of this parameter.
If an objective function is provided, the return of the objective function should match the size.
The `phase` that the parameter applies to.
Even though a parameter is time-independent, one biorbd_model is loaded per phase. 
Since parameters are associated to a specific bio_model, one must define a parameter per phase.

## The multinode constraints
`Bioptim` can declare multiphase optimisation programs. The goal of a multiphase ocp is usually to handle changing dynamics. 
The user must understand that each phase is, therefore, a full ocp by itself, with constraints that links the end of which with the beginning of the following.

### Class: BinodeConstraintList
The BinodeConstraintList provides a class that prepares the binode constraints.
When constructing an `OptimalControlProgram()`, BinodeConstraintList is the expected class for the `binode_constraints` parameter. 

The BinodeConstraintList class is the main class to define parameters.
Please note that, unlike other lists, `BinodeConstraint` is not accessible since binode constraints do not make sense for single-phase ocp.
Therefore, one should not call the PhaseTransition constructor directly. 

Here is the full signature of the `add()` method of the `BinodeConstraintList`:
```python
BinodeConstraintList.add(BinodeConstraintFcn, phase_first_idx, phase_second_idx, first_node, second_node, **extra_parameters)
```
The `BinodeConstraintFcn` is binode constraints function to use.
The default is EQUALITY.
When declaring a custom transition phase, BinodeConstraintFcn is the function handler to the custom function.
The signature of the custom function is: `custom_function(binode_constraint:BinodeConstraint, nlp_pre: NonLinearProgram, nlp_post: NonLinearProgram, **extra_parameters)`,
where `nlp_pre` is the non linear program of the considered phase, `nlp_post` is the non linear program of the second considered phase, and the `**extra_parameters` are those sent to the add() method.
This function is expected to return the cost of the binode constraint computed in the form of an MX. Please note that MX type is a CasADi type.
Anyone who wants to define binode constraints should be at least familiar with this type beforehand.
The `phase_first_idx` is the index of the first phase. 
The `phase_second_idx` is the index of the second phase. 
The `first_node` is the first node considered. 
The `second_node` is the second node considered. 

### Class: BinodeConstraintFcn
The `BinodeConstraintFcn` class is the already available binode constraint in `bioptim`. 
Since this is an Enum, it is possible to use the tab key on the keyboard to dynamically list them all, depending on the capabilities of your IDE. 

- **EQUALITY**   &mdash; The states are equals.
- **COM_EQUALITY**   &mdash; The positions of centers of mass are equals.
- **COM_VELOCITY_EQUALITY**   &mdash; The velocities of centers of mass are equals.
- **CUSTOM**   &mdash; CUSTOM should not be directly sent by the user, but the user should pass the custom_transition function directly. 
You can look at the BinodeConstraintList section for more information about defining a custom transition function.

## The phase transitions
`Bioptim` can declare multiphase optimisation programs. 
The goal of a multiphase ocp is usually to handle changing dynamics. 
The user must understand that each phase is, therefore, a full ocp by itself, with constraints that links the end of which with the beginning of the following.
Due to some limitations created by using MX variables, some things can be done, and some cannot during a phase transition. 

### Class: PhaseTransitionList
The PhaseTransitionList provides a class that prepares the phase transitions.
When constructing an `OptimalControlProgram()`, PhaseTransitionList is the expected class for the `phase_transitions` parameter. 

The PhaseTransitionList class is the main class to define parameters.
Please note that, unlike other lists, `PhaseTransition` is not accessible since phase transition does not make sense for single-phase ocp.
Therefore, one should not call the PhaseTransition constructor directly. 

Here is the full signature of the `add()` method of the `PhaseTransitionList`:
```python
PhaseTransitionList.add(PhaseTransitionFcn, phase_pre_idx, **extra_parameters)
```
The `PhaseTransitionFcn` is the transition phase function to use.
The default is CONTINUOUS.
When declaring a custom transition phase,  PhaseTransitionFcn is the function handler to the custom function.
The signature of the custom function is: `custom_function(transition: PhaseTransition nlp_pre: NonLinearProgram, nlp_post: NonLinearProgram, **extra_parameters)`,
where `nlp_pre` is the nonlinear program at the end of the phase before the transition, `nlp_post` is the nonlinear program at the beginning of the phase after the transition, and the `**extra_parameters` are those sent to the add() method.
This function is expected to return the cost of the phase transition computed from the states pre- and post-transition in the form of an MX.
Please note that MX type is a CasADi type.
Anyone who wants to define phase transitions should be at least familiar with this type beforehand.
The `phase_pre_idx` is the index of the phase before the transition.
If the `phase_pre_idx` is set to the index of the last phase, then this is equivalent to set `PhaseTransitionFcn.CYCLIC`.  

### Class: PhaseTransitionFcn
The `PhaseTransitionFcn` class is the already available phase transitions in `bioptim`. 
Since this is an Enum, it is possible to use the tab key on the keyboard to dynamically list them all, depending on the capabilities of your IDE. 

- **CONTINUOUS**  &mdash; The states at the end of the phase_pre equals the states at the beginning of the phase_post
- **IMPACT**   &mdash; The impulse function of `biorbd`: `qdot_post = bio_model.qdot_from_impact, q_pre, qdot_pre)` is applied to compute the velocities of the joint post impact.
These computed states at the end of the phase_pre equals those at the beginning of the phase_post.
If a bioMod has more contact points than the model in the previous phase, then the IMPACT transition phase should also be used.
- **CYCLIC** &mdash; Apply the CONTINUOUS phase transition from the end of the last phase to the beginning the first one, effectively creating a cyclic movement.
- **CUSTOM** &mdash; the user should not send CUSTOM directly but pass the custom_transition function. 
You can look at the PhaseTransitionList section for more information about defining a custom transition function.

## The results
`Bioptim` offers different ways to manage and visualize the results from an optimization. 
This section explores the different methods that can be called to have a look at your data.

Everything related to managing the results can be accessed from the solution class returned from 
```python
sol = ocp.solve()
```

### Data manipulation
The Solution structure holds all the optimized values. 
To get the states variable, control variables, and time, one can invoke each property.

```python
states = sol.states
controls = sol.controls
time = sol.time
```

If the program was a single-phase problem, then the returned values are dictionaries, otherwise, it is a list of dictionaries of size equal to the number of phases.
The keys of the returned dictionaries correspond to the name of the variables. 
For instance, if generalized coordinates (*q*) are states, the state dictionary has *q* as key.
In any case, the key `all` is always there.

```python
# single-phase case
q = sol.states["q"]  # generalized coordinates
q = sol.states["all"]  # all states
# multiple-phase case - states of the first phase
q = sol.states[0]["q"]
q = sol.states[0]["all"]
```

The values inside the dictionaries are np.ndarray of dimension `n_elements` x `n_shooting`, unless the data were previously altered by integrating or interpolating (then the number of columns may differ).

The parameters are very similar but differ because they are always a dictionary (since they do not depend on the phases).
Also, the values inside the dictionaries are of dimension `n_elements` x 1. 

#### Integrate

It is possible to integrate (also called simulate) the states at will by calling the `sol.integrate()` method.
The `shooting_type: Shooting` parameter allows you to select the type of integration to perform (see the enum Shooting for more detail).
The `keep_intermediate_points` parameter allows us to keep the intermediate shooting points (usually a multiple of n_steps of the Runge-Kutta) or collocation points.
If set to false, these points are not stored in the output structure.
By definition, setting `keep_intermediate_points` to True while asking for `Shooting.MULTIPLE` would return the same structure.
This will therefore raise an error if set to False with `Shooting.MULTIPLE`.
The `merge_phase: bool` parameter requests to merge all the phases into one [True] or not [False].
The `continuous: bool` parameter can be deceiving. It is mostly for internal purposes.

Here are the tables of the combinations for `sol.integrate` and shooting_types.
As the argument `keep_intermediates_points` does not significantly affect the implementations, it has been withdrawn from the tables.
If implemented, it will be done with `keep_intermediates_points=True or False`.

Let us begin with `shooting_type = Shooting.SINGLE`, it re-integrates the ocp as a single phase ocp :

##### Shooting.SINGLE

OdeSolver | <div style="width:110px">merge_phase</div> | <div style="width:80px">Solution<br>Integrator</div> | Implemented | Comment|
----|-------------|-----------|:----:|:-----------:|
DMS | True  | OCP | :white_check_mark: | |
DMS | False | OCP | :white_check_mark: | |
DMS | True  | SCIPY | :white_check_mark: | |
DMS | False | SCIPY | :white_check_mark: | |
COLLOCATION | True | OCP | :x: | COLLOCATION Solvers cannot be used with single shooting|
COLLOCATION | False |  OCP |  :x: | COLLOCATION Solvers cannot be used with single shooting|
COLLOCATION | True | SCIPY | :white_check_mark: | |
COLLOCATION | False | SCIPY | :white_check_mark: | |

##### Shooting.SINGLE_DISCONTINUOUS_PHASES
Let's pursue with `shooting_type = Shooting.SINGLE_DISCONTINUOUS_PHASES`, it re-integrates each phase of the ocp as a single phase ocp.
Thus, SINGLE and SINGLE_DISCONTINUOUS_PHASES are equivalent if there is only one phase. Here is the table:

OdeSolver | <div style="width:110px">merge_phase</div> |  <div style="width:80px">Solution<br>Integrator</div> | Implemented | Comment|
----|-------------|-----------|:----:|:-----------:|
DMS | True | OCP | :white_check_mark: | |
DMS | False | OCP | :white_check_mark: | |
DMS | True | SCIPY | :white_check_mark: | |
DMS | False | SCIPY | :white_check_mark: | |
COLLOCATION | True | OCP | :x: | COLLOCATION Solvers cannot be used with single shooting|
COLLOCATION | False | OCP | :x: | COLLOCATION Solvers cannot be used with single shooting|
COLLOCATION | True | SCIPY | :white_check_mark: |
COLLOCATION | False | SCIPY | :white_check_mark: | |

##### Shooting.MULTIPLE

Let us finish with `shooting_type = Shooting.MULTIPLE`,
please note that this cannot be used with `keep_intermediates_points=False`.
Also, the word `MULTIPLE` refers to direct multiple shooting.

OdeSolver | <div style="width:110px">merge_phase</div>  | <div style="width:80px">Solution<br>Integrator</div> | Implemented | Comment|
----|-------------|-----------|:----:|:-----------:|
DMS | True | OCP | :white_check_mark: | |
DMS | False | OCP | :white_check_mark: | |
DMS | True | SCIPY | :white_check_mark: | |
DMS | False | SCIPY | :white_check_mark: | |
COLLOCATION | True | OCP | :x: | The solution cannot be re-integrated with the ocp solver|
COLLOCATION | False | OCP | :x: | The solution cannot be re-integrated with the ocp solver|
COLLOCATION | True | SCIPY  | :white_check_mark: | This is re-integrated with solve_ivp, as direct multiple shooting problem |
COLLOCATION | False | SCIPY | :white_check_mark: | This is re-integrated with solve_ivp, as direct multiple shooting problem |

#### Interpolation

The `sol.interpolation(n_frames: [int, tuple])` method returns the states interpolated by changing the number of shooting points.
If the program is multiphase, but only a `int` is sent, then the phases are merged, and the interpolation keeps their respective time ratio consistent.
If one does not want to merge the phases, then a `tuple` with one value per phase can be sent. 

#### Merge phases

Finally, `sol.merge_phases()` returns a Solution structure with all the phases merged into one.

Please note that, apart from `sol.merge_phases()`, these data manipulation methods return an incomplete Solution structure.
This structure can be used for further analyses but cannot be used for visualization. 
If one wants to visualize integrated or interpolated data, they must use the corresponding parameters or the visualization method they use.

### Data visualization
The first data visualizing method is `sol.graphs()`. 
This method will spawn all the graphs associated with the ocp. 
This is the same method that is called by the online plotter. 
To add and modify plots, one should use the `ocp.add_plot()` method.
By default, this graphs the states as multiple shootings.
If one wants to simulate in single shooting, the option `shooting_type=Shooting.SINGLE` will do the trick.

A second one is `sol.animate()`.
This method summons one or more `bioviz` figures (depending on whether phases were merged) and animates the model.
Please note that despite `bioviz` best efforts, plotting a lot of meshing vertices in MX format is slow.
So even though it is possible, it is suggested to animate without the bone meshing (by passing the parameter `show_meshes=False`)
To do so, we strongly suggest saving the data and loading them in an environment where `bioptim` is compiled with the Eigen backend, which will be much more efficient.
If `n_frames` is set, an interpolation is performed. Otherwise, the phases are merged if possible, so a single animation is shown. 
To prevent phase merging, one can set `n_frames=-1`.

In order to print the values of the objective functions and constraints, one can use the `sol.print_cost()` method.
If the parameter `cost_type=CostType.OBJECTIVE` is passed, only the values of each objective functions are printed.
The same is true for the constraints with `CostType.CONSTRAINTS`.
Please note that for readability purposes, this method prints the sum by phases for the constraints. 

## The extra stuff and the Enum
It was hard to categorize the remaining classes and enum. 
So I present them in bulk in this extra stuff section.

### The mappings
The mapping is a way to link things stored in a list.
For instance, consider these vectors: a = [0, 0, 0, 10, -9] and b = [10, 9]. 
Even though they are quite different, they share some common values. 
It is, therefore, possible to retrieve a from b, and conversely.

This is what the Mapping class does for the rows of numpy arrays.
So if one was to declare the following Mapping: `b_from_a = Mapping([3, -4])`.
Then, assuming a is a numpy.ndarray column vector (`a = np.array([a]).T`), it would be possible to summon b from a like so: 
```python
b = b_from_a.map(a)
```
Note that the `-4` opposed the fourth value.
Conversely, using the `a_from_b = Mapping([None, None, None, 0, -1])` mapping, and assuming b is a numpy.ndarray column vector (`b = np.array([b]).T`), it would be possible to summon b from a like so:
```python
a = a_from_b.map(b)
```
Note that the `None` are replaced by zeros.

The BiMapping is no more no less than a list of two mappings that link two matrices both ways: `BiMapping(a_to_b, b_to_a)`

The SelectionMapping is a subclass of BiMapping where you only have to precise the size of the first matrix, 
and the mapping b_to_a to get the second matrix from the first. If some elements depend on others, 
you can add an argument dependency:`SelectionMapping(size(int), b_to_a; tuple[int, int, ...], dependencies :tuple([int, int, bool]))`

### Enum: Node
The node targets some specific nodes of the ocp or a phase.
The accepted values are:
- START: The first node
- MID: The middle node
- INTERMEDIATES: All the nodes but the first and the last one
- PENULTIMATE: The second to last node of the phase
- END: The last node
- ALL: All the nodes
- TRANSITION: The last node of a phase and the first node of the next phase

### Class: OdeSolver
The ordinary differential equation (ode) solver to solve the dynamics of the system. 
The RK4 and RK8 are the ones with the most options available.
IRK may be more robust but slower . 
CVODES is the one with the least options since it is not in-house implemented.

The accepted values are:
- For Direct multiple shooting:
   - RK1: Runge-Kutta of the 1st order also known as Forward Euler
   - RK2: Runge-Kutta of the 2nd order also known as Midpoint Euler
   - RK4: Runge-Kutta of the 4th order
   - RK8: Runge-Kutta of the 8th order
   - IRK: Implicit Runge-Kutta (Legendre and Radau, from 0th to 9th order)
   - CVODES: cvodes solver
- For Direct collocation:
   - COLLOCATION: Legendre and Radau, from 0th to 9th order

### Enum: Solver
The nonlinear solver to solve the whole ocp. 
Each solver has some requirements (for instance, ̀`Acados` necessitates that the graph is SX). 
Feel free to test each of them to see which fits your needs best.
̀`Ipopt` is a robust solver, that may be slow.
̀`Acados`, on the other hand, is a very fast solver, but is much more sensitive to the relative weightings of the objective functions and the initial guess.
It is perfectly designed for MHE and NMPC problems.

The accepted values are:
- ̀`Ipopt`
- ̀`Acados`
- ̀`SQP`

### Enum: ControlType
The type the controls are. 
Typically, the controls for an optimal control program are constant over the shooting intervals. 
However, one may want to get non-constant values.
`Bioptim` has therefore implemented some other types of controls.

The accepted values are:
- CONSTANT: The controls remain constant over the interval. The number of control is, therefore, equal to the number of shooting points.
- LINEAR_CONTINUOUS: The controls are linearly interpolated over the interval. Since they are continuous, the end of an interval corresponds to the beginning of the next. The number of control equals the number shooting point + 1.

### Enum: PlotType
When adding a plot, it is possible to change the aspect of it.

The accepted values are:
PLOT: Normal plot that links the points.
INTEGRATED: Plot that links the points within an interval but is discrete between its end and the beginning of the next interval.
STEP: Step plot, constant over an interval.
POINT: Point plot.

### Enum: InterpolationType
Defines wow a time-dependent variable is interpolated.
It is mainly used for phases time span.
Therefore, first and last nodes refer to the first and last nodes of a phase.

The accepted values are:
- CONSTANT: Requires only one column; all the values are equal during the whole period of time.
- CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT: Requires three columns. The first and last columns correspond to the first and last nodes, while the middle corresponds to all the other nodes.
- LINEAR: Requires two columns. It corresponds to the first and last nodes. The middle nodes are linearly interpolated to get their values.
- EACH_FRAME: Requires as many columns as there are nodes. It is not an interpolation per se, but it allows the user to specify all the nodes individually.
- ALL_POINTS: Requires as many columns as there are collocation points. It is not an interpolation per se, but it allows the user to specify all the collocation points individually.
- SPLINE: Requires five columns. It performs a cubic spline to interpolate between the nodes.
- CUSTOM: User-defined interpolation function.

### Enum: MagnitudeType
The type of magnitude you want for the added noise. Either relative to the bounds (0 is no noise, 1 is the value of your bounds), or absolute

The accepted values are:
- ABSOLUTE: Absolute noise of a chosen magnitude.
- RELATIVE: Relative noise to the bounds (0 is no noise, 1 is the value of your bounds).

### Enum: Shooting
The type of integration to perform
- SINGLE: It re-integrates the solution as a single-phase optimal control problem
- SINGLE_DISCONTINUOUS_PHASE: It re-integrates each phase of the solution as a single-phase optimal control problem. The phases are, therefore, not continuous.
- MULTIPLE: The word `MULTIPLE` is used as a common terminology to be able to execute DMS and COLLOCATION. It refers to the fact that there are several points per interval, shooting points in DMS and collocation points in COLLOCATION.

### Enum: CostType
The type of cost
- OBJECTIVES: The objective functions
- CONSTRAINTS: The constraints
- ALL: All the previously described cost type

### Enum: SolutionIntegrator
The type of integrator used to integrate the solution of the optimal control problem
- OCP: The OCP integrator initially chosen with [OdeSolver](#class-odesolver)
- SCIPY_RK23: The scipy integrator RK23
- SCIPY_RK45: The scipy integrator RK45
- SCIPY_DOP853: The scipy integrator DOP853
- SCIPY_BDF: The scipy integrator BDF
- SCIPY_LSODA: The scipy integrator LSODA

### Enum: QuadratureRule
The type of integration used to integrate the cost function terms of Lagrange:
- RECTANGLE_LEFT: The integral is approximated by a left rectangle rule (Left Riemann sum).
- RECTANGLE_RIGHT: The integral is approximated by a right rectangle rule (Right Riemann sum).
- MIDPOINT: The integral is approximated by a midpoint rectangle rule (Midpoint Riemann sum).
- APPROXIMATE_TRAPEZOIDAL: The integral is approximated by a trapezoidal rule using the state at the beginning of the next interval.
- TRAPEZOIDAL: The integral is approximated by a trapezoidal rule using the state at the end of the current interval.

### Enum: RigidBodyDynamics
The type of transcription of any dynamics (e.g., rigidbody_dynamics or soft_contact_dynamics):
- ODE: the dynamics is handled explicitly in the continuity constraint of the ordinary differential equation of the Direct Multiple Shooting approach.
- DAE_INVERSE_DYNAMICS: it adds an extra control *qddot* to respect inverse dynamics on nodes; this is a DAE-constrained OCP.
- DAE_FORWARD_DYNAMICS: it adds an extra control *qddot* to respect forward dynamics on nodes; this is a DAE-constrained OCP.
- DAE_INVERSE_DYNAMICS_JERK: it adds an extra control *qdddot* and an extra state *qddot* to respect inverse dynamics on nodes; this is a DAE-constrained OCP.
- DAE_FORWARD_DYNAMICS_JERK: it adds an extra control *qdddot* and an extra state *qddot* to respect forward dynamics on nodes; this is a DAE-constrained OCP.

### Enum: SoftContactDynamics
The type of transcription of any dynamics (e.g., rigidbody_dynamics or soft_contact_dynamics):
- ODE: soft contact dynamics is handled explicitly.
- CONSTRAINT: an extra control *fext* is added, and it ensures respecting soft contact_dynamics on nodes through a constraint.

### Enum: DefectType
- EXPLICIT: The defect comes from the explicit formulation.
- IMPLICIT: The defect comes from the implicit formulation.
- NOT_APPLICABLE: The defect is not applicable.

# Examples
In this section, we describe all the examples implemented with bioptim. They are ordered in separate files. Each subsection corresponds to the different files, dealing with different examples and topics.
Please note that the examples from the paper (see [Citing](#citing)) can be found in this repo
[https://github.com/s2mLab/BioptimPaperExamples](https://github.com/s2mLab/BioptimPaperExamples).

## Run examples
A GUI to access the examples can be run to facilitate the testing of bioptim
You can run the file `__main__.py` in the `examples` folder or execute the following command.
```bash
python -m bioptim.examples
```
Please note that `pyqtgraph` must be installed to run this GUI. 




## Getting started
In this subsection, all the examples of the getting_started file are described.

### The [custom_bounds.py](./bioptim/examples/getting_started/custom_bounds.py) file
This example is a trivial box sent upward. It is designed to investigate the different bounds defined in bioptim.
Therefore, it shows how to define the bounds, i.e., the minimal and maximal values of the state and control variables.

All the types of interpolation are shown: `CONSTANT`, `CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT`, `LINEAR`, `EACH_FRAME`,
`SPLINE`, and `CUSTOM`. 

When the `CUSTOM` interpolation is chosen, the functions `custom_x_bounds_min` and `custom_x_bounds_max`  
provide custom x bounds. The functions `custom_u_bounds_min` and `custom_u_bounds_max` provide custom 
u bounds. 
In this particular example,  linear interpolation is mimicked using these four functions.

### The [custom_constraints.py](./bioptim/examples/getting_started/custom_constraints.py) file
This example is a trivial box that must superimpose one of its corners on a marker at the beginning of the movement and superimpose the same corner on a different marker at the end.
It is designed to show how to define custom constraints function if the available constraints do not fulfill your need.

This example reproduces the behavior of the `SUPERIMPOSE_MARKERS` constraint.

### The [custom_dynamics.py](./bioptim/examples/getting_started/custom_dynamics.py) file
This example is a trivial box that must superimpose one of its corners on a marker at the beginning of the movement
and superimpose the same corner on a different marker at the end.
It is designed to show how to define a custom dynamics function if the provided ones are not 
sufficient.

This example reproduces the behavior of the `DynamicsFcn.TORQUE_DRIVEN` using custom dynamics. 

The custom_dynamic function is used to provide the derivative of the states. The custom_configure function is used 
to tell the program which variables are states and controls. 

### The [custom_initial_guess.py](./bioptim/examples/getting_started/custom_initial_guess.py) file
This example is a trivial box that must superimpose one of its corners on a marker at the beginning of the movement
and superimpose the same corner on a different marker at the end.
It is designed to investigate the different ways to define the initial guesses at each node sent to the solver.

All the types of interpolation are shown: `CONSTANT`, `CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT`, `LINEAR`, `EACH_FRAME`,
`SPLINE`, and `CUSTOM`. 

When the CUSTOM interpolation is chosen, the `custom_init_func` function is used to custom the initial guesses of the 
states and controls. In this particular example, the CUSTOM interpolation mimics linear interpolation. 

### The [custom_objectives.py](./bioptim/examples/getting_started/custom_objectives.py) file
This example is a trivial box that tries to superimpose one of its corners on a marker at the beginning of the movement
and superimpose the same corner on a different marker at the end.
It is designed to show how one can define its own custom objective function if the provided ones are not
sufficient.

This example reproduces the behavior of the `Mayer.SUPERIMPOSE_MARKERS` objective function. 

This example is close to the example of the custom_constraint.py file. We use the custom_func_track_markers to define 
the objective function. In this example, the CUSTOM objective mimics `ObjectiveFcn.SUPERIMPOSE_MARKERS`.

### The [custom_parameters.py](./bioptim/examples/getting_started/custom_parameters.py) file 
This example is a clone of the pendulum.py example with the difference that the
model now evolves in an environment where gravity can be modified.
The goal of the solver is to find the optimal gravity (target = 8 N/kg) while performing the
pendulum balancing task.

It is designed to show how to define parameters.

The `my_parameter_function function` is used to modify the dynamics. In our case, we want to optimize the 
gravity. This function is called right before defining the dynamics of the system. The `my_target_function` function is 
a penalty function. Both functions define a new parameter, and then a parameter objective function 
is linked to this new parameter.

### The [custom_phase_transitions.py](./bioptim/examples/getting_started/custom_phase_transitions.py) file 
This example is a trivial multiphase box that must superimpose different markers at the beginning and end of each
phase with one of its corners.
It is designed to show how to define CUSTOM phase transition constraints if the provided ones are insufficient.

This example mimics the behavior of the most common `PhaseTransitionFcn.CONTINUOUS`

The custom_phase_transition function defines the constraint of the transition to apply. This function can be 
used when adding some phase transitions to the list of phase transitions. 

Different phase transitions can be considered. By default, all the phase transitions are continuous. However, if one or more phase transitions are desired to be continuous, it is possible to define and use a function like 
the `custom_phase_transition` function or directly use `PhaseTransitionFcn.IMPACT`. If a phase transition is desired 
between the last and the first phase, use the dedicated `PhaseTransitionFcn.Cyclic`. 

### The [custom_plot_callback.py](./bioptim/examples/getting_started/custom_plot_callback.py) file
This example is a trivial example of using the pendulum without any objective. It is designed to show how to create new
plots and expand pre-existing ones with new information.

We define the `custom_plot_callback` function, which returns the value(s) to plot. We use this function as an argument of 
`ocp.add_plot`. Let us describe the creation of the plot "My New Extra Plot". `custom_plot_callback` 
takes two arguments, x and the array [0, 1, 3], as you can see below :

```python
ocp.add_plot("My New Extra Plot", lambda x, u, p: custom_plot_callback(x, [0, 1, 3]), plot_type=PlotType.PLOT)
```

We use the plot_type `PlotType.PLOT`. It is a way to plot the first, 
second, and fourth states (i.e., `q_Seg1_TransY`, `q_Seg1_RotX` and `qdot_Seg1_RotX`) in a new window entitled "My New 
Extra Plot". Please note that for further information about the different plot types, you can refer to the section 
"Enum: PlotType".

### The [example_cyclic_movement.py](./bioptim/examples/getting_started/example_cyclic_movement.py) file 
This example is a trivial box that must superimpose one of its corners on a marker at the beginning of the movement
and superimpose the same corner on a different marker at the end. Moreover, the movement must be cyclic, meaning
that the states at the end and the beginning are equal. It is designed to provide a comprehensible example of the way
to declare a cyclic constraint or objective function.

A phase transition loop constraint is treated as a hard penalty (constraint)
if weight is <= 0 [or if no weight is provided], or as a soft penalty (objective) otherwise, as shown in the example below :

```python
phase_transitions = PhaseTransitionList()
if loop_from_constraint:
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, weight=0)
else:
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, weight=10000)
```

`loop_from_constraint` is a boolean. It is one of the parameters of the `prepare_ocp` function of the example. This parameter is a way to determine if the looping cost should be a constraint [True] or an objective [False]. 

### The [example_external_forces.py](./bioptim/examples/getting_started/example_external_forces.py) file
This example is a trivial box that must superimpose one of its corners on a marker at the beginning of the movement
and superimpose the same corner on a different marker at the end. While doing so, a force pushes the box upward.
The solver must minimize the force to lift the box while reaching the marker in time.
It is designed to show how to use external forces. An example of external forces that depends on the state (for
example, a spring) can be found at 'examples/torque_driven_ocp/spring_load.py'

Please note that the point of application of the external forces is defined in the `bioMod` file by the
`externalforceindex` tag in segment and is acting at the center of mass of this particular segment. Please note that
this segment must have at least one degree of freedom defined (translations and/or rotations). Otherwise, the
external_force is silently ignored. 

`Bioptim` expects `external_forces` to be a list (one element for each phase) of
a list (for each shooting node) of np.ndarray [6 x n], where the six components are [Mx, My, Mz, Fx, Fy, Fz], for the ith force platform
(defined by the `externalforceindex`) for each node n. Let us take a look at the definition of the external forces in 
this example :

```python
external_forces = external_forces = [[np.array([[0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 5]]).T for _ in range(n_shooting)]]
```

`external_forces` is of len 1 because there is only one phase. The list is 30-element long, and each array are 6x2 since there
is [Mx, My, Mz, Fx, Fy, Fz] for the two `externalforceindex` for each node (in this example, we take 30 shooting nodes).

### The [example_inequality_constraint.py](./bioptim/examples/getting_started/example_inequality_constraint.py) file
This example mimics what a jumper does when maximizing the predicted height of the center of mass at the peak of an aerial phase. It does so with a simplistic two segments model.
It is a clone of 'torque_driven_ocp/maximize_predicted_height_CoM.py' using
the option `MINIMIZE_PREDICTED_COM_HEIGHT`. It is different in that the contact forces on the ground have
to be downward (meaning that the object is limited to push on the ground, as one would expect when jumping). 

Moreover, the lateral forces must respect some `NON_SLIPPING` constraint (i.e., the ground reaction
forces have to remain inside of a cone of friction), as shown in the part of the code defining the constraints:

```python
constraints = ConstraintList()
   constraints.add(
   ConstraintFcn.TRACK_CONTACT_FORCES,
   min_bound=min_bound,
   max_bound=max_bound,
   node=Node.ALL,
   contact_index=1,
   )
constraints.add(
    ConstraintFcn.TRACK_CONTACT_FORCES,
    min_bound=min_bound,
    max_bound=max_bound,
    node=Node.ALL,
    contact_index=2,
    )
constraints.add(
    ConstraintFcn.NON_SLIPPING,
    node=Node.ALL,
    normal_component_idx=(1, 2),
    tangential_component_idx=0,
    static_friction_coefficient=mu,
    )
```

Let us describe the code above. First, we create a list of constraints. Then, two contact forces are defined with the indexes 1 and 2, respectively. The last step is the implementation of the 
non-slipping constraint for the two forces defined before.   

This example is designed to show how to use min_bound and max_bound values so they define inequality constraints instead
of equality constraints, which can be used with any `ConstraintFcn`.

### The example_mapping.py file 
An example of mapping can be found at 'examples/symmetrical_torque_driven_ocp/symmetry_by_mapping.py'.
Another example of mapping can be found at 'examples/getting_started/example_inequality_constraint.py'. 

### The example_multiphase.py file
This example is a trivial box that must superimpose one of its corners on a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
It is designed to show how one can define a multiphase optimal control program.

In this example, three phases are implemented. The `long_optim` boolean allows users to choose between solving the precise
optimization or the approximate. In the first case, 500 points are considered and `n_shooting = (100, 300, 100)`. 
Otherwise, 50 points are considered and `n_shooting = (20, 30, 20)`. Three steps are necessary to define the 
objective functions, the dynamics, the constraints, the path constraints, the initial guesses and the control path 
contsraints. Each step corresponds to one phase. 

Let's take a look at the definition of the constraints:

```python
constraints = ConstraintList()
constraints.add(
    ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1, phase=0
)
constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2, phase=0)
constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=1, phase=1)
constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2, phase=2)
```

First, we define a list of constraints, and then we add constraints to the list. At the beginning, marker 0 must 
superimpose marker 1. At the end of the first phase (the first 100 shooting nodes if we solve the precise optimization), 
marker 0 must superimpose marker 2. Then, at the end of the second phase, marker 0 must superimpose marker 1. At the 
end of the last step, marker 0 must superimpose marker 2. Please, note that the definition of the markers is 
implemented in the `bioMod` file corresponding to the model. Further information about the definition of the markers is
available in the `biorbd` documentation.

### The example_optimal_time.py file
Examples of time optimization can be found in 'examples/optimal_time_ocp/'.

### The example_save_and_load.py file
This is a clone of the getting_started/pendulum.py example. It is designed to show how to create and solve a problem,
and afterward, save it to the hard drive and reload it. It shows an example of *.bo method. 

Let's take a look at the most important lines of the example. To save the optimal control program and the solution, use
ocp.save(sol, "pendulum.bo"). To load the optimal control program and the solution, use 
`ocp_load, sol_load = OptimalControlProgram.load("pendulum.bo")`. Then, to show the results, 
simply use `sol_load.animate()`.

### The example_simulation.py file
The first part of this example of a single shooting simulation from initial guesses.
It is not an optimal control program. It is merely the simulation of values, that is applying the dynamics.
The main goal of this kind of simulation is to get a sens of the initial guesses passed to the solver.

The second part of the example is to actually solve the program and then simulate the results from this solution.
The main goal of this kind of simulation, especially in single shooting (that is not resetting the states at each node)
is to validate the dynamics of multiple shooting. If they both are equal, it usually means that a great confidence
can be held in the solution. Another goal would be to reload fast a previously saved optimized solution.

### The example_joints_acceleration_driven.py file
This example shows how to use the joints acceleration dynamic to achieve the same goal as the simple pendulum, but with a double pendulum for which only the angular acceleration of the second pendulum is controled.

### The pendulum.py file
This is another way to present the pendulum example of the 'Getting started' section.

## Muscle driven OCP
In this file, you will find four examples about muscle driven optimal control programs. The two first refer to traking 
examples. The two last refer to reaching tasks. 

### The muscle_activations_tracker.py file
This is an example of muscle activation/skin marker or state tracking.
Random data are created by generating a random set of muscle activations and then by generating the kinematics
associated with these data. The solution is trivial since no noise is applied to the data. Still, it is a relevant
example to show how to track data using a musculoskeletal model. In real situation, the muscle activation
and kinematics would indeed be acquired via data acquisition devices.

The difference between muscle activation and excitation is that the latter is the derivative of the former.

The generate_data function is used to create random data. First, a random set of muscle activation is generated, as 
shown below:
`U = np.random.rand(n_shooting, n_mus).T`

Then, the kinematics associated with these data are generated by numerical integration, using 
`scipy.integrate.solve_ivp`. 

To implement this tracking task, we use the ObjectiveFcn.Lagrange.TRACK_STATE objective function in the case of a state 
tracking, or the `ObjectiveFcn.Lagrange.TRACK_MARKERS` objective function in the case of a marker tracking. We also use 
the `ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL` objective function. The user can choose between marker or state 
tracking thanks to the string `kin_data_to_track` which is one of the `prepare_ocp` function parameters. 

### The muscle_excitations_tracker.py file
This is an example of muscle excitation(EMG)/skin marker or state tracking.
Random data are created by generating a random set of EMG and then by generating the kinematics associated with these
data. The solution is trivial since no noise is applied to the data. Still, it is a relevant example to show how to
track data using a musculoskeletal model. In real world, the EMG and kinematics would indeed be acquired via
data acquisition devices.

There is no huge difference with the precedent example. Some dynamic equations make the link between muscle activation
and excitation. 

### The static_arm.py file
This is a basic example on how to use `biorbd` model driven by muscle to perform an optimal reaching task.
The arms must reach a marker placed upward in front while minimizing the muscles activity.

For this reaching task, we use the `ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS` objective function. At the end of the 
movement, marker 0 and marker 5 should superimpose. The weight applied to the `SUPERIMPOSE_MARKERS` objective function 
is 1000. Please note that the bigger this number, the greater the model will try to reach the marker. 

Please note that using show_meshes=True in the animator may be long due to the creation of a huge `CasADi` graph of the
mesh points.

### The static_arm_with_contact.py file
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task with a
contact dynamics.
The arms must reach a marker placed upward in front while minimizing the muscles activity.

The only difference with the precedent example is that we use the arm26_with_contact.bioMod model and the 
`DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT` dynamics function instead of 
`DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN`.

Please note that using show_meshes=True in the animator may be long due to the creation of a huge `CasADi` graph of the
mesh points.

## Muscle driven with contact
All the examples in muscle_driven_with_contact are merely to show some dynamics and prepare some OCP for the tests.
It is not really relevant and will be removed when unitary tests for the dynamics will be implemented.

### The contact_forces_inequality_constraint_muscle.py file
In this example, we implement inequality constraints on two contact forces. It is designed to show how to use min_bound 
and max_bound values so they define inequality constraints instead of equality constraints, which can be used with 
any ConstraintFcn.

In this case, the dynamics function used is `DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT`.

### The contact_forces_inequality_constraint_muscle_excitations.py file
In this example, we implement inequality constraints on two contact forces. It is designed to show how to use `min_bound` 
and `max_bound` values so they define inequality constraints instead of equality constraints, which can be used with any 
`ConstraintFcn`.

In this case, the dynamics function used is `DynamicsFcn.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT` instead of 
`DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT` used in the precedent example. 

### The muscle_activations_contacts_tracker.py file 
In this example, we track both muscle controls and contact forces, as it is defined when adding the two objective 
functions below, using both `ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL` and 
`ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES` objective functions. 

```python
objective_functions = ObjectiveList()
objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL, target=muscle_activations_ref)
objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES, target=contact_forces_ref)
```

Let's take a look at the structure of this example. First, we load data to track, and we generate data using the 
`data_to_track.prepare_ocp` optimization control program. Then, we track these data using `muscle_activation_ref` and 
`contact_forces_ref` as shown below:

```python
ocp = prepare_ocp(
    biorbd_model_path=model_path,
    phase_time=final_time,
    n_shooting=ns,
    muscle_activations_ref=muscle_activations_ref[:, :-1],
    contact_forces_ref=contact_forces_ref,
)
```

## Optimal time OCP
In this section, you will find four examples showing how to play with time parameters.  

### The multiphase_time_constraint.py file
This example is a trivial multiphase box that must superimpose different markers at beginning and end of each
phase with one of its corner. The time is free for each phase.
It is designed to show how one can define a multi-phase ocp problem with free time. 

In this example, the number of phases is 1 or 3. prepare_ocp function takes `time_min`, `time_max` and `final_time` as 
arguments. There are arrays of length 3 in the case of a 3-phase problem. In the example, these arguments are defined 
as shown below:

```python
final_time = [2, 5, 4]
time_min = [1, 3, 0.1]
time_max = [2, 4, 0.8]
ns = [20, 30, 20]
ocp = prepare_ocp(final_time=final_time, time_min=time_min, time_max=time_max, n_shooting=ns)
```

We can make out different time constraints for each phase, as shown in the code below:

```python
constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[0], max_bound=time_max[0], phase=0)
if n_phases == 3:
    constraints.add(
        ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[1], max_bound=time_max[1], phase=1
    )
    constraints.add(
        ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[2], max_bound=time_max[2], phase=2
    )
```

### The pendulum_min_time_Lagrange.py file
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
the time to perform the task is now free and minimized by the solver, as shown in the definition of the objective 
function used for this example:

```python
objective_functions = ObjectiveList()
objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=1)
```

Please note that a weight of -1 will maximize time. 

This example shows how to define such an optimal
control program with a Lagrange criteria (integral of dt).

The difference between Mayer and Lagrange minimization time is that the former can define bounds to
the values, while the latter is the most common way to define optimal time. 

### The pendulum_min_time_Mayer.py file
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
the time to perform the task is now free and minimized by the solver, as shown in the definition of the objective 
function used for this example: 

```python
objective_functions = ObjectiveList()
objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight, min_bound=min_time, max_bound=max_time)
```

Please note that a weight of -1 will maximize time. 

This example shows how to define such an optimal
control program with a Mayer criteria (value of `final_time`).

The difference between Mayer and Lagrange minimization time is that the former can define bounds to
the values, while the latter is the most common way to define optimal time.

### The time_constraint.py file
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
the time to perform the task is now free for the solver to change. This example shows how to define such an optimal
control program. 

In this example, a time constraint is implemented:

```python
constraints = Constraint(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min, max_bound=time_max)
```

## Symmetrical torque driven OCP
In this section, you will find an example using symmetry by constraint and another using symmetry by mapping. In both 
cases, we simulate two rodes. We must superimpose a marker on one rod at the beginning and another marker on the
same rod at the end, while keeping the degrees of freedom opposed. 

The difference between the first example (symmetry_by_mapping) and the second one (symmetry_by_constraint) is that one 
(mapping) removes the degree of freedom from the solver, while the other (constraints) imposes a proportional 
constraint (equals to -1) so they are opposed.
Please note that even though removing a degree of freedom seems a good idea, it is unclear if it is actually faster when
solving with `IPOPT`.

### The symmetry_by_constraint.py file
This example imposes a proportional constraint (equals to -1) so that the rotation around the x axis remains opposed 
for the two rodes during the movement. 

Let's take a look at the definition of such a constraint:

```python
constraints.add(ConstraintFcn.PROPORTIONAL_STATE, node=Node.ALL, first_dof=2, second_dof=3, coef=-1)
```

In this case, a proportional constraint is generated between the third degree of freedom defined in the `bioMod` file 
(`first_dof=2`) and the fourth one (`second_dof=3`). Looking at the cubeSym.bioMod file used in this example, we can make 
out that the dof with index 2 corresponds to the rotation around the x axis for the first segment `Seg1`. The dof 
with index 3 corresponds to the rotation around the x axis for the second segment `Seg2`. 

### The symmetry_by_mapping.py file
This example imposes the symmetry as a mapping, that is by completely removing the degree of freedom from the solver 
variables but interpreting the numbers properly when computing the dynamics.

A `BiMapping` is used. The way to understand the mapping is that if one is provided with two vectors, what
would be the correspondence between those vector. For instance, `BiMapping([None, 0, 1, 2, -2], [0, 1, 2])`
would mean that the first vector (v1) has 3 components and to create it from the second vector (v2), you would do:
v1 = [v2[0], v2[1], v2[2]]. Conversely, the second v2 has 5 components and is created from the vector v1 using:
v2 = [0, v1[0], v1[1], v1[2], -v1[2]]. For the dynamics, it is assumed that v1 is what is to be sent to the dynamic
functions (the full vector with all the degrees of freedom), while v2 is the one sent to the solver (the one with less
degrees of freedom).

The `BiMapping` used is defined as a problem parameter, as shown below:

```python
all_generalized_mapping = BiMapping([0, 1, 2, -2], [0, 1, 2])
```

## Torque driven OCP
In this section, you will find different examples showing how to implement torque driven optimal control programs.

### The maximize_predicted_height_CoM.py file
This example mimics by essence what a jumper does which is maximizing the predicted height of the
center of mass at the peak of an aerial phase. It does so with a very simple two segments model though.
It is designed to give a sense of the goal of the different MINIMIZE_COM functions and the use of
`weight=-1` to maximize instead of minimizing.

Let's take a look at the definition of the objetive functions used for this example to better understand how to 
implement that:

```python
objective_functions = ObjectiveList()
if objective_name == "MINIMIZE_PREDICTED_COM_HEIGHT":
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)
elif objective_name == "MINIMIZE_COM_POSITION":
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, axis=Axis.Z, weight=-1)
elif objective_name == "MINIMIZE_COM_VELOCITY":
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_VELOCITY, axis=Axis.Z, weight=-1)
```

Another interesting point of this example is the definition of the constraints. Thanks to the `com_constraints` boolean, 
the user can easily choose to apply constraints on the center of mass. Here is the definition of the constraints for our 
example:

```python
constraints = ConstraintList()
if com_constraints:
    constraints.add(
        ConstraintFcn.TRACK_COM_VELOCITY,
        node=Node.ALL,
        min_bound=np.array([-100, -100, -100]),
        max_bound=np.array([100, 100, 100]),
    )
    constraints.add(
        ConstraintFcn.TRACK_COM_POSITION,
        node=Node.ALL,
        min_bound=np.array([-1, -1, -1]),
        max_bound=np.array([1, 1, 1]),
    )
```

This example is designed to show how to use `min_bound` and `max_bound` values so they define inequality constraints 
instead of equality constraints, which can be used with any `ConstraintFcn`. This example is closed to the 
example_inequality_constraint.py file you can find in 'examples/getting_started/example_inequality_constraint.py'.

### The spring_load.py file 
This trivial spring example targets to have the highest upward velocity. It is however only able to load a spring by
pulling downward and afterward to let it go so it gains velocity. It is designed to show how one can use the external
forces to interact with the body.

This example is closed to the custom_dynamics.py file you can find in 'examples/getting_started/custom_dynamics.py'. 
Indeed, we generate an external force thanks to the custom_dynamic function. Then, we configure the dynamics with 
the `custom_configure` function. 

### The track_markers_2D_pendulum.py file

This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked. 

In this example, we use both `ObjectiveFcn.Lagrange.TRACK_MARKERS` and `ObjectiveFcn.Lagrange.TRACK_TORQUE` objective 
functions to track data, as shown in the definition of the objective functions used in this example:

```python
objective_functions = ObjectiveList()
objective_functions.add(
    ObjectiveFcn.Lagrange.TRACK_MARKERS, axis_to_track=[Axis.Y, Axis.Z], weight=100, target=markers_ref
)
objective_functions.add(ObjectiveFcn.Lagrange.TRACK_TORQUE, target=tau_ref)
```

This is a good example of how to load data for tracking tasks, and how to plot data. The extra parameter 
`axis_to_track` allows users to specify the axes on which to track the markers (x and y axes in this example).
This example is closed to the example_save_and_load.py and custom_plotting.py files you can find in the 
examples/getting_started repository. 

### The track_markers_with_torque_actuators.py file

This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end. It is a clone of
'getting_started/custom_constraint.py' 

It is designed to show how to use the `TORQUE_ACTIVATIONS_DRIVEN` which limits
the torque to [-1; 1]. This is useful when the maximal torque are not constant. Please note that this dynamic then
to not converge when it is used on more complicated model. A solution that defines non-constant constraints seems a
better idea. An example of which can be found with the `bioptim` paper.

Let's take a look at the structure of the code. First, tau_min, tau_max and tau_init are respectively initialized 
to -1, 1 and 0 if the integer `actuator_type` (which is a parameter of the `prepare_ocp` function) equals to 1. 
In this particular case, the dynamics function used is `DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN`. 

### The trampo_quaternions.py file

This example uses a representation of a human body by a trunk_leg segment and two arms.
It is designed to show how to use a model that has quaternions in their degrees of freedom.

## Track
In this section, you will find the description of two tracking examples. 

### The track_marker_on_segment.py file
This example is a trivial example where a stick must keep a corner of a box in line for the whole duration of the
movement. The initial and final position of the box are dictated, the rest is fully optimized. It is designed
to show how one can use the tracking function to track a marker with a body segment.

In this case, we use the `ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS` constraint function, as shown below in the 
definition of the constraints of the problem:

```python
constraints = ConstraintList()
constraints.add(
ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker_idx=1, segment_idx=2, axis=Axis.X
)
```

Here, we minimize the distance between the marker with index 1 ans the x axis of the segment with index 2. We align 
the axis toward the marker. 

### The track_segment_on_rt.py file
This example is a trivial example where a stick must keep its coordinate system of axes aligned with the one
from a box during the whole duration of the movement. The initial and final position of the box are dictated,
the rest is fully optimized. It is designed to show how one can use the tracking RT function to track
any RT (for instance Inertial Measurement Unit [IMU]) with a body segment.

To implement this tracking task, we use the `ConstraintFcn.TRACK_SEGMENT_WITH_CUSTOM_RT` constraint function, which 
minimizes the distance between a segment and an RT. The extra parameters `segment_idx: int` and `rt_idx: int` must be 
passed to the Objective constructor.

## Moving estimation horizon
In this section, we perform mhe on the pendulum example.

### The mhe.py file
In this example, mhe (Moving Horizon Estimation) is applied on a simple pendulum simulation. Data are generated (states,
controls, and marker trajectories) to simulate the movement of a pendulum, using `scipy.integrate.solve_ivp`. These data
are used to perform mhe.

In this example, 500 shooting nodes are defined. As the size of the mhe window is 10, 490 iterations are performed to
solve the complete problem.

For each iteration, the new marker trajectory is taken into account so that a real time data acquisition is simulated.
For each iteration, the list of objectives is updated, the problem is solved with the new frame added to the window,
the oldest frame is discarded with the `warm_start_mhe function`, and it is saved. The results are plotted so that
estimated data can be compared to real data. 

## Acados
In this section, you will find three examples to investigate `bioptim` using `acados`. 

### The cube.py file
This is a basic example of a cube which have to reach a target at the end of the movement, starting from an initial 
position, and minimizing states and torques. This problem is solved using `acados`. 

### The pendulum.py file 
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating `bioptim` using `acados` as it describes the most common
dynamics out there (the joint torque driven), it defines an objective function and some boundaries and initial guesses.

### The static_arm.py file
This basic example shows how to use biorbd model driven by muscle to perform an optimal reaching task.
The arm must reach a marker while minimizing the muscles' activity and the states. We solve the problem using both 
`acados` and `ipopt`.

## Inverse optimal control
In this section, you will find an example about inverse optimal control with `bioptim`. 

### The double_pendulum_torque_driven_IOCP.py file
This basic example is a rigid double pendulum that circles around a fixed point.
The movement is inspired by the motion of gymnasts on the bar apparatus.
This example is separated into three parts:
- The first part is the definition of the problem. The problem is solved with specific weightings.
- The second part solves the problem with only one objective at a time for the Pareto front.
- The third part solves the inverse optimal control problem aiming to retrieve the initial weightings.
A the end of the example, the markers' trajectories are plotted to show that the movement is the same.

# Performance
If you find yourself asking, "Why is bioptim so slow? I thought it was lightning fast!"
Then this section may help you improve your code to get better performance.

## use_sx
Set use_sx to True in the OptimalControlProgram class to use the SX symbolic variables.
These are faster but require more RAM, so ensure you have enough RAM to use this option.

## n_threads
Set n_threads to the number of threads you want to use in the OptimalControlProgram class.
By default, it is set to 1. It will split the computation of the continuity constraints between threads and speed up the computation. If applicable to your problem, use the next option too.

## assume_phase_dynamics
Set assume_phase_dynamics to True in the OptimalControlProgram class if your dynamic equations are phase-independent but not node-specific.
It will speed up the computation of the continuity constraints since only one continuity constraint will be computed per phase instead of one per node.

## expand
(For objective and constraint functions)
Set the expand argument to True for objective and constraint functions to speed up the computation.
It will turn MX symbolic variables into SX symbolic variables, which is faster but requires more RAM.

# Troubleshooting
Despite our best efforts to assist you with this long Readme and several examples, you may experience some problems with bioptim.
Fortunately, this troubleshooting section will guide you through solving some known issues.

## Freezing compute
If your computer freezes before any optimization is performed, it is probably because your problem requires too much RAM.
If you are using use_sx and/or expand options, try turning them off. If it does not work, try reducing the number of nodes. If assume_phase_dynamics is set to False, try setting it to True, if applicable to your problem.

## Free variables
Sometimes when working on advanced custom problems, you may have *free variables* that prevent the solver from being launched.
If this occurs, try reloading your model inside of the custom function. We have found this solution to be effective when working with biorbd models.

## Non-converging problems
If Ipopt converges to an infeasible solution, ensure the boundaries are sound for the problem's constraints.
If the problem still does not converge, try changing the initial guess of the problem.

If the problem takes numerous iterations to solve (much more than expected), check the weights on objective functions and the weight of the actual variables.

If the problem still does not converge, try observing the evolution of the objective function and the constraints through a live plot.
It is always good to see how they evolve through the iterations.

# Citing
If you use `bioptim`, we would be grateful if you could cite it as follows:
@article{michaud2022bioptim,
  title={Bioptim, a python framework for musculoskeletal optimal control in biomechanics},
  author={Michaud, Benjamin and Bailly, Fran{\c{c}}ois and Charbonneau, Eve and Ceglia, Amedeo and Sanchez, L{\'e}a and Begon, Mickael},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  year={2022},
  publisher={IEEE}
}
