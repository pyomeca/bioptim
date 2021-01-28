"""
The first part of this example of a single shooting simulation from initial guesses.
It is NOT an optimal control program. It is merely the simulation of values, that is applying the dynamics.
The main goal of this kind of simulation is to get a sens of the initial guesses passed to the solver

The second part of the example is to actually solve the program and then simulate the results from this solution.
The main goal of this kind of simulation, especially in single shooting (that is not resetting the states at each node)
is to validate the dynamics of multiple shooting. If they both are equal, it usually means that a great confidence
can be held in the solution. Another goal would be to reload fast a previously saved optimized solution
"""

import importlib.util
from pathlib import Path

from bioptim import InitialGuess, Simulate, ShowResult, Data


# --- Load pendulum --- #
PROJECT_FOLDER = Path(__file__).parent / "../.."
spec = importlib.util.spec_from_file_location("pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py")
pendulum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum)


ocp = pendulum.prepare_ocp(
    biorbd_model_path="pendulum.bioMod",
    final_time=2,
    n_shooting=10,
)

X = InitialGuess([0, 0, 0, 0])
U = InitialGuess([-1, 1])

# --- Single shooting --- #
sol_simulate_single_shooting = Simulate.from_controls_and_initial_states(ocp, X, U, single_shoot=True)
result_single = ShowResult(ocp, sol_simulate_single_shooting)
result_single.graphs()

# --- Multiple shooting --- #
sol_simulate_multiple_shooting = Simulate.from_controls_and_initial_states(ocp, X, U, single_shoot=False)
result_multiple = ShowResult(ocp, sol_simulate_multiple_shooting)
result_multiple.graphs()

# --- Simulation --- #
# It is not an optimal control, it only apply a Runge Kutta at each nodes
ocp = pendulum.prepare_ocp(
    biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
    final_time=2,
    n_shooting=10,
)
sol = ocp.solve()
sol_from_sol = Simulate.from_solve(ocp, sol, single_shoot=True)
ShowResult(ocp, sol_from_sol).graphs()

sol_from_data = Simulate.from_data(ocp, Data.get_data(ocp, sol), single_shoot=False)
ShowResult(ocp, sol_from_data).graphs()
