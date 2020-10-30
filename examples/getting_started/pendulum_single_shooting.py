import importlib.util
from pathlib import Path

from bioptim import InitialGuess, Simulate, ShowResult


# It is not an optimal control, it only apply a Runge Kutta at each nodes

# --- Load pendulum --- #
PROJECT_FOLDER = Path(__file__).parent / "../.."
spec = importlib.util.spec_from_file_location("pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py")
pendulum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum)


ocp = pendulum.prepare_ocp(
    biorbd_model_path="pendulum.bioMod",
    final_time=2,
    number_shooting_points=10,
    nb_threads=2,
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

# --- Single shooting --- #
result_single.animate()

# --- Multiple shooting --- #
result_multiple.animate()
