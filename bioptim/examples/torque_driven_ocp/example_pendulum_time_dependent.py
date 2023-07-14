"""
This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked.
"""

from typing import Callable
import importlib.util
from pathlib import Path
import platform

import biorbd_casadi as biorbd
import numpy as np
from casadi import MX, horzcat, DM
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    ObjectiveList,
    ObjectiveFcn,
    Axis,
    PlotType,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
)

from bioptim.examples.getting_started.custom_dynamics import custom_dynamic, custom_configure

# Load track_segment_on_rt
EXAMPLES_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location("data_to_track", str(EXAMPLES_FOLDER) + "/getting_started/pendulum.py")
data_to_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_to_track)


def prepare_ocp(
    bio_model: BiorbdModel,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    bio_model: BiorbdModel
        The loaded biorbd model
    final_time: float
        The time at final node
    n_shooting: int
        The number of shooting points
    markers_ref: np.ndarray
        The markers to track
    tau_ref: np.ndarray
        The generalized forces to track
    ode_solver: OdeSolverBase
        The ode solver to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True

    # dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=expand)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Firstly, it solves the getting_started/pendulum.py example. Afterward, it gets the marker positions and joint
    torque from the solution and uses them to track. It then creates and solves this ocp and show the results
    """

    biorbd_path = str(EXAMPLES_FOLDER) + "/getting_started/models/pendulum.bioMod"
    bio_model = BiorbdModel(biorbd_path)
    final_time = 1
    n_shooting = 20

    ocp = prepare_ocp(
        bio_model,
        final_time=final_time,
        n_shooting=n_shooting,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    sol.graphs()


if __name__ == "__main__":
    main()
