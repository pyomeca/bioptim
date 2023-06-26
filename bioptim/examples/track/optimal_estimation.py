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

# Load track_segment_on_rt
EXAMPLES_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location("data_to_track", str(EXAMPLES_FOLDER) + "/getting_started/pendulum.py")
data_to_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_to_track)


def get_markers_pos(x: DM | np.ndarray, idx_marker: int, fun: Callable, n_q: int) -> DM | np.ndarray:
    """
    Get the position of a specific marker from the states

    Parameters
    ----------
    x: DM | np.ndarray
        The states to get the marker positions from
    idx_marker: int
        The index of the marker to get the position
    fun: Callable
        The casadi function of the marker position

    Returns
    -------
    The 3xT ([X, Y, Z] x [Time]) matrix of data
    """

    marker_pos = []
    for i in range(x.shape[1]):
        marker_pos.append(fun(x[:n_q, i])[:, idx_marker])
    marker_pos = horzcat(*marker_pos)
    return marker_pos


def prepare_ocp(
    bio_model: BiorbdModel,
    final_time: float,
    n_shooting: int,
    markers_ref: np.ndarray,
    tau_ref: np.ndarray,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
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

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_MARKERS,
        axes=[Axis.Y, Axis.Z],
        node=Node.ALL,
        weight=100,
        target=markers_ref[1:, :, :],
    )
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="tau", target=tau_ref)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

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
        objective_functions=objective_functions,
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

    ocp_to_track = data_to_track.prepare_ocp(
        biorbd_model_path=biorbd_path, final_time=final_time, n_shooting=n_shooting
    )
    sol = ocp_to_track.solve()
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    n_q = bio_model.nb_q
    n_marker = bio_model.nb_markers
    x = np.concatenate((q, qdot))

    symbolic_states = MX.sym("q", n_q, 1)
    markers_fun = biorbd.to_casadi_func("ForwardKin", bio_model.markers, symbolic_states)
    markers_ref = np.zeros((3, n_marker, n_shooting + 1))
    for i in range(n_shooting + 1):
        markers_ref[:, :, i] = markers_fun(x[:n_q, i])
    tau_ref = tau[:, :-1]

    ocp = prepare_ocp(
        bio_model,
        final_time=final_time,
        n_shooting=n_shooting,
        markers_ref=markers_ref,
        tau_ref=tau_ref,
    )

    # --- plot markers position --- #
    title_markers = ["x", "y", "z"]
    marker_color = ["tab:red", "tab:orange"]

    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda t, x, u, p: get_markers_pos(x, 0, markers_fun, n_q),
        linestyle=".-",
        plot_type=PlotType.STEP,
        color=marker_color[0],
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda t, x, u, p: get_markers_pos(x, 1, markers_fun, n_q),
        linestyle=".-",
        plot_type=PlotType.STEP,
        color=marker_color[1],
    )

    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda t, x, u, p: markers_ref[:, 0, :],
        plot_type=PlotType.PLOT,
        color=marker_color[0],
        legend=title_markers,
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda t, x, u, p: markers_ref[:, 1, :],
        plot_type=PlotType.PLOT,
        color=marker_color[1],
        legend=title_markers,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
