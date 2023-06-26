"""
This is an example of how to state an optimal estimation problem.
The only objective of the OCP is to track the experimental data.
It allows to get a reconstruction that is dynamically consistent in opposition to the Kalman filter for example
(however, it is slower to compute).
See https://www.tandfonline.com/doi/full/10.1080/14763141.2022.2066015 for comparison.
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
    BiMappingList,
)

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


def prepare_ocp_to_track(
    biorbd_model_path: str,
    final_time: tuple[float],
    n_shooting: tuple[int],
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

    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Rotations of the pendulum are passive
    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[0, 1, None, None], to_first=[0, 1], phase=0)
    variable_mappings.add("tau", to_second=[0, 1, None, None], to_first=[0, 1], phase=1)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_ANGULAR_MOMENTUM, node=Node.END, weight=-1000, quadratic=False, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1e-6, quadratic=True, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1e-6, quadratic=True, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, node=Node.ALL_SHOOTING, weight=1, quadratic=True, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, node=Node.ALL_SHOOTING, weight=1, quadratic=True, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS_VELOCITY, index=1, node=Node.END, weight=1000, quadratic=True, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, axes=Axis.Z, node=Node.END, weight=-1000, quadratic=False, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["qdot"][:, 0] = 0

    # Define control path constraint
    tau_min, tau_max = -50, 50
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2, phase=1)

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        variable_mappings=variable_mappings,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )

def prepare_optimal_estimation(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    markers_ref: np.ndarray,
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

    bio_model = BiorbdModel(biorbd_model_path, biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_MARKERS,
        node=Node.ALL,
        weight=100,
        target=markers_ref,
        phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_MARKERS,
        node=Node.ALL,
        weight=100,
        target=markers_ref,
        phase=1
    )
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="tau", weight=1e-6, quadratic=True)

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
    1) An OCP which tries to rotate a 3D pendulum as fast as possible around the Z-axis, and then
    stabilizes it in an inverted position is solved to generate data.
    2) Noise is added to the marker positions to simulate experimental data.
    3) The marker positions are tracked to find the torques that best match the experimental kinematics.
    """

    biorbd_model_path = "models/pendulum3D.bioMod"
    final_time = (5, 5)
    n_shooting = (30, 30)

    ocp_to_track = prepare_ocp_to_track(
        biorbd_model_path=biorbd_model_path, final_time=final_time, n_shooting=n_shooting
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
    sol.animate()


if __name__ == "__main__":
    main()
