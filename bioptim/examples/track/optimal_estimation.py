"""
This is an example of how to state an optimal estimation problem.
The only objective of the OCP is to track the experimental data.
It allows to get a reconstruction that is dynamically consistent in opposition to the Kalman filter for example
(however, it is slower to compute).
See https://www.tandfonline.com/doi/full/10.1080/14763141.2022.2066015 for comparison.
"""

from typing import Callable

import biorbd_casadi as biorbd
import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, horzcat, DM
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    BoundsList,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    BiMappingList,
    SolutionMerge,
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
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
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
    ode_solver: OdeSolverBase
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Rotations of the pendulum are passive
    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[None, None, None, 0, 1, 2], to_first=[3, 4, 5], phase=0)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1e-6, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        derivative=True,
        node=Node.ALL_SHOOTING,
        weight=1e-6,
        quadratic=True,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.END, index=[3, 5], weight=-1000, quadratic=False
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=-1, quadratic=True)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(ode_solver=ode_solver)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"].min[:3, 0] = 0

    # Define control path constraint
    tau_min, tau_max = -5, 5
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 3, max_bound=[tau_max] * 3)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        variable_mappings=variable_mappings,
    )


def prepare_optimal_estimation(
    biorbd_model_path: str,
    time_ref: float,
    n_shooting: int,
    markers_ref: np.ndarray,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    bio_model: BiorbdModel
        The loaded biorbd model
    time_ref: float
        The duration of the original movement
    n_shooting: int
        The number of shooting points
    markers_ref: np.ndarray
        The markers to track
    ode_solver: OdeSolverBase
        The ode solver to use
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.ALL, weight=100, target=markers_ref, quadratic=True
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, target=time_ref, quadratic=True)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(ode_solver=ode_solver)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max = -5, 5
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        time_ref,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


def main():
    """
    1) An OCP which tries to rotate a 3D pendulum as fast as possible around the Z-axis, and then
    stabilizes it in an inverted position is solved to generate data.
    2) Noise is added to the marker positions to simulate experimental data.
    3) The marker positions are tracked to find the torques that best match the experimental kinematics.
    """

    biorbd_model_path = "models/cube_6dofs.bioMod"
    final_time = 3
    n_shooting = 30

    ocp_to_track = prepare_ocp_to_track(
        biorbd_model_path=biorbd_model_path, final_time=final_time, n_shooting=n_shooting
    )
    sol = ocp_to_track.solve()
    # sol.animate()
    # sol.graphs()

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    time = float(sol.decision_time(to_merge=SolutionMerge.NODES)[-1])

    model = biorbd.Model(biorbd_model_path)
    n_q = model.nbQ()
    n_marker = model.nbMarkers()

    symbolic_states = MX.sym("q", n_q, 1)
    markers_ref = np.zeros((3, n_marker, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        markers_ref[:, :, i_node] = model.markers()(q[:, i_node])

    ocp = prepare_optimal_estimation(
        biorbd_model_path=biorbd_model_path,
        time_ref=time,
        n_shooting=n_shooting,
        markers_ref=markers_ref,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]

    markers_opt = np.zeros((3, n_marker, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        markers_opt[:, :, i_node] = markers_fun(q[:, i_node])

    # --- Plot --- #
    plt.figure("Markers")
    for i in range(markers_opt.shape[1]):
        plt.plot(
            np.linspace(0, final_time, n_shooting + 1),
            markers_ref[:, i, :].T,
            "k",
        )
        plt.plot(
            np.linspace(0, final_time, n_shooting + 1),
            markers_opt[:, i, :].T,
            "r--",
        )
    plt.show()

    # sol.animate(show_tracked_markers=True)


if __name__ == "__main__":
    main()
