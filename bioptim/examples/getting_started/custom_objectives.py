"""
This example is a trivial box that tries to superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end.
It is designed to show how one can define its own custom objective function if the provided ones are not
sufficient.

More specifically this example reproduces the behavior of the Mayer.SUPERIMPOSE_MARKERS objective function.
"""

import platform

from casadi import MX
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ObjectiveFcn,
    ObjectiveList,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    PenaltyController,
    Solver,
    PhaseDynamics,
)


def custom_func_track_markers(controller: PenaltyController, first_marker: str, second_marker: str) -> MX:
    """
    The used-defined objective function (This particular one mimics the ObjectiveFcn.SUPERIMPOSE_MARKERS)
    Except for the last two

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    first_marker: str
        The index of the first marker in the bioMod
    second_marker: str
        The index of the second marker in the bioMod

    Returns
    -------
    The cost that should be minimize in the MX format. If the cost is quadratic, do not put
    the square here, but use the quadratic=True parameter instead
    """

    # Get the index of the markers from their name
    marker_0_idx = controller.model.marker_index(first_marker)
    marker_1_idx = controller.model.marker_index(second_marker)

    # Convert the function to the required format and then subtract
    markers = controller.model.markers()(controller.states["q"].cx, controller.parameters.cx)
    markers_diff = markers[:, marker_1_idx] - markers[:, marker_0_idx]

    return markers_diff


def prepare_ocp(
    biorbd_model_path,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolverBase
        The type of ode solver used
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should alos be used when multi-node penalties are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objective_functions.add(
        custom_func_track_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.START,
        quadratic=True,
        first_marker="m0",
        second_marker="m1",
        weight=1000,
    )
    objective_functions.add(
        custom_func_track_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        quadratic=True,
        first_marker="m0",
        second_marker="m2",
        weight=1000,
    )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["q"][2, -1] = 1.57
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model.nb_tau, max_bound=[tau_max] * bio_model.nb_tau)

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
    )


def main():
    """
    Solve and animate the solution
    """

    model_path = "models/cube.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # Custom plots
    ocp.add_plot_penalty()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
