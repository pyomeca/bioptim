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
    Bounds,
    InitialGuess,
    OdeSolver,
    OdeSolverBase,
    PenaltyController,
    Solver,
)


def custom_func_track_markers(controller: PenaltyController, first_marker: str, second_marker: str, method: int) -> MX:
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
    method: int
        Two identical ways are shown to help the new user to navigate the biorbd API

    Returns
    -------
    The cost that should be minimize in the MX format. If the cost is quadratic, do not put
    the square here, but use the quadratic=True parameter instead
    """

    # Get the index of the markers from their name
    marker_0_idx = controller.model.marker_index(first_marker)
    marker_1_idx = controller.model.marker_index(second_marker)

    if method == 0:
        # Convert the function to the required format and then subtract
        markers = controller.mx_to_cx("markers", controller.model.markers, controller.states["q"])
        markers_diff = markers[:, marker_1_idx] - markers[:, marker_0_idx]

    else:
        # Do the calculation in biorbd API and then convert to the required format
        markers = controller.model.markers(controller.states["q"].mx)
        markers_diff = markers[marker_1_idx] - markers[marker_0_idx]
        markers_diff = controller.mx_to_cx("markers", markers_diff, controller.states["q"])

    return markers_diff


def prepare_ocp(
    biorbd_model_path, ode_solver: OdeSolverBase = OdeSolver.RK4(), assume_phase_dynamics: bool = True
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolver
        The type of ode solver used
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

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
        method=0,
    )
    objective_functions.add(
        custom_func_track_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        quadratic=True,
        first_marker="m0",
        second_marker="m2",
        weight=1000,
        method=1,
    )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    # Initial guess
    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))

    # Define control path constraint
    u_bounds = Bounds([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)

    u_init = InitialGuess([tau_init] * bio_model.nb_tau)

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
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
