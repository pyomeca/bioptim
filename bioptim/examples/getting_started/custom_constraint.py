"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end.
It is designed to show how one can define its own custom constraints function if the provided ones are not
sufficient.

More specifically this example reproduces the behavior of the SUPERIMPOSE_MARKERS constraint.
"""

import platform

from casadi import MX
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    PenaltyController,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
)


def custom_func_track_markers(controller: PenaltyController, first_marker: str, second_marker: str, method) -> MX:
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
        from bioptim import BiorbdModel

        # noinspection PyTypeChecker
        model: BiorbdModel = controller.model
        markers = controller.mx_to_cx("markers", model.model.markers, controller.states["q"])
        markers_diff = markers[:, marker_1_idx] - markers[:, marker_0_idx]

    else:
        # Do the calculation in biorbd API and then convert to the required format
        markers = controller.model.markers(controller.states["q"].mx)
        markers_diff = markers[marker_1_idx] - markers[marker_0_idx]
        markers_diff = controller.mx_to_cx("markers", markers_diff, controller.states["q"])

    return markers_diff


def prepare_ocp(
    biorbd_model_path: str,
    ode_solver: OdeSolverBase = OdeSolver.IRK(),
    assume_phase_dynamics: bool = True,
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
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
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

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand=expand_dynamics)

    # Constraints
    constraints = ConstraintList()
    constraints.add(custom_func_track_markers, node=Node.START, first_marker="m0", second_marker="m1", method=0)
    constraints.add(custom_func_track_markers, node=Node.END, first_marker="m0", second_marker="m2", method=1)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][1:, [0, -1]] = 0  # Start and end at 0, except for translation...
    x_bounds["q"][2, -1] = 1.57  # ...and end with cube 90 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Solve and animate the solution
    """

    model_path = "models/cube.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
