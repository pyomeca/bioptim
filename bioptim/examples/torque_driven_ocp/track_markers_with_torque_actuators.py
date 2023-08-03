"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end. It is a clone of
'getting_started/custom_constraint.py' It is designed to show how to use the TORQUE_ACTIVATIONS_DRIVEN which limits
the torque to [-1; 1]. This is useful when the maximal torque are not constant. Please note that this dynamic then
to not converge when it is used on more complicated model. A solution that defines non-constant constraints seems a
better idea. An example of which can be found with the bioptim paper.
"""

import platform

from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    actuator_type: int = None,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    assume_phase_dynamics: bool = True,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        Path to the bioMod
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at final node
    actuator_type: int
        The type of actuator to use: 1 (torque activations) or 2 (torque max constraints)
    ode_solver: OdeSolverBase
        The ode solver to use
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
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)

    # Problem parameters
    if actuator_type and actuator_type == 1:
        tau_min, tau_max = -1, 1
    else:
        tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    if actuator_type:
        if actuator_type == 1:
            dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, expand=expand_dynamics)
        elif actuator_type == 2:
            dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand_dynamics)
        else:
            raise ValueError("actuator_type is 1 (torque activations) or 2 (torque max constraints)")
    else:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand_dynamics)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")
    if actuator_type == 2:
        constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, node=Node.ALL_SHOOTING, min_torque=7.5)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds[0]["q"][2, [0, -1]] = [0, 1.57]
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds[0]["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    # ------------- #

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
    Prepares and solves an ocp with torque actuators, the animates it
    """

    ocp = prepare_ocp("models/cube.bioMod", n_shooting=30, final_time=2, actuator_type=2)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
