"""
This is an example of the use of torque actuator using a model of 2segments and 2 degrees of freedom
"""

import platform

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
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
    ode_solver: OdeSolverBase
        The ode solver to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note that with_contact is set to True, then this must be
        set to False if ode_solver is IRK

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max = -10, 10

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="residual_tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_residual_torque=True, expand=expand_dynamics)

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    pose_at_first_node = [-0.75, 0.75]
    pose_at_final_node = [3.00, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = pose_at_first_node
    x_bounds["q"][:, 2] = pose_at_final_node
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, 2]] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = pose_at_first_node

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [-1] * bio_model.nb_tau, [1] * bio_model.nb_tau
    u_bounds["residual_tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Prepares and solves an ocp with torque actuators, the animates it
    """

    ocp = prepare_ocp(
        biorbd_model_path="models/2segments_2dof_2contacts.bioMod",
        n_shooting=30,
        final_time=2,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()
    sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
