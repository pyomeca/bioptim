"""
This is an example of the use of torque actuator using a model of 2segments and 2 degrees of freedom
"""
import biorbd_casadi as biorbd
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
    BiMappingList,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolver = OdeSolver.RK4(),
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
    ode_solver: OdeSolver
        The ode solver to use
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max, tau_init = -10, 10, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="residual_tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_residual_torque=True)

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    pose_at_first_node = [-0.75, 0.75]
    pose_at_final_node = [3.00, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot
    x_bounds[0][:, 2] = pose_at_final_node + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [-1] * bio_model.nb_tau + [tau_min] * bio_model.nb_tau, [1] * bio_model.nb_tau + [tau_max] * bio_model.nb_tau
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau * 2)

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
        assume_phase_dynamics=True,
    )


def main():
    """
    Prepares and solves an ocp with torque actuators, the animates it
    """

    ocp = prepare_ocp(
        biorbd_model_path=(
            "/home/lim/Documents/Anais/bioptim/bioptim/examples/torque_driven_ocp/models/2segments_2dof_2contacts.bioMod"
        ),
        n_shooting=30,
        final_time=2,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()
    sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
