"""
This is an example of the use of torque actuator using a model of 2segments and 2 degrees of freedom
"""

from bioptim import (
    TorqueActivationBiorbdModel,
    OptimalControlProgram,
    DynamicsOptionsList,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    PhaseDynamics,
    OnlineOptim,
)
from bioptim.examples.utils import ExampleUtils


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
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
    bio_model = TorqueActivationBiorbdModel(biorbd_model_path, with_residual_torque=True)
    tau_min, tau_max = -10, 10

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="residual_tau", weight=100)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(
        ode_solver=ode_solver,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

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
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
    )


def main():
    """
    Prepares and solves an ocp with torque actuators, the animates it
    """

    ocp = prepare_ocp(
        biorbd_model_path=ExampleUtils.folder + "/models/2segments_2dof_2contacts.bioMod",
        n_shooting=30,
        final_time=2,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(online_optim=OnlineOptim.DEFAULT))

    # --- Show results --- #
    sol.animate()
    sol.print_cost()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
