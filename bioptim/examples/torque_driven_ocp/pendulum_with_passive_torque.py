"""
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
this time there is a passive torque which is applied on Seg1 in the model "pendulum_with_passive_torque.bioMod".
The expression of the tau is therefore not the same here.
"""

import platform

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    BiorbdModel,
    RigidBodyDynamics,
    BoundsList,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    rigidbody_dynamics=RigidBodyDynamics.DAE_INVERSE_DYNAMICS,
    with_passive_torque=False,
    assume_phase_dynamics: bool = True,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    rigidbody_dynamics : RigidBodyDynamics
        rigidbody dynamics DAE or ODE
    with_passive_torque: bool
        If the passive torque is used in dynamics
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

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        rigidbody_dynamics=rigidbody_dynamics,
        with_passive_torque=with_passive_torque,
        expand=expand_dynamics,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    tau_min, tau_max = -100, 100
    qddot_min, qddot_max = -1000, 1000

    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively rotate

    if (
        rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
    ):
        u_bounds["qddot"] = [qddot_min] * bio_model.nb_qddot, [qddot_max] * bio_model.nb_qddot

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
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path="models/pendulum_with_passive_torque.bioMod",
        final_time=1,
        n_shooting=30,
        with_passive_torque=False,
    )

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    # sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
