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
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    BiorbdModel,
    RigidBodyDynamics,
    Bounds,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    rigidbody_dynamics=RigidBodyDynamics.DAE_INVERSE_DYNAMICS,
    with_passive_torque=False,
    assume_phase_dynamics: bool = True,
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

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN, rigidbody_dynamics=rigidbody_dynamics, with_passive_torque=with_passive_torque
    )

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    qddot_min, qddot_max, qddot_init = -1000, 1000, 0

    if rigidbody_dynamics == RigidBodyDynamics.ODE:
        u_bounds = Bounds(
            [tau_min] * bio_model.nb_tau,
            [tau_max] * bio_model.nb_tau,
        )
        u_init = InitialGuess([tau_init] * bio_model.nb_tau)
        u_bounds[1, :] = 0  # Prevent the model from actively rotate

    elif (
        rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
    ):
        u_bounds = Bounds(
            [tau_min] * bio_model.nb_tau + [qddot_min] * bio_model.nb_qddot,
            [tau_max] * bio_model.nb_tau + [qddot_max] * bio_model.nb_qddot,
        )
        u_init = InitialGuess([tau_init] * bio_model.nb_tau + [qddot_init] * bio_model.nb_qddot)
        u_bounds[1, :] = 0
    else:
        raise NotImplementedError("dynamic not implemented yet")

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
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
    sol.detailed_cost_values()
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
