"""
This is a simple example in which a mass is dropped and held by a ligament that plays the role of a spring without
damping, it uses the model mass_point_with_ligament.bioMod
"""

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    BiorbdModel,
    DynamicsFcn,
    RigidBodyDynamics,
)


def prepare_ocp(
    biorbd_model_path: str,
    use_sx: bool = False,
    ode_solver=OdeSolver.RK4(),
    rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
    assume_phase_dynamics: bool = True,
    n_threads: int = 8,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    use_sx: bool
        If the project should be build in mx [False] or sx [True]
    ode_solver: OdeSolverBase
        The type of integrator
    rigidbody_dynamics: RigidBodyDynamics
        The rigidbody dynamics to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
    n_threads: int
        Number of threads to use
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    # Model path
    bio_model = BiorbdModel(biorbd_model_path)

    # ConfigureProblem parameters
    number_shooting_points = 100
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0
    qddot_min, qddot_max, qddot_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10000000)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        rigidbody_dynamics=rigidbody_dynamics,
        with_ligament=True,
        expand=expand_dynamics,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][0, 0] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][0, 0] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialGuessList()

    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau
    u_init["tau"] = [tau_init] * bio_model.nb_tau
    if (
        rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
        or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
    ):
        u_bounds["qddot"] = [qddot_min] * bio_model.nb_qddot, [qddot_max] * bio_model.nb_qddot
        u_init["qddot"] = [qddot_init] * bio_model.nb_qddot

    return OptimalControlProgram(
        bio_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        u_init=u_init,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        n_threads=n_threads,
        use_sx=use_sx,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    model_path = "./models/mass_point_with_ligament.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve()
    sol.graphs()


if __name__ == "__main__":
    main()
