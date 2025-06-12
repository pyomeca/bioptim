"""
This is a simple example in which a mass is dropped and held by a ligament that plays the role of a spring without
damping, it uses the model mass_point_with_ligament.bioMod
"""

from bioptim import (
    OptimalControlProgram,
    Dynamics,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    TorqueBiorbdModel,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str,
    use_sx: bool = False,
    ode_solver=OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
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
    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # ConfigureProblem parameters
    number_shooting_points = 100
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0
    qddot_min, qddot_max, qddot_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10000000)

    # Dynamics
    dynamics = Dynamics(
        ode_solver=ode_solver,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
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

    return OptimalControlProgram(
        bio_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        u_init=u_init,
        objective_functions=objective_functions,
        n_threads=n_threads,
        use_sx=use_sx,
    )


def main():
    model_path = "./models/mass_point_with_ligament.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve()
    sol.graphs()


if __name__ == "__main__":
    main()
