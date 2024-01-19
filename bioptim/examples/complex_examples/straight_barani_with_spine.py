from bioptim import (
    BiorbdModel,
    BoundsList,
    ConstraintList,
    DynamicsFcn,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    use_sx: bool = True,
) -> OptimalControlProgram:
    """
    This function build an optimal control program and instantiate it.
    It can be seen as a factory for the OptimalControlProgram class.

    Parameters
    ----------
    # TODO fill this section

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Declaration of generic elements
    n_shooting = (40, 40)
    phase_time = (0.85, 0.85)
    nb_phases = 2

    bio_model = [BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path)]

    # Declaration of the constraints and objectives of the ocp
    constraints = ConstraintList()
    objective_functions = ObjectiveList()

    for i in range(nb_phases):
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            weight=1.0,
            key="qddot_joints",
            node=Node.ALL_SHOOTING,
            quadratic=True,
            phase=i,
        )

        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            weight=1.0,
            key="qddot_joints",
            node=Node.ALL_SHOOTING,
            quadratic=True,
            derivative=True,
            phase=i,
        )

        objective_functions.add(
            objective=ObjectiveFcn.Mayer.MINIMIZE_TIME,
            weight=1.0,
            min_bound=0.1,
            max_bound=2.0,
            node=Node.END,
            quadratic=True,
            phase=i,
        )

        stomach_rotx, stomach_roty, stomach_rotz = 6, 7, 8
        ribcage_rotx, ribcage_roty, ribcage_rotz = 9, 10, 11
        nipple_rotx, nipple_roty, nipple_rotz = 12, 13, 14
        shoulder_rotx, shoulder_roty, shoulder_rotz = 15, 16, 17

        for spine_dof1, spine_dof2 in (
            (stomach_rotx, ribcage_rotx),
            (stomach_roty, ribcage_roty),
            (stomach_rotz, ribcage_rotz),
            (ribcage_rotx, nipple_rotx),
            (ribcage_roty, nipple_roty),
            (ribcage_rotz, nipple_rotz),
            (nipple_rotx, shoulder_rotx),
            (nipple_roty, shoulder_roty),
            (nipple_rotz, shoulder_rotz),
        ):
            objective_functions.add(
                objective=ObjectiveFcn.Lagrange.PROPORTIONAL_CONTROL,
                weight=10.0,
                key="qddot_joints",
                first_dof=spine_dof1 - 6,
                second_dof=spine_dof2 - 6,
                coef=1.0,
                node=Node.ALL_SHOOTING,
                quadratic=True,
                phase=i,
            )

    # minimize arm states
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[18, 19, 20, 21],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    # minimize tilt state
    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_STATE,
        weight=100.0,
        key="q",
        index=[4],
        node=Node.ALL,
        quadratic=True,
        phase=0,
    )

    # MINIMIZE spine state during somersault
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    # minimize tilt state during landing
    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_STATE,
        weight=1000.0,
        key="q",
        index=[4],
        node=Node.END,
        quadratic=True,
        phase=1,
    )

    # Declaration of the dynamics function used during integration
    dynamics = DynamicsList()

    for i in range(nb_phases):
        dynamics.add(
            DynamicsFcn.JOINTS_ACCELERATION_DRIVEN,
            phase=i,
        )

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(
        MultinodeConstraintFcn.TRACK_TOTAL_TIME,
        nodes_phase=(0, 1),
        nodes=(Node.END, Node.END),
        min_bound=final_time - 0.1,
        max_bound=final_time + 0.1,
    )

    # Declaration of optimization variables bounds and initial guesses
    # Path constraint
    x_bounds = BoundsList()
    x_initial_guesses = InitialGuessList()

    u_bounds = BoundsList()
    u_initial_guesses = InitialGuessList()

    x_bounds.add(
        "q",
        min_bound=[
            [-0.0, -1.0, -1.0],
            [-0.0, -1.0, -1.0],
            [-0.0, -0.1, -0.1],
            [0.0, -0.1, 4.61],
            [0.0, -0.79, -0.79],
            [0.0, -0.99, 3.04],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.26, -0.26],
            [0.0, -0.65, -0.65],
            [2.9, -0.05, -0.05],
            [0.0, -2.0, -2.0],
            [-2.9, -3.0, -3.0],
        ],
        max_bound=[
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 15.0, 15.0],
            [0.0, 4.81, 4.81],
            [0.0, 0.79, 0.79],
            [0.0, 4.13, 3.24],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 0.26, 0.26],
            [0.0, 2.0, 2.0],
            [2.9, 3.0, 3.0],
            [0.0, 0.65, 0.65],
            [-2.9, 0.05, 0.05],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=0,
    )

    x_bounds.add(
        "q",
        min_bound=[
            [-1.0, -1.0, -0.01],
            [-1.0, -1.0, -0.01],
            [-0.1, -0.1, 0.0],
            [4.61, 4.61, 6.18],
            [-0.2, -0.2, -0.2],
            [3.04, 3.04, 3.04],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.01],
            [-0.26, -0.26, -0.26],
            [-0.65, -0.65, -0.1],
            [0.0, -0.05, 2.8],
            [-2.0, -2.0, -0.1],
            [-0.79, -3.0, -3.0],
        ],
        max_bound=[
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
            [15.0, 15.0, 0.01],
            [4.81, 6.38, 6.38],
            [0.2, 0.2, 0.2],
            [3.24, 3.24, 3.24],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.01],
            [0.26, 0.26, 0.26],
            [2.0, 2.0, 0.1],
            [0.79, 3.0, 3.0],
            [0.65, 0.65, 0.1],
            [0.0, 0.05, -2.8],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=1,
    )

    x_bounds.add(
        "qdot",
        min_bound=[
            [-0.5, -10.0, -10.0],
            [-0.5, -10.0, -10.0],
            [2.9, -100.0, -100.0],
            [0.5, 0.5, 0.5],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
        ],
        max_bound=[
            [0.5, 10.0, 10.0],
            [0.5, 10.0, 10.0],
            [6.9, 100.0, 100.0],
            [200.0, 200.0, 200.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=0,
    )

    x_bounds.add(
        "qdot",
        min_bound=[
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
            [-100.0, -100.0, -100.0],
            [0.5, 0.5, 0.5],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ],
        max_bound=[
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=1,
    )

    for i in range(nb_phases):
        u_bounds.add(
            "qddot_joints",
            min_bound=[
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
            ],
            max_bound=[
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
            ],
            interpolation=InterpolationType.CONSTANT,
            phase=i,
        )

    x_initial_guesses.add(
        "q",
        initial_guess=[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 7.45],
            [0.0, 4.71],
            [0.0, 0.0],
            [0.0, 3.14],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.68],
            [2.9, 1.48],
            [0.0, -0.68],
            [-2.9, -1.48],
        ],
        interpolation=InterpolationType.LINEAR,
        phase=0,
    )

    x_initial_guesses.add(
        "q",
        initial_guess=[
            [0.0, 0.0],
            [0.0, 0.0],
            [7.45, 0.0],
            [4.71, 6.28],
            [0.0, 0.0],
            [3.14, 3.14],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.68, 0.0],
            [0.39, 2.9],
            [-0.68, 0.0],
            [-0.39, -2.9],
        ],
        interpolation=InterpolationType.LINEAR,
        phase=1,
    )

    x_initial_guesses.add(
        "qdot",
        initial_guess=[
            [0.0],
            [0.0],
            [2.45],
            [6.28],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        interpolation=InterpolationType.CONSTANT,
        phase=0,
    )

    x_initial_guesses.add(
        "qdot",
        initial_guess=[
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        interpolation=InterpolationType.CONSTANT,
        phase=1,
    )

    for i in range(nb_phases):
        u_initial_guesses.add(
            "qddot_joints",
            initial_guess=[
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ],
            interpolation=InterpolationType.CONSTANT,
            phase=i,
        )

    # Construct and return the optimal control program (OCP)
    return OptimalControlProgram(
        bio_model=bio_model,
        n_shooting=n_shooting,
        phase_time=phase_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_initial_guesses,
        u_init=u_initial_guesses,
        objective_functions=objective_functions,
        use_sx=use_sx,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
    )


def main():
    # --- Prepare the multi-start and run it --- #
    ocp = prepare_ocp(
        biorbd_model_path="models/straight_barani_with_spine.bioMod",
        final_time=1.7,
        use_sx=False,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(3000)
    solver.set_convergence_tolerance(1e-6)

    sol = ocp.solve(solver=solver)

    # --- Show results --- #
    sol.print_cost()
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
