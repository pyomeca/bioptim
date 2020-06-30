import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
    Data,
)


def prepare_ocp(
    final_time, time_min, time_max, number_shooting_points, biorbd_model_path="cube.bioMod", ode_solver=OdeSolver.RK
):
    # --- Options --- #
    nb_phases = len(number_shooting_points)
    if nb_phases > 3:
        raise RuntimeError("Number of phases must be 1 to 3")

    # Model path
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = (
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
    )

    # Dynamics
    problem_type = (
        {"type": ProblemType.TORQUE_DRIVEN},
        {"type": ProblemType.TORQUE_DRIVEN},
        {"type": ProblemType.TORQUE_DRIVEN},
    )

    # Constraints
    constraints = (
        (
            {
                "type": Constraint.ALIGN_MARKERS,
                "instant": Instant.START,
                "first_marker_idx": 0,
                "second_marker_idx": 1,
            },
            {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
            {"type": Constraint.TIME_CONSTRAINT, "minimum": time_min[0], "maximum": time_max[0],},
        ),
        (
            {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 1,},
            {"type": Constraint.TIME_CONSTRAINT, "minimum": time_min[1], "maximum": time_max[1],},
        ),
        (
            {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
            {"type": Constraint.TIME_CONSTRAINT, "minimum": time_min[2], "maximum": time_max[2],},
        ),
    )

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0])]

    for bounds in X_bounds:
        for i in [1, 3, 4, 5]:
            bounds.min[i, [0, -1]] = 0
            bounds.max[i, [0, -1]] = 0
    X_bounds[0].min[2, 0] = 0.0
    X_bounds[0].max[2, 0] = 0.0
    X_bounds[2].min[2, [0, -1]] = [0.0, 1.57]
    X_bounds[2].max[2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    X_init = (X_init, X_init, X_init)

    # Define control path constraint
    U_bounds = [
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
    ]
    U_init = InitialConditions([torque_init] * biorbd_model[0].nbGeneralizedTorque())
    U_init = (U_init, U_init, U_init)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model[:nb_phases],
        problem_type[:nb_phases],
        number_shooting_points[:nb_phases],
        final_time[:nb_phases],
        X_init[:nb_phases],
        U_init[:nb_phases],
        X_bounds[:nb_phases],
        U_bounds[:nb_phases],
        objective_functions[:nb_phases],
        constraints[:nb_phases],
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    ns = (20, 30, 20)
    ocp = prepare_ocp(final_time=final_time, time_min=time_min, time_max=time_max, number_shooting_points=ns)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    print(f"The optimized phase time are: {param['time'][0, 0]}s, {param['time'][1, 0]}s and {param['time'][2, 0]}s.")

    result = ShowResult(ocp, sol)
    result.animate()
