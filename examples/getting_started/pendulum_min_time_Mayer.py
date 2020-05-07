import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Objective,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
        {"type": Objective.Mayer.MINIMIZE_TIME},
    )

    # Mapping
    tau_mapping = BidirectionalMapping(Mapping([0, -1],), Mapping([0]))

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.first_node_min = [0, 0, 0, 0]
    X_bounds.first_node_max = [0, 0, 0, 0]
    X_bounds.last_node_min = [0, 3.14, 0, 0]
    X_bounds.last_node_max = [0, 3.14, 0, 0]

    # Initial guess
    X_init = InitialConditions([0, 0, 0, 0])

    # Define control path constraint
    U_bounds = [Bounds(min_bound=[torque_min], max_bound=[torque_max])]

    U_init = [InitialConditions([torque_init])]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        tau_mapping=tau_mapping,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="pendulum.bioMod", final_time=2, number_shooting_points=50, show_online_optim=False,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    param = Data.get_data_from_V(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    print(f"The optimized phase time is: {param['time'][0, 0]}, good job Mayer!")

    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
