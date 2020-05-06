import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Constraint,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
)

def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -1, 1, 0

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
        {"type": Objective.Lagrange.MINIMIZE_TIME, "minimum": 0, "maximum": 1.5, "weight": 1},
    )

    # Mapping
    tau_mapping = BidirectionalMapping(Mapping([0,-1],), Mapping([0]))

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    for i in range(biorbd_model.nbQ() + biorbd_model.nbQdot()):
        X_bounds.first_node_min[i] = 0
        X_bounds.first_node_max[i] = 0
        X_bounds.last_node_min[i] = 0
        X_bounds.last_node_max[i] = 0


    #Q_rotation
    X_bounds.last_node_min[1] = 3.14
    X_bounds.last_node_max[1] = 3.14

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = [
        Bounds(min_bound=[torque_min] * tau_mapping.reduce.len, max_bound=[torque_max] * tau_mapping.reduce.len)
    ]

    U_init = [InitialConditions([torque_init] * tau_mapping.reduce.len)]

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
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=2, number_shooting_points=20, show_online_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()