import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions
from biorbd_optim.plot import ShowResult


def prepare_ocp(biorbd_model_path="eocar.bioMod", show_online_optim=True):
    # --- Options --- #
    # Model path
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    number_shooting_points = (20, 30)
    final_time = (2, 5)
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = (
        (
            {"type": ObjectiveFunction.minimize_torque, "weight": 100},
            {"type": ObjectiveFunction.cyclic, "weight": 100},
        ),
        (
            {"type": ObjectiveFunction.minimize_torque, "weight": 100},
            {"type": ObjectiveFunction.cyclic, "weight": 100},
        ),
    )

    # Dynamics
    variable_type = (ProblemType.torque_driven, ProblemType.torque_driven)

    # Constraints
    constraints = (
        (
            {
                "type": Constraint.Type.MARKERS_TO_PAIR,
                "instant": Constraint.Instant.START,
                "first_marker": 0,
                "second_marker": 1,
            },
            {
                "type": Constraint.Type.MARKERS_TO_PAIR,
                "instant": Constraint.Instant.END,
                "first_marker": 0,
                "second_marker": 2,
            },
        ),
        (
            {
                "type": Constraint.Type.MARKERS_TO_PAIR,
                "instant": Constraint.Instant.END,
                "first_marker": 0,
                "second_marker": 1,
            },
        ),
    )

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0])]

    for bounds in X_bounds:
        for i in range(6):
            if i != 0 and i != 2:
                bounds.first_node_min[i] = 0
                bounds.last_node_min[i] = 0
                bounds.first_node_max[i] = 0
                bounds.last_node_max[i] = 0
    X_bounds[0].first_node_min[2] = 0.0
    X_bounds[0].first_node_max[2] = 0.0
    # X_bounds[0].last_node_min[2] = 1.57
    # X_bounds[0].last_node_max[2] = 1.57
    # X_bounds[1].last_node_min[2] = 1.0
    # X_bounds[1].last_node_max[2] = 1.0

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    U_bounds = [
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
    ]
    U_init = InitialConditions([torque_init] * biorbd_model[0].nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        objective_functions,
        (X_init, X_init),
        (U_init, U_init),
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.show_biorbd_viz()
