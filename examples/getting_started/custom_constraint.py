"""
File that shows an example of a custom constraint.
As an example, this custom constraint reproduces exactly the behavior of the ALIGN_MARKERS constraint.
"""
import biorbd
from casadi import vertcat

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
)


def custom_func_align_markers(ocp, nlp, t, x, u, p, first_marker_idx, second_marker_idx):
    nq = nlp["nbQ"]
    val = []
    for v in x:
        q = v[:nq]
        first_marker = nlp["model"].marker(q, first_marker_idx).to_mx()
        second_marker = nlp["model"].marker(q, second_marker_idx).to_mx()
        val = vertcat(val, first_marker - second_marker)
    return val


def prepare_ocp(biorbd_model_path, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        {
            "type": Constraint.CUSTOM,
            "function": custom_func_align_markers,
            "instant": Instant.START,
            "first_marker_idx": 0,
            "second_marker_idx": 1,
        },
        {
            "type": Constraint.CUSTOM,
            "function": custom_func_align_markers,
            "instant": Instant.END,
            "first_marker_idx": 0,
            "second_marker_idx": 2,
        },
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[1:6, [0, -1]] = 0
    X_bounds.max[1:6, [0, -1]] = 0
    X_bounds.min[2, -1] = 1.57
    X_bounds.max[2, -1] = 1.57

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque(), [torque_max] * biorbd_model.nbGeneralizedTorque(),
    )
    U_init = InitialConditions([torque_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    model_path = "cube.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
