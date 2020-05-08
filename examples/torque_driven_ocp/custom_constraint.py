"""
File that shows an example of using a custom constraint.
As an example, this custom constraint reproduces exactly the behavior of the ALIGN_MARKERS constraint.
"""
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
)

def custom_func_align_markers(ocp, nlp, t, x, u, first_marker_idx, second_marker_idx):
    nq = nlp["q_mapping"].reduce.len
    for v in x:
        q = nlp["q_mapping"].expand.map(v[:nq])
        first_marker = nlp["model"].marker(q, first_marker_idx).to_mx()
        second_marker = nlp["model"].marker(q, second_marker_idx).to_mx()

        return first_marker - second_marker

def prepare_ocp(biorbd_model_path, show_online_optim=False, ode_solver=OdeSolver.RK):
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
        {"type": Constraint.CUSTOM, "function":custom_func_align_markers, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 1,},
        {"type": Constraint.CUSTOM, "function": custom_func_align_markers, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    for i in range(1, 6):
        X_bounds.first_node_min[i] = 0
        X_bounds.last_node_min[i] = 0
        X_bounds.first_node_max[i] = 0
        X_bounds.last_node_max[i] = 0
    X_bounds.last_node_min[2] = 1.57
    X_bounds.last_node_max[2] = 1.57

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
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        ode_solver=ode_solver,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    model_path = "cube.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path, show_online_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
