"""
File that shows an example of a custom dynamic.
As an example, this custom constraint reproduces exactly the behavior of the TORQUE_DRIVEN problem_type and dynamic.
"""
import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Problem,
    Dynamics,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
)


def custom_dynamic(states, controls, parameters, nlp):
    Dynamics.apply_parameters(parameters, nlp)
    q, qdot, tau = Dynamics.dispatch_q_qdot_tau_data(states, controls, nlp)

    qddot = nlp["model"].ForwardDynamics(q, qdot, tau).to_mx()

    return (qdot, qddot)


def custom_configure(ocp, nlp):
    Problem.configure_q_qdot(nlp, as_states=True, as_controls=False)
    Problem.configure_tau(nlp, as_states=False, as_controls=True)
    Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)


def prepare_ocp(biorbd_model_path, problem_type_custom=True, ode_solver=OdeSolver.RK):
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
    if problem_type_custom:
        problem_type = {"type": ProblemType.CUSTOM, "configure": custom_configure, "dynamic": custom_dynamic}
    else:
        problem_type = {"type": ProblemType.TORQUE_DRIVEN, "dynamic": custom_dynamic}

    # Constraints
    constraints = (
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 1,},
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
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
