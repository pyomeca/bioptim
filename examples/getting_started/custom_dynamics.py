"""
File that shows an example of a custom dynamic.
As an example, this custom constraint reproduces exactly the behavior of the TORQUE_DRIVEN problem_type and dynamic.
"""
import biorbd

from bioptim import (
    Instant,
    OptimalControlProgram,
    DynamicsTypeList,
    Problem,
    DynamicsType,
    DynamicsFunctions,
    ObjectiveOption,
    Objective,
    ConstraintList,
    Constraint,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    ShowResult,
    OdeSolver,
)


def custom_dynamic(states, controls, parameters, nlp):
    DynamicsFunctions.apply_parameters(parameters, nlp)
    q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

    qddot = nlp.model.ForwardDynamics(q, qdot, tau).to_mx()

    return qdot, qddot


def custom_configure(ocp, nlp):
    Problem.configure_q_qdot(nlp, as_states=True, as_controls=False)
    Problem.configure_tau(nlp, as_states=False, as_controls=True)
    Problem.configure_forward_dyn_func(ocp, nlp, custom_dynamic)


def prepare_ocp(biorbd_model_path, problem_type_custom=True, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsTypeList()
    if problem_type_custom:
        dynamics.add(custom_configure, dynamic_function=custom_dynamic)
    else:
        dynamics.add(DynamicsType.TORQUE_DRIVEN, dynamic_function=custom_dynamic)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.END, first_marker_idx=0, second_marker_idx=2)

    # Path constraint
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    # Initial guess
    x_init = InitialGuessOption([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsOption(
        [[tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque()]
    )

    u_init = InitialGuessOption([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
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
