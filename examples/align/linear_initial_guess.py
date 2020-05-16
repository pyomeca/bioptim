import numpy as np
import biorbd

from biorbd_optim import (
    Instant,
    Axe,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
    InterpolationType,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, ode_solver, show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 4,},
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 5,},
        {
            "type": Constraint.ALIGN_MARKER_WITH_SEGMENT_AXIS,
            "instant": Instant.ALL,
            "marker_idx": 1,
            "segment_idx": 2,
            "axis": (Axe.X),
        },
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    for i in range(1, 8):
        if i != 3:
            X_bounds.min[i, [0, -1]] = 0
            X_bounds.max[i, [0, -1]] = 0
    X_bounds.min[2, -1] = 1.57
    X_bounds.max[2, -1] = 1.57

    # Initial guess
    q_init = np.array(((1.0, 0.0, 0.0, 0.46364761), (2.0, 0.0, 1.57, 0.78539785))).T
    q_dot_init = np.array(((0, 0, 0, 0), (0, 0, 0, 0))).T
    X_init = InitialConditions(np.concatenate((q_init, q_dot_init)), interpolation_type=InterpolationType.LINEAR)

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque(), [torque_max] * biorbd_model.nbGeneralizedTorque(),
    )
    tau_init = np.array(
        ((6.45997035, 10.47049302, 8.55628251, 3.57204643), (-4.46859579, 11.15084889, -10.15098446, 1.519847))
    ).T
    U_init = InitialConditions(tau_init, interpolation_type=InterpolationType.LINEAR)

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
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="cube_and_line.bioMod",
        number_shooting_points=30,
        final_time=1.0,
        ode_solver=OdeSolver.RK,
        show_online_optim=False,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
