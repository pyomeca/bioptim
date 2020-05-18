import numpy as np
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
    InterpolationType,
)


def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, initial_guess=InterpolationType.CONSTANT):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()

    # Problem parameters
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}

    # Dynamics
    problem_type = ProblemType.torque_driven

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

    # Define control path constraint
    U_bounds = Bounds([torque_min] * ntau, [torque_max] * ntau,)

    # Initial guesses
    if initial_guess == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [torque_init] * ntau
    elif initial_guess == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T
    elif initial_guess == InterpolationType.LINEAR:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T
    elif initial_guess == InterpolationType.EACH_FRAME:
        x = np.random.random((nq + nqdot, number_shooting_points + 1))
        u = np.random.random((ntau, number_shooting_points))
    else:
        raise RuntimeError("Initial guess not implemented yet")
    X_init = InitialConditions(x, interpolation_type=initial_guess)
    U_init = InitialConditions(u, interpolation_type=initial_guess)
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
    )


if __name__ == "__main__":
    for initial_guess in InterpolationType:
        print(f"Solving problem using {initial_guess} initial guess")
        ocp = prepare_ocp("cube.bioMod", number_shooting_points=30, final_time=2, initial_guess=initial_guess)
        sol = ocp.solve()
        print("\n")

    # Print the last solution
    result_plot = ShowResult(ocp, sol)
    result_plot.graphs()
