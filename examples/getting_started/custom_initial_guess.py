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


def custom_init_func(current_shooting_point, my_values, nb_shooting):
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / nb_shooting


def prepare_ocp(
    biorbd_model_path,
    number_shooting_points,
    final_time,
    initial_guess=InterpolationType.CONSTANT,
    boundsInterpolation=False,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}

    # Dynamics
    problem_type = {"type": ProblemType.TORQUE_DRIVEN}

    # Constraints
    constraints = (
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 1,},
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
    )

    spline_time = None
    # Path constraint and control path constraints
    if boundsInterpolation:
        if initial_guess == InterpolationType.SPLINE:
            spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
            x_min = np.random.random((nq + nqdot, 5)) - 100
            x_max = np.random.random((nq + nqdot, 5)) + 100
            u_min = np.random.random((ntau, 5)) - 100
            u_max = np.random.random((ntau, 5)) + 100
            X_bounds = Bounds(x_min, x_max, interpolation_type=InterpolationType.SPLINE)
            U_bounds = Bounds(u_min, u_max, interpolation_type=InterpolationType.SPLINE)
        elif initial_guess == InterpolationType.CUSTOM:
            x_min = np.random.random((nq + nqdot, 2)) - 100
            x_max = np.random.random((nq + nqdot, 2)) + 100
            u_min = np.random.random((ntau, 2)) - 100
            u_max = np.random.random((ntau, 2)) + 100
            X_bounds = Bounds(x_min, x_max, interpolation_type=InterpolationType.CUSTOM)
            U_bounds = Bounds(u_min, u_max, interpolation_type=InterpolationType.CUSTOM)
        else:
            X_bounds = QAndQDotBounds(biorbd_model)
            X_bounds.min[1:6, [0, -1]] = 0
            X_bounds.max[1:6, [0, -1]] = 0
            X_bounds.min[2, -1] = 1.57
            X_bounds.max[2, -1] = 1.57
            U_bounds = Bounds([torque_min] * ntau, [torque_max] * ntau,)
    else:
        X_bounds = QAndQDotBounds(biorbd_model)
        X_bounds.min[1:6, [0, -1]] = 0
        X_bounds.max[1:6, [0, -1]] = 0
        X_bounds.min[2, -1] = 1.57
        X_bounds.max[2, -1] = 1.57
        U_bounds = Bounds([torque_min] * ntau, [torque_max] * ntau,)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
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
    elif initial_guess == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))
    elif initial_guess == InterpolationType.CUSTOM:
        # The custom function refers to the one at the beginning of the file. It emulates a Linear interpolation
        x = custom_init_func
        u = custom_init_func
        extra_params_x = {"my_values": np.random.random((nq + nqdot, 2)), "nb_shooting": number_shooting_points}
        extra_params_u = {"my_values": np.random.random((ntau, 2)), "nb_shooting": number_shooting_points}
    else:
        raise RuntimeError("Initial guess not implemented yet")
    X_init = InitialConditions(x, t=t, interpolation_type=initial_guess, extra_params=extra_params_x)
    U_init = InitialConditions(u, t=t, interpolation_type=initial_guess, extra_params=extra_params_u)
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
