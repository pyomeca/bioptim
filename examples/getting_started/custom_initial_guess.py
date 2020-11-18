import numpy as np
import biorbd

from bioptim import (
    Node,
    OptimalControlProgram,
    DynamicsTypeOption,
    DynamicsType,
    ObjectiveOption,
    Objective,
    ConstraintList,
    Constraint,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
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
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)

    # Path constraint and control path constraints
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57
    u_bounds = BoundsOption([[tau_min] * ntau, [tau_max] * ntau])

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if initial_guess == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [tau_init] * ntau
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
    x_init = InitialGuessOption(x, t=t, interpolation=initial_guess, **extra_params_x)

    u_init = InitialGuessOption(u, t=t, interpolation=initial_guess, **extra_params_u)
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
