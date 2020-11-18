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
    InitialGuessOption,
    ShowResult,
    InterpolationType,
)


def custom_x_bounds_min(current_shooting_point, n_elements, nb_shooting):
    my_values = np.array([[-10, -5]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / nb_shooting


def custom_x_bounds_max(current_shooting_point, n_elements, nb_shooting):
    my_values = np.array([[10, 5]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / nb_shooting


def custom_u_bounds_min(current_shooting_point, n_elements, nb_shooting):
    my_values = np.array([[-20, -10]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / nb_shooting


def custom_u_bounds_max(current_shooting_point, n_elements, nb_shooting):
    my_values = np.array([[20, 10]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / nb_shooting


def prepare_ocp(
    biorbd_model_path,
    number_shooting_points,
    final_time,
    interpolation_type=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE)

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)

    # Path constraints
    if interpolation_type == InterpolationType.CONSTANT:
        x_min = [-100] * (nq + nqdot)
        x_max = [100] * (nq + nqdot)
        x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.CONSTANT)
        u_min = [tau_min] * ntau
        u_max = [tau_max] * ntau
        u_bounds = BoundsOption([u_min, u_max], interpolation=InterpolationType.CONSTANT)
    elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x_min = np.random.random((6, 3)) * (-10) - 5
        x_max = np.random.random((6, 3)) * 10 + 5
        x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
        u_min = np.random.random((3, 3)) * tau_min + tau_min / 2
        u_max = np.random.random((3, 3)) * tau_max + tau_max / 2
        u_bounds = BoundsOption([u_min, u_max], interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    elif interpolation_type == InterpolationType.LINEAR:
        x_min = np.random.random((6, 2)) * (-10) - 5
        x_max = np.random.random((6, 2)) * 10 + 5
        x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.LINEAR)
        u_min = np.random.random((3, 2)) * tau_min + tau_min / 2
        u_max = np.random.random((3, 2)) * tau_max + tau_max / 2
        u_bounds = BoundsOption([u_min, u_max], interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.EACH_FRAME:
        x_min = np.random.random((nq + nqdot, number_shooting_points + 1)) * (-10) - 5
        x_max = np.random.random((nq + nqdot, number_shooting_points + 1)) * 10 + 5
        x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.EACH_FRAME)
        u_min = np.random.random((ntau, number_shooting_points)) * tau_min + tau_min / 2
        u_max = np.random.random((ntau, number_shooting_points)) * tau_max + tau_max / 2
        u_bounds = BoundsOption([u_min, u_max], interpolation=InterpolationType.EACH_FRAME)
    elif interpolation_type == InterpolationType.SPLINE:
        spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        x_min = np.random.random((nq + nqdot, 5)) * (-10) - 5
        x_max = np.random.random((nq + nqdot, 5)) * 10 + 5
        u_min = np.random.random((ntau, 5)) * tau_min + tau_min / 2
        u_max = np.random.random((ntau, 5)) * tau_max + tau_max / 2
        x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.SPLINE, t=spline_time)
        u_bounds = BoundsOption([u_min, u_max], interpolation=InterpolationType.SPLINE, t=spline_time)
    elif interpolation_type == InterpolationType.CUSTOM:
        # The custom functions refer to the ones at the beginning of the file.
        # For this particular instance, they emulate a Linear interpolation
        extra_params_x = {"n_elements": nq + nqdot, "nb_shooting": number_shooting_points}
        extra_params_u = {"n_elements": ntau, "nb_shooting": number_shooting_points}
        x_bounds = BoundsOption(
            [custom_x_bounds_min, custom_x_bounds_max], interpolation=InterpolationType.CUSTOM, **extra_params_x
        )
        u_bounds = BoundsOption(
            [custom_u_bounds_min, custom_u_bounds_max], interpolation=InterpolationType.CUSTOM, **extra_params_u
        )
    else:
        raise NotImplementedError("Not implemented yet")

    # Initial guess
    x_init = InitialGuessOption([0] * (nq + nqdot))
    u_init = InitialGuessOption([tau_init] * ntau)
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
    print(f"Show the bounds")
    for interpolation_type in InterpolationType:
        print(f"Solving problem using {interpolation_type} bounds")
        ocp = prepare_ocp("cube.bioMod", number_shooting_points=30, final_time=2, interpolation_type=interpolation_type)
        sol = ocp.solve()
        print("\n")

        # Print the last solution
        result_plot = ShowResult(ocp, sol)
        result_plot.graphs(adapt_graph_size_to_bounds=True)

    for interpolation_type in InterpolationType:
        print(f"Solving problem using {interpolation_type} bounds")
        ocp = prepare_ocp("cube.bioMod", number_shooting_points=30, final_time=2, interpolation_type=interpolation_type)
        sol = ocp.solve()
        print("\n")

        # Print the last solution
        result_plot = ShowResult(ocp, sol)
        result_plot.graphs(adapt_graph_size_to_bounds=False)
