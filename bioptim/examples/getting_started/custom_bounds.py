"""
This example is a trivial box sent upward. It is designed to investigate the different
bounds one can define in bioptim.
Therefore, it shows how one can define the bounds, that is the minimal and maximal values
of the state and control variables.

All the types of interpolation are shown:
InterpolationType.CONSTANT: All the values are the same at each node
InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT: Same as constant, but have the first
    and last nodes different. This is particularly useful when you want to fix the initial and
    final position and leave the rest of the movement free.
InterpolationType.LINEAR: The values are linearly interpolated between the first and last nodes.
InterpolationType.EACH_FRAME: Each node values are specified
InterpolationType.SPLINE: The values are interpolated from the first to last node using a cubic spline
InterpolationType.CUSTOM: Provide a user-defined interpolation function
"""

import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    InitialGuess,
    InterpolationType,
)


def custom_x_bounds_min(current_shooting_point: int, n_elements: int, n_shooting: int) -> np.ndarray:
    """
    The custom function for the x bound (this particular one mimics linear interpolation)

    Parameters
    ----------
    current_shooting_point: int
        The current point to return the value, it is defined between [0; n_shooting] for the states
        and [0; n_shooting[ for the controls
    n_elements: int
        The number of rows of the matrix
    n_shooting: int
        The number of shooting point

    Returns
    -------
    The vector value of the bounds at current_shooting_point
    """

    my_values = np.array([[-10, -5]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / n_shooting


def custom_x_bounds_max(current_shooting_point: int, n_elements: int, n_shooting: int) -> np.ndarray:
    """
    The custom function for the x bound (this particular one mimics linear interpolation)

    Parameters
    ----------
    current_shooting_point: int
        The current point to return the value, it is defined between [0; n_shooting] for the states
        and [0; n_shooting[ for the controls
    n_elements: int
        The number of rows of the matrix
    n_shooting: int
        The number of shooting point

    Returns
    -------
    The vector value of the bounds at current_shooting_point
    """

    my_values = np.array([[10, 5]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / n_shooting


def custom_u_bounds_min(current_shooting_point: int, n_elements: int, n_shooting: int) -> np.ndarray:
    """
    The custom function for the x bound (this particular one mimics linear interpolation)

    Parameters
    ----------
    current_shooting_point: int
        The current point to return the value, it is defined between [0; n_shooting] for the states
        and [0; n_shooting[ for the controls
    n_elements: int
        The number of rows of the matrix
    n_shooting: int
        The number of shooting point

    Returns
    -------
    The vector value of the bounds at current_shooting_point
    """

    my_values = np.array([[-20, -10]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / n_shooting


def custom_u_bounds_max(current_shooting_point: int, n_elements: int, n_shooting: int) -> np.ndarray:
    """
    The custom function for the x bound (this particular one mimics linear interpolation)

    Parameters
    ----------
    current_shooting_point: int
        The current point to return the value, it is defined between [0; n_shooting] for the states
        and [0; n_shooting[ for the controls
    n_elements: int
        The number of rows of the matrix
    n_shooting: int
        The number of shooting point

    Returns
    -------
    The vector value of the bounds at current_shooting_point
    """

    my_values = np.array([[20, 10]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / n_shooting


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    interpolation_type: InterpolationType = InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
) -> OptimalControlProgram:
    """
    Prepare the ocp for the specified interpolation type

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    n_shooting: int
        The number of shooting point
    final_time: float
        The movement time
    interpolation_type: InterpolationType
        The requested InterpolationType

    Returns
    -------
    The OCP fully prepared and ready to be solved
    """

    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # Path constraints
    if interpolation_type == InterpolationType.CONSTANT:
        x_min = [-100] * (nq + nqdot)
        x_max = [100] * (nq + nqdot)
        x_bounds = Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT)
        u_min = [tau_min] * ntau
        u_max = [tau_max] * ntau
        u_bounds = Bounds(u_min, u_max, interpolation=InterpolationType.CONSTANT)
    elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x_min = np.random.random((6, 3)) * (-10) - 5
        x_max = np.random.random((6, 3)) * 10 + 5
        x_bounds = Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
        u_min = np.random.random((3, 3)) * tau_min + tau_min / 2
        u_max = np.random.random((3, 3)) * tau_max + tau_max / 2
        u_bounds = Bounds(u_min, u_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    elif interpolation_type == InterpolationType.LINEAR:
        x_min = np.random.random((6, 2)) * (-10) - 5
        x_max = np.random.random((6, 2)) * 10 + 5
        x_bounds = Bounds(x_min, x_max, interpolation=InterpolationType.LINEAR)
        u_min = np.random.random((3, 2)) * tau_min + tau_min / 2
        u_max = np.random.random((3, 2)) * tau_max + tau_max / 2
        u_bounds = Bounds(u_min, u_max, interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.EACH_FRAME:
        x_min = np.random.random((nq + nqdot, n_shooting + 1)) * (-10) - 5
        x_max = np.random.random((nq + nqdot, n_shooting + 1)) * 10 + 5
        x_bounds = Bounds(x_min, x_max, interpolation=InterpolationType.EACH_FRAME)
        u_min = np.random.random((ntau, n_shooting)) * tau_min + tau_min / 2
        u_max = np.random.random((ntau, n_shooting)) * tau_max + tau_max / 2
        u_bounds = Bounds(u_min, u_max, interpolation=InterpolationType.EACH_FRAME)
    elif interpolation_type == InterpolationType.SPLINE:
        spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        x_min = np.random.random((nq + nqdot, 5)) * (-10) - 5
        x_max = np.random.random((nq + nqdot, 5)) * 10 + 5
        u_min = np.random.random((ntau, 5)) * tau_min + tau_min / 2
        u_max = np.random.random((ntau, 5)) * tau_max + tau_max / 2
        x_bounds = Bounds(x_min, x_max, interpolation=InterpolationType.SPLINE, t=spline_time)
        u_bounds = Bounds(u_min, u_max, interpolation=InterpolationType.SPLINE, t=spline_time)
    elif interpolation_type == InterpolationType.CUSTOM:
        # The custom functions refer to the ones at the beginning of the file.
        # For this particular instance, they emulate a Linear interpolation
        extra_params_x = {"n_elements": nq + nqdot, "n_shooting": n_shooting}
        extra_params_u = {"n_elements": ntau, "n_shooting": n_shooting}
        x_bounds = Bounds(
            custom_x_bounds_min, custom_x_bounds_max, interpolation=InterpolationType.CUSTOM, **extra_params_x
        )
        u_bounds = Bounds(
            custom_u_bounds_min, custom_u_bounds_max, interpolation=InterpolationType.CUSTOM, **extra_params_u
        )
    else:
        raise NotImplementedError("Not implemented yet")

    # Initial guess
    x_init = InitialGuess([0] * (nq + nqdot))
    u_init = InitialGuess([tau_init] * ntau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        assume_phase_dynamics=True,
    )


def main():
    """
    Show all the InterpolationType implemented in bioptim
    """

    print(f"Show the bounds")
    for interpolation_type in InterpolationType:
        print(f"Solving problem using {interpolation_type} bounds")
        ocp = prepare_ocp("models/cube.bioMod", n_shooting=30, final_time=2, interpolation_type=interpolation_type)
        sol = ocp.solve()
        print("\n")

        # Print the last solution
        sol.graphs(show_bounds=True)

    for interpolation_type in InterpolationType:
        print(f"Solving problem using {interpolation_type} bounds")
        ocp = prepare_ocp("models/cube.bioMod", n_shooting=30, final_time=2, interpolation_type=interpolation_type)
        sol = ocp.solve()
        print("\n")

        # Print the last solution
        sol.graphs(show_bounds=False)


if __name__ == "__main__":
    main()
