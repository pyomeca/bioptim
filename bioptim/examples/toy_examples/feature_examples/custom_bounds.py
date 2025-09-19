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
from bioptim import (
    TorqueBiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsOptions,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InterpolationType,
    PhaseDynamics,
)


def custom_x_bounds_min(current_shooting_point: int, n_elements: int, n_shooting: int, slicer: slice) -> np.ndarray:
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
    slicer: slice
        Which rows to use

    Returns
    -------
    The vector value of the bounds at current_shooting_point
    """

    my_values = np.array([[-10, -5]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[slicer, 0] + (my_values[slicer, -1] - my_values[slicer, 0]) * current_shooting_point / n_shooting


def custom_x_bounds_max(current_shooting_point: int, n_elements: int, n_shooting: int, slicer: slice) -> np.ndarray:
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
    slicer: slice
        Which rows to use

    Returns
    -------
    The vector value of the bounds at current_shooting_point
    """

    my_values = np.array([[10, 5]] * n_elements)
    # Linear interpolation created with custom function
    return my_values[slicer, 0] + (my_values[slicer, -1] - my_values[slicer, 0]) * current_shooting_point / n_shooting


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
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OCP fully prepared and ready to be solved
    """

    # BioModel path
    bio_model = TorqueBiorbdModel(biorbd_model_path)
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # DynamicsOptions
    dynamics = DynamicsOptions(expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # Path constraints
    if interpolation_type == InterpolationType.CONSTANT:
        # Here we need to use the .add nomenclature because interpolation is not the default
        x_bounds = BoundsList()
        x_bounds.add("q", min_bound=[-100] * nq, max_bound=[100] * nq, interpolation=InterpolationType.CONSTANT)
        x_bounds.add(
            "qdot", min_bound=[-100] * nqdot, max_bound=[100] * nqdot, interpolation=InterpolationType.CONSTANT
        )
        u_bounds = BoundsList()
        u_bounds.add(
            "tau", min_bound=[tau_min] * ntau, max_bound=[tau_max] * ntau, interpolation=InterpolationType.CONSTANT
        )
    elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        # Here we can use the direct variable assignment because no extra parameters are sent
        x_min = np.random.random((6, 3)) * (-10) - 5
        x_max = np.random.random((6, 3)) * 10 + 5
        x_bounds = BoundsList()
        x_bounds["q"] = x_min[:nq, :], x_max[:nq, :]
        x_bounds["qdot"] = x_min[nq:, :] * nqdot, x_max[nq:, :]

        u_min = np.random.random((3, 3)) * tau_min + tau_min / 2
        u_max = np.random.random((3, 3)) * tau_max + tau_max / 2
        u_bounds = BoundsList()
        u_bounds["tau"] = u_min, u_max

    elif interpolation_type == InterpolationType.LINEAR:
        # Here we need to use the .add nomenclature because interpolation is not the default
        x_min = np.random.random((6, 2)) * (-10) - 5
        x_max = np.random.random((6, 2)) * 10 + 5
        x_bounds = BoundsList()
        x_bounds.add("q", min_bound=x_min[:nq, :], max_bound=x_max[:nq, :], interpolation=InterpolationType.LINEAR)
        x_bounds.add(
            "qdot", min_bound=x_min[nq:, :] * nqdot, max_bound=x_max[nq:, :], interpolation=InterpolationType.LINEAR
        )

        u_min = np.random.random((3, 2)) * tau_min + tau_min / 2
        u_max = np.random.random((3, 2)) * tau_max + tau_max / 2
        u_bounds = BoundsList()
        u_bounds.add("tau", min_bound=u_min, max_bound=u_max, interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.EACH_FRAME:
        # Here we need to use the .add nomenclature because interpolation is not the default
        x_min = np.random.random((nq + nqdot, n_shooting + 1)) * (-10) - 5
        x_max = np.random.random((nq + nqdot, n_shooting + 1)) * 10 + 5
        x_bounds = BoundsList()
        x_bounds.add("q", min_bound=x_min[:nq, :], max_bound=x_max[:nq, :], interpolation=InterpolationType.EACH_FRAME)
        x_bounds.add(
            "qdot", min_bound=x_min[nq:, :] * nqdot, max_bound=x_max[nq:, :], interpolation=InterpolationType.EACH_FRAME
        )

        u_min = np.random.random((ntau, n_shooting)) * tau_min + tau_min / 2
        u_max = np.random.random((ntau, n_shooting)) * tau_max + tau_max / 2
        u_bounds = BoundsList()
        u_bounds.add("tau", min_bound=u_min, max_bound=u_max, interpolation=InterpolationType.EACH_FRAME)
    elif interpolation_type == InterpolationType.SPLINE:
        # Here we need to use the .add nomenclature because interpolation is not the default
        spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        x_min = np.random.random((nq + nqdot, 5)) * (-10) - 5
        x_max = np.random.random((nq + nqdot, 5)) * 10 + 5
        x_bounds = BoundsList()
        x_bounds.add(
            "q", min_bound=x_min[:nq, :], max_bound=x_max[:nq, :], interpolation=InterpolationType.SPLINE, t=spline_time
        )
        x_bounds.add(
            "qdot",
            min_bound=x_min[nq:, :] * nqdot,
            max_bound=x_max[nq:, :],
            interpolation=InterpolationType.SPLINE,
            t=spline_time,
        )

        u_min = np.random.random((ntau, 5)) * tau_min + tau_min / 2
        u_max = np.random.random((ntau, 5)) * tau_max + tau_max / 2
        u_bounds = BoundsList()
        u_bounds.add("tau", min_bound=u_min, max_bound=u_max, interpolation=InterpolationType.SPLINE, t=spline_time)
    elif interpolation_type == InterpolationType.CUSTOM:
        # The custom functions refer to the ones at the beginning of the file.
        # For this particular instance, they emulate a Linear interpolation
        extra_params_x = {"n_elements": nq + nqdot, "n_shooting": n_shooting}
        x_bounds = BoundsList()
        x_bounds.add(
            "q",
            min_bound=custom_x_bounds_min,
            max_bound=custom_x_bounds_max,
            interpolation=InterpolationType.CUSTOM,
            slicer=slice(0, nq),
            **extra_params_x,
        )
        x_bounds.add(
            "qdot",
            min_bound=custom_x_bounds_min,
            max_bound=custom_x_bounds_max,
            interpolation=InterpolationType.CUSTOM,
            slicer=slice(nq, nq + nqdot),
            **extra_params_x,
        )

        extra_params_u = {"n_elements": ntau, "n_shooting": n_shooting}
        u_bounds = BoundsList()
        u_bounds.add(
            "tau",
            min_bound=custom_u_bounds_min,
            max_bound=custom_u_bounds_max,
            interpolation=InterpolationType.CUSTOM,
            **extra_params_u,
        )
    else:
        raise NotImplementedError("Not implemented yet")

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
    )


def main():
    """
    Show all the InterpolationType implemented in bioptim
    """

    print(f"Show the bounds")
    for interpolation_type in InterpolationType:
        if interpolation_type == InterpolationType.ALL_POINTS:
            continue

        print(f"Solving problem using {interpolation_type} bounds")
        ocp = prepare_ocp("models/cube.bioMod", n_shooting=30, final_time=2, interpolation_type=interpolation_type)
        sol = ocp.solve()
        print("\n")

        # Print the last solution
        sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
