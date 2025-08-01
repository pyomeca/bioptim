"""
This example is a trivial box sent upward. It is designed to investigate the different types of objective weights one
can define in bioptim. Therefore, it shows how one can define the weight of the minimize controls objective.

All the types of interpolation are shown:
InterpolationType.CONSTANT: All the values are the same at each node
InterpolationType.LINEAR: The values are linearly interpolated between the first and last nodes.
InterpolationType.EACH_FRAME: Each node values are specified
InterpolationType.SPLINE: The values are interpolated from the first to last node using a cubic spline
InterpolationType.CUSTOM: Provide a user-defined interpolation function

Please note that InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT is available, but does not make much sense in
this context.
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
    Weight,
)


def custom_weight(current_shooting_point: int, n_elements: int, n_shooting: int) -> np.ndarray:
    """
    The custom function for the objective wright (this particular one mimics linear interpolation)

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


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    interpolation_type: InterpolationType = InterpolationType.CONSTANT,
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

    # Weight
    if interpolation_type == InterpolationType.CONSTANT:
        weight = Weight([1], interpolation=InterpolationType.CONSTANT)
    elif interpolation_type == InterpolationType.LINEAR:
        weight = Weight([0, 1], interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.EACH_FRAME:
        weight = Weight(np.linspace(0, 1, n_shooting + 1), interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.SPLINE:
        spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        spline_points = np.random.random((nq + nqdot, 5)) * (-10) - 5
        weight = Weight(spline_points, interpolation=InterpolationType.SPLINE, t=spline_time)
    elif interpolation_type == InterpolationType.CUSTOM:
        # The custom functions refer to the one at the beginning of the file.
        # For this particular instance, they emulate a Linear interpolation
        extra_params = {"n_elements": nq + nqdot, "n_shooting": n_shooting}
        weight = Weight(custom_weight, interpolation=InterpolationType.CUSTOM, **extra_params)
    else:
        raise NotImplementedError("Not implemented yet")

    # Add objective functions
    objective_functions = Objective(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=weight
    )

    # DynamicsOptions
    dynamics = DynamicsOptions(expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # Path condition
    x_bounds = BoundsList()
    x_bounds.add("q", min_bound=[-100] * nq, max_bound=[100] * nq, interpolation=InterpolationType.CONSTANT)
    x_bounds.add("qdot", min_bound=[-100] * nqdot, max_bound=[100] * nqdot, interpolation=InterpolationType.CONSTANT)
    u_bounds = BoundsList()
    u_bounds.add(
        "tau", min_bound=[tau_min] * ntau, max_bound=[tau_max] * ntau, interpolation=InterpolationType.CONSTANT
    )

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
        # TODO REMOVE !!!!!!!!!!!
        if interpolation_type != InterpolationType.LINEAR:
            continue

        print(f"Solving problem using {interpolation_type} bounds")
        ocp = prepare_ocp("models/cube.bioMod", n_shooting=30, final_time=2, interpolation_type=interpolation_type)
        sol = ocp.solve()
        print("\n")

        # Print the last solution
        sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
