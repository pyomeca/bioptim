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
    ObjectiveWeight,
)


def custom_weight(node: int, n_nodes: int) -> float:
    """
    The custom function for the objective wright (this particular one mimics linear interpolation)

    Parameters
    ----------
    node: int
        The index of the current point to return the value
    n_nodes: int
        The number of index available

    Returns
    -------
    The vector value of the bounds at current_shooting_point
    """

    my_values = [0, 1]
    # Linear interpolation created with custom function
    if n_nodes == 1:
        return my_values[0]
    else:
        return my_values[0] + (my_values[1] - my_values[0]) * node / (n_nodes - 1)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    interpolation_type: InterpolationType = InterpolationType.CONSTANT,
    node: Node = Node.ALL_SHOOTING,
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

    if node == Node.START:
        n_nodes = 1
    elif node == Node.INTERMEDIATES:
        n_nodes = n_shooting - 2
    elif node == Node.ALL_SHOOTING:
        n_nodes = n_shooting
    else:
        raise RuntimeError("This example is not designed to work with this node type.")

    # ObjectiveWeight
    if interpolation_type == InterpolationType.CONSTANT:
        weight = [1]
        weight = ObjectiveWeight(weight, interpolation=InterpolationType.CONSTANT)
    elif interpolation_type == InterpolationType.LINEAR:
        weight = [0, 1]
        weight = ObjectiveWeight(weight, interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.EACH_FRAME:
        weight = np.linspace(0, 1, n_nodes)
        weight = ObjectiveWeight(weight, interpolation=InterpolationType.EACH_FRAME)
    elif interpolation_type == InterpolationType.SPLINE:
        spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        spline_points = np.random.random((5,)) * (-10) - 5
        weight = ObjectiveWeight(spline_points, interpolation=InterpolationType.SPLINE, t=spline_time)
    elif interpolation_type == InterpolationType.CUSTOM:
        # The custom functions refer to the one at the beginning of the file.
        # For this particular instance, they emulate a Linear interpolation
        extra_params = {"n_nodes": n_nodes}
        weight = ObjectiveWeight(custom_weight, interpolation=InterpolationType.CUSTOM, **extra_params)
    else:
        raise NotImplementedError("Not implemented yet")

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_CONTROL, key="tau", node=node, weight=weight)

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

    nodes_to_test = [Node.START, Node.INTERMEDIATES, Node.ALL_SHOOTING]

    for interpolation_type in InterpolationType:
        for node in nodes_to_test:
            if (
                interpolation_type == InterpolationType.ALL_POINTS
                or interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            ):
                continue

            print(f"Solving problem using {interpolation_type} weight applied at {node} nodes.")
            ocp = prepare_ocp(
                "models/cube.bioMod", n_shooting=30, final_time=2, interpolation_type=interpolation_type, node=node
            )
            sol = ocp.solve()
            print("\n")

            # Print the last solution
            sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
