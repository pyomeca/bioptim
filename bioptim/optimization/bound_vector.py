import numpy as np

from ..misc.parameters_types import (
    DoubleNpArrayTuple,
    Callable,
)

from ..misc.enums import InterpolationType
from ..limits.path_conditions import BoundsList
from ..optimization.optimization_variable import OptimizationVariableContainer

DEFAULT_MIN_BOUND = -np.inf
DEFAULT_MAX_BOUND = np.inf


def _dispatch_state_bounds(
    nlp: "NonLinearProgram",
    states: OptimizationVariableContainer,
    states_bounds: BoundsList,
    states_scaling: "VariableScalingList",
    n_steps_callback: Callable,
) -> DoubleNpArrayTuple:
    states.node_index = 0
    original_repeat = n_steps_callback(0)

    # Dimension checks
    real_keys = [key for key in states_bounds.keys() if key is not "None"]
    for key in real_keys:
        repeat_for_key = original_repeat if states_bounds[key].type == InterpolationType.ALL_POINTS else 1
        n_shooting = nlp.ns * repeat_for_key
        states_bounds[key].check_and_adjust_dimensions(states[key].cx.shape[0], n_shooting)

    v_bounds_min = _compute_bounds_for_all_nodes(
        nlp,
        DEFAULT_MIN_BOUND,
        states,
        states_bounds.min(),
        states_scaling,
        nlp.n_states_nodes,
        original_repeat,
    )

    v_bounds_max = _compute_bounds_for_all_nodes(
        nlp,
        DEFAULT_MAX_BOUND,
        states,
        states_bounds.max(),
        states_scaling,
        nlp.n_states_nodes,
        original_repeat,
    )
    return v_bounds_min, v_bounds_max


def _dispatch_control_bounds(
    nlp: "NonLinearProgram",
    controls: OptimizationVariableContainer,
    control_bounds: BoundsList,
    control_scaling: "VariableScalingList",
    n_steps_callback: Callable,
) -> DoubleNpArrayTuple:
    original_repeat = 1
    nlp.set_node_index(0)

    ns = nlp.ns
    if nlp.control_type.has_a_final_node:
        ns += 1

    # Dimension checks
    real_keys = [key for key in control_bounds.keys() if key is not "None"]
    for key in real_keys:
        control_bounds[key].check_and_adjust_dimensions(controls[key].cx.shape[0], ns - 1)

    v_bounds_min = _compute_bounds_for_all_nodes(
        nlp,
        DEFAULT_MIN_BOUND,
        controls,
        control_bounds.min(),
        control_scaling,
        ns,
        original_repeat,
    )

    v_bounds_max = _compute_bounds_for_all_nodes(
        nlp,
        DEFAULT_MAX_BOUND,
        controls,
        control_bounds.max(),
        control_scaling,
        ns,
        original_repeat,
    )
    return v_bounds_min, v_bounds_max


def _compute_bounds_for_all_nodes(
    nlp: "NonLinearProgram",
    default_bound: np.ndarray,
    optimization_variable: OptimizationVariableContainer,
    bounds: dict,  # "min" or "max" only, not both
    scaling: "VariableScalingList",
    n_nodes: int,
    repeat: int,
) -> np.ndarray:
    """
    Compute bounds for all nodes in the discretized problem.

    This function iterates through all nodes and their intervals to compute
    either minimum or maximum bounds for the state variables.

    Parameters
    ----------
    nlp: NonLinearProgram
        The non-linear program containing the optimization problem
    default_bound : np.ndarray
        The default bound value (either DEFAULT_MIN_BOUND or DEFAULT_MAX_BOUND)
    optimization_variable : OptimizationVariableContainer
        The states variables container
    bounds : dict
        The bounds list object (either min or max bounds)
    scaling : VariableScalingList
        The scaling factors for states
    n_nodes : int
        The number of state nodes in the optimization problem
    repeat : int
        Number of steps in each interval (for direct collocation, this is the number of nodes per interval)
        Only relevant for states or algebraic variables not for controls.

    Returns
    -------
    np.ndarray
        The concatenated bounds for all nodes
    """
    all_bounds = []
    for node in range(n_nodes):
        nlp.set_node_index(node)
        is_final_node = node == nlp.ns
        for interval_node in range(1 if is_final_node else repeat):
            collapsed = _compute_bound_for_node(
                node,
                interval_node,
                default_bound,
                repeat,
                optimization_variable,
                bounds,
                scaling,
            )
            all_bounds += [np.reshape(collapsed.T, (-1, 1))]
    return np.concatenate(all_bounds, axis=0)


def _compute_bound_for_node(
    node: int,
    interval_node: int,
    default_bound: np.ndarray,  # "min" or "max"
    repeat: int,
    states: OptimizationVariableContainer,
    states_bounds: dict,  # min or max only not both
    states_scaling: "VariableScalingList",
) -> np.ndarray:
    collapsed_values = np.ndarray((states.shape, 1))

    real_keys = [key for key in states_bounds.keys() if key is not "None"]
    for key in real_keys:

        if states_bounds[key].type == InterpolationType.ALL_POINTS:
            point = node * repeat + interval_node
        else:
            point = _get_interpolation_point(node, interval_node)

        value = (
            states_bounds[key].evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
            / states_scaling[key].scaling
        )

        collapsed_values[states[key].index, :] = value

    key_not_in_bounds = set(states.keys()) - set(states_bounds.keys())
    for key in key_not_in_bounds:
        collapsed_values[states[key].index, :] = default_bound

    return collapsed_values


def _get_interpolation_point(node: int, interval_node: int) -> int:
    """
    This function determines the interpolation point to use for InterpolationType OTHER THAN ALL_POINTS.

    NOTE: This logic allows CONSTANT_WITH_FIRST_AND_LAST to work with OdeSolver.COLLOCATION,
    This would also work for InterpolationType.CONSTANT, and ALL_POINTS, but not for the others.

    In the case of direct collocation, we ENFORCE the nodes within the first interval to take the value of
     the node of index 1 instead of 0.

    n = node, i = interval, p = returned point

    Standard Case:                  Collocation Case:
    (direct multiple shooting)      (Direct Collocation)
    n0 n1 n2 n3 ... nN              n_{0,0} n_{0,1} n_{0,2} ... n_{0,N}, n_{1,0} ...
    |  |  |  |      |               /       |       |             |       |
    p0 p1 p2 p3 ... pN             0        1       1             1       1

    Parameters
    ----------
    node: int
        The current node index
    interval_node: int
        The current interval node index (0 for the first node, 1 for the second node, etc.),
            in the case of direct collocation more decision variable exist within an interval.
    Returns
    -------
    int
        The new point/node to use for the given node and interval_node

    """
    is_first_node = node == 0

    # always true for direct multiple shooting, but not for direct collocation
    is_first_node_in_interval = interval_node == 0

    if is_first_node and is_first_node_in_interval:
        return 0
    elif is_first_node and not is_first_node_in_interval:
        return 1  # NOTE: This is the hack
    else:
        return node
