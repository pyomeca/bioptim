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

    all_bounds = []
    for k in range(nlp.n_states_nodes):
        states.node_index = k
        for p in range(original_repeat if k != nlp.ns else 1):
            collapsed = _compute_bound_for_node(
                k, p, DEFAULT_MIN_BOUND, original_repeat, n_steps_callback, states, states_bounds.min(), states_scaling
            )
            all_bounds += [np.reshape(collapsed.T, (-1, 1))]
    v_bounds_min = np.concatenate(all_bounds, axis=0)

    all_bounds = []
    for k in range(nlp.n_states_nodes):
        states.node_index = k
        for p in range(original_repeat if k != nlp.ns else 1):
            collapsed = _compute_bound_for_node(
                k, p, DEFAULT_MAX_BOUND, original_repeat, n_steps_callback, states, states_bounds.max(), states_scaling
            )
            all_bounds += [np.reshape(collapsed.T, (-1, 1))]
    v_bounds_max = np.concatenate(all_bounds, axis=0)

    return v_bounds_min, v_bounds_max


def _compute_bound_for_node(
    k: int,
    p: int,
    default_bound: np.ndarray,  # "min" or "max"
    repeat: int,
    n_steps_callback: Callable,
    states: OptimizationVariableContainer,
    states_bounds: dict,  # min or max only not both
    states_scaling: "VariableScalingList",
) -> np.ndarray:
    collapsed_values = np.ndarray((states.shape, 1))

    real_keys = [key for key in states_bounds.keys() if key is not "None"]
    for key in real_keys:

        if states_bounds[key].type == InterpolationType.ALL_POINTS:
            point = k * n_steps_callback(0) + p
        else:
            point = _get_interpolation_point(k, p)

        value = states_bounds[key].evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
        value /= states_scaling[key].scaling

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
    is_first_node_in_interval = interval_node == 0

    if is_first_node and is_first_node_in_interval:
        return 0
    elif is_first_node and not is_first_node_in_interval:
        return 1  # NOTE: This is the hack
    else:
        return node
