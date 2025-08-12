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
    repeat = n_steps_callback(0)

    # Dimension check
    real_keys = [key for key in states_bounds.keys() if key is not "None"]
    for key in real_keys:
        if states_bounds[key].type == InterpolationType.ALL_POINTS:
            states_bounds[key].check_and_adjust_dimensions(states[key].cx.shape[0], nlp.ns * repeat)
        else:
            states_bounds[key].check_and_adjust_dimensions(states[key].cx.shape[0], nlp.ns)

    all_bounds = []
    for k in range(nlp.n_states_nodes):
        states.node_index = k
        for p in range(repeat if k != nlp.ns else 1):
            collapsed = _compute_bound_for_node(
                k, p, DEFAULT_MIN_BOUND, repeat, n_steps_callback, states, states_bounds.min(), states_scaling
            )
            all_bounds += [np.reshape(collapsed.T, (-1, 1))]
    v_bounds_min = np.concatenate(all_bounds, axis=0)

    all_bounds = []
    for k in range(nlp.n_states_nodes):
        states.node_index = k
        for p in range(repeat if k != nlp.ns else 1):
            collapsed = _compute_bound_for_node(
                k, p, DEFAULT_MAX_BOUND, repeat, n_steps_callback, states, states_bounds.max(), states_scaling
            )
            all_bounds += [np.reshape(collapsed.T, (-1, 1))]
    v_bounds_max = np.concatenate(all_bounds, axis=0)

    return v_bounds_min, v_bounds_max


def _compute_bound_for_node(
    k: int,
    p: int,
    default_bound: str,  # "min" or "max"
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
            point = k if k != 0 else 0 if p == 0 else 1

        value = (
            states_bounds[key].evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
            / states_scaling[key].scaling
        )
        collapsed_values[states[key].index, :] = value

    key_not_in_bounds = set(states.keys()) - set(states_bounds.keys())
    for key in key_not_in_bounds:
        collapsed_values[states[key].index, :] = default_bound

    return collapsed_values
