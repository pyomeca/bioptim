import numpy as np

from ..misc.parameters_types import (
    DoubleNpArrayTuple,
)

from ..misc.enums import InterpolationType
from ..limits.path_conditions import BoundsList
from ..optimization.optimization_variable import OptimizationVariableContainer
from .vector_utils import _compute_values_for_all_nodes, DEFAULT_MAX_BOUND, DEFAULT_MIN_BOUND


def _dispatch_state_bounds(
    nlp: "NonLinearProgram",
    states: OptimizationVariableContainer,
    states_bounds: BoundsList,
    states_scaling: "VariableScalingList",
    original_repeat: int,
) -> DoubleNpArrayTuple:
    """
    Dispatch the bounds for the states of the optimization problem to the bound vector.
    Parameters
    ----------
    nlp
    states
    states_bounds
    states_scaling
    original_repeat : int
        The number of steps in each interval for direct collocation only.

    """
    states.node_index = 0

    for key in states_bounds.real_keys():
        repeat_for_key = original_repeat if states_bounds[key].type == InterpolationType.ALL_POINTS else 1
        n_shooting = nlp.ns * repeat_for_key
        states_bounds[key].check_and_adjust_dimensions(states[key].cx.shape[0], n_shooting)

    v_bounds_min = _compute_values_for_all_nodes(
        nlp,
        DEFAULT_MIN_BOUND,
        states,
        states_bounds.min(),
        states_scaling,
        nlp.n_states_nodes,
        original_repeat,
    )

    v_bounds_max = _compute_values_for_all_nodes(
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
) -> DoubleNpArrayTuple:
    nlp.set_node_index(0)

    ns = nlp.ns + 1 if nlp.control_type.has_a_final_node else nlp.ns

    # Dimension checks
    for key in control_bounds.real_keys():
        control_bounds[key].check_and_adjust_dimensions(controls[key].cx.shape[0], ns - 1)

    v_bounds_min = _compute_values_for_all_nodes(
        nlp,
        DEFAULT_MIN_BOUND,
        controls,
        control_bounds.min(),
        control_scaling,
        ns,
        repeat=1,
    )

    v_bounds_max = _compute_values_for_all_nodes(
        nlp,
        DEFAULT_MAX_BOUND,
        controls,
        control_bounds.max(),
        control_scaling,
        ns,
        repeat=1,
    )
    return v_bounds_min, v_bounds_max
