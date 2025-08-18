import numpy as np

from ..misc.parameters_types import (
    DoubleNpArrayTuple,
)

from ..misc.enums import InterpolationType
from ..limits.path_conditions import BoundsList
from ..optimization.optimization_variable import OptimizationVariableContainer
from .vector_utils import _compute_values_for_all_nodes, dimension_check, DEFAULT_MAX_BOUND, DEFAULT_MIN_BOUND


def _dispatch_state_bounds(
    nlp: "NonLinearProgram",
    states: OptimizationVariableContainer,
    states_bounds: BoundsList,
    states_scaling: "VariableScalingList",
    original_repeat: int,
) -> DoubleNpArrayTuple:
    nlp.set_node_index(0)

    dimension_check(states, states_bounds, nlp.ns, repeat=original_repeat)

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
    dimension_check(controls, control_bounds, ns - 1, repeat=1)

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
