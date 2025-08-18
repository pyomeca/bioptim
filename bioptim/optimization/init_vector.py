from ..misc.parameters_types import (
    NpArray,
)

from ..misc.enums import InterpolationType
from ..optimization.non_linear_program import NonLinearProgram
from ..limits.path_conditions import InitialGuessList
from ..optimization.optimization_variable import OptimizationVariableContainer
from .vector_utils import _compute_values_for_all_nodes, DEFAULT_INITIAL_GUESS


def _dispatch_state_initial_guess(
    nlp: "NonLinearProgram",
    states: OptimizationVariableContainer,
    states_init: InitialGuessList,
    states_scaling: "VariableScalingList",
    original_repeat: int,
) -> NpArray:
    states.node_index = 0

    for key in states_init.real_keys():
        repeat_for_key = original_repeat if states_init[key].type == InterpolationType.ALL_POINTS else 1
        n_shooting = nlp.ns * repeat_for_key
        states_init[key].check_and_adjust_dimensions(states[key].cx.shape[0], n_shooting)

    v_init = _compute_values_for_all_nodes(
        nlp,
        DEFAULT_INITIAL_GUESS,
        states,
        states_init,
        states_scaling,
        nlp.n_states_nodes,
        original_repeat,
    )

    return v_init


def _dispatch_control_initial_guess(
    nlp: "NonLinearProgram",
    controls: OptimizationVariableContainer,
    controls_init: InitialGuessList,
    controls_scaling: "VariableScalingList",
) -> NpArray:
    nlp.set_node_index(0)
    ns = nlp.ns + 1 if nlp.control_type.has_a_final_node else nlp.ns

    for key in controls_init.real_keys():
        controls_init[key].check_and_adjust_dimensions(controls[key].cx.shape[0], ns - 1)

    v_init = _compute_values_for_all_nodes(
        nlp,
        DEFAULT_INITIAL_GUESS,
        controls,
        controls_init,
        controls_scaling,
        nlp.n_controls_nodes,
        repeat=1,
    )

    return v_init
