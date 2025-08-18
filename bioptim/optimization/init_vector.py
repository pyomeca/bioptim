from ..misc.parameters_types import (
    NpArray,
)

from ..misc.enums import InterpolationType
from ..optimization.non_linear_program import NonLinearProgram
from ..limits.path_conditions import InitialGuessList
from ..optimization.optimization_variable import OptimizationVariableContainer
from .vector_utils import _compute_values_for_all_nodes, dimension_check, DEFAULT_INITIAL_GUESS


def _dispatch_state_initial_guess(
    nlp: "NonLinearProgram",
    states: OptimizationVariableContainer,
    states_init: InitialGuessList,
    states_scaling: "VariableScalingList",
    original_repeat: int,
) -> NpArray:
    states.node_index = 0

    dimension_check(states, states_init, nlp.ns, repeat=original_repeat)

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
    dimension_check(controls, controls_init, ns - 1, repeat=1)

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
