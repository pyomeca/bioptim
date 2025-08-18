from ..misc.parameters_types import (
    NpArray,
)

from ..misc.enums import InterpolationType
from ..optimization.non_linear_program import NonLinearProgram
from ..limits.path_conditions import InitialGuessList
from ..optimization.optimization_variable import OptimizationVariableContainer
from .vector_utils import _compute_values_for_all_nodes

DEFAULT_INITIAL_GUESS = 0


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
