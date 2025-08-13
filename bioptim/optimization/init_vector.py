import numpy as np

from ..misc.parameters_types import (
    DoubleNpArrayTuple,
    Callable,
    NpArray,
)

from ..misc.enums import InterpolationType
from ..optimization.non_linear_program import NonLinearProgram
from ..limits.path_conditions import InitialGuessList, InitialGuess
from ..optimization.optimization_variable import OptimizationVariableContainer

DEFAULT_INITIAL_GUESS = 0


def _dispatch_state_initial_guess(
    nlp: "NonLinearProgram",
    states: OptimizationVariableContainer,
    states_init: InitialGuessList,
    states_scaling: "VariableScalingList",
    original_repeat: int,
) -> NpArray:
    states.node_index = 0

    # Dimension checks
    real_keys = [key for key in states_init.keys() if key is not "None"]
    for key in real_keys:
        repeat_for_key = original_repeat if states_init[key].type == InterpolationType.ALL_POINTS else 1
        n_shooting = nlp.ns * repeat_for_key
        states_init[key].check_and_adjust_dimensions(states[key].cx.shape[0], n_shooting)

    v_init = np.ndarray((0, 1))
    for k in range(nlp.n_states_nodes):
        states.node_index = k

        for p in range(original_repeat if k != nlp.ns else 1):
            collapsed_values_init = np.ndarray((states.shape, 1))
            for key in states:
                if key in states_init.keys():
                    if states_init[key].type == InterpolationType.ALL_POINTS:
                        point = k * original_repeat + p
                    else:
                        # This allows CONSTANT_WITH_FIRST_AND_LAST to work in collocations, but is flawed for the other ones
                        # point refers to the column to use in the bounds matrix
                        point = k if k != 0 else 0 if p == 0 else 1

                    value_init = (
                        states_init[key].init.evaluate_at(shooting_point=point, repeat=original_repeat)[:, np.newaxis]
                        / states_scaling[key].scaling
                    )

                else:
                    value_init = DEFAULT_INITIAL_GUESS

                # Organize the controls according to the correct indices
                collapsed_values_init[states[key].index, :] = value_init

            v_init = np.concatenate((v_init, np.reshape(collapsed_values_init.T, (-1, 1))))

    return v_init
