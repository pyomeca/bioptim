import numpy as np
from ...misc.parameters_types import (
    Bool,
    NpArray,
    NpArrayList,
    NpArrayDict,
)


def concatenate_optimization_variables_dict(variable: list[NpArrayDict], continuous: Bool = True) -> list[NpArrayDict]:
    """
    This function concatenates the decision variables of the phases of the system
    into a single array, omitting the last element of each phase except for the last one.

    Parameters
    ----------
    variable : list or dict
        list of decision variables of the phases of the system
    continuous: bool
        If the arrival value of a node should be discarded [True] or kept [False].

    Returns
    -------
    z_concatenated : np.ndarray or dict
        array of the decision variables of the phases of the system concatenated
    """
    if isinstance(variable, list):
        if isinstance(variable[0], dict):
            variable_dict = dict()
            for key in variable[0].keys():
                variable_dict[key] = [v_i[key] for v_i in variable]
                final_tuple = [
                    y[:, :-1] if i < (len(variable_dict[key]) - 1) and continuous else y
                    for i, y in enumerate(variable_dict[key])
                ]
                variable_dict[key] = np.hstack(final_tuple)
            return [variable_dict]
    else:
        raise ValueError("the input must be a list")


def concatenate_optimization_variables(
    variable: NpArrayList | NpArray,
    continuous_phase: Bool = True,
    continuous_interval: Bool = True,
    merge_phases: Bool = True,
) -> NpArray | list[NpArrayDict]:
    """
    This function concatenates the decision variables of the phases of the system
    into a single array, omitting the last element of each phase except for the last one.

    Parameters
    ----------
    variable : list or dict
        list of decision variables of the phases of the system
    continuous_phase: bool
        If the arrival value of a node should be discarded [True] or kept [False]. The value of an integrated
    continuous_interval: bool
        If the arrival value of a node of each interval should be discarded [True] or kept [False].
        Only useful in direct multiple shooting
    merge_phases: bool
        If the decision variables of each phase should be merged into a single array [True] or kept separated [False].

    Returns
    -------
    z_concatenated : np.ndarray or dict
        array of the decision variables of the phases of the system concatenated
    """
    if len(variable[0].shape):
        if isinstance(variable[0][0], np.ndarray):
            z_final = [concatenate_optimization_variables(zi, continuous_interval) for zi in variable]

            return concatenate_optimization_variables(z_final, continuous_phase) if merge_phases else z_final

        else:
            final_tuple = []
            for i, y in enumerate(variable):
                if i < (len(variable) - 1) and continuous_phase:
                    final_tuple.append(y[:, :-1] if len(y.shape) == 2 else y[:-1])
                else:
                    final_tuple.append(y)

        return np.hstack(final_tuple)
