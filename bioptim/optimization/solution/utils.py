import numpy as np
from ...misc.parameters_types import (
    Bool,
    NpArray,
    NpArrayList,
    NpArrayDict,
)

from ...limits.path_conditions import InitialGuessList
from ...misc.enums import InterpolationType
from .solution_data import SolutionMerge


def _resample_initial_guess(values: np.ndarray, n_columns: int) -> np.ndarray:
    """Linearly resample node values while preserving both endpoints."""

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    finite_columns = np.all(np.isfinite(values), axis=0)
    values = values[:, finite_columns]
    if values.shape[1] == 0:
        raise ValueError("A solution variable contains no finite node that can be transferred")
    if n_columns == 1:
        return values[:, :1].copy()
    if values.shape[1] == 1:
        return np.repeat(values, n_columns, axis=1)

    source_grid = np.linspace(0.0, 1.0, values.shape[1])
    target_grid = np.linspace(0.0, 1.0, n_columns)
    return np.vstack([np.interp(target_grid, source_grid, row) for row in values])


def _target_column_count(initial_guess) -> int:
    interpolation = initial_guess.type
    if interpolation == InterpolationType.CONSTANT:
        return 1
    if interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        return 3
    if interpolation == InterpolationType.LINEAR:
        return 2
    if interpolation in (InterpolationType.EACH_FRAME, InterpolationType.ALL_POINTS):
        return initial_guess.init.shape[1]
    raise NotImplementedError(f"Adapting a solution to {interpolation} is not implemented")


def _as_phase_list(data, n_phases: int) -> list[dict]:
    if n_phases == 1 and isinstance(data, dict):
        return [data]
    if not isinstance(data, list) or len(data) != n_phases:
        raise ValueError(f"The solution has {len(data) if isinstance(data, list) else 1} phases, expected {n_phases}")
    return data


def _adapt_variable_group(source, target: InitialGuessList, group_name: str) -> InitialGuessList:
    adapted = InitialGuessList()
    n_phases = len(target.options)
    source_phases = _as_phase_list(source, n_phases)
    for phase, target_phase in enumerate(target.options):
        for key, target_guess in target_phase.items():
            if key not in source_phases[phase]:
                raise KeyError(f"{group_name} '{key}' is absent from the solution phase {phase}")
            values = _resample_initial_guess(source_phases[phase][key], _target_column_count(target_guess))
            adapted.add(key, values, interpolation=target_guess.type, phase=phase)
    return adapted


def adapt_solution_to_initial_guesses(
    solution,
    state_initial_guesses: InitialGuessList,
    control_initial_guesses: InitialGuessList,
    parameter_initial_guesses: InitialGuessList | None = None,
    algebraic_state_initial_guesses: InitialGuessList | None = None,
) -> tuple[InitialGuessList, InitialGuessList, InitialGuessList, InitialGuessList]:
    """Adapt a solution's primal variables to the grids described by new initial-guess lists.

    Resampling is performed on a normalized phase grid, so it supports different numbers of shooting nodes,
    collocation points and control nodes. Solver multipliers are deliberately not transferred by this function.
    """

    states = _adapt_variable_group(
        solution.decision_states(to_merge=SolutionMerge.NODES), state_initial_guesses, "State"
    )
    controls = _adapt_variable_group(
        solution.decision_controls(to_merge=SolutionMerge.NODES), control_initial_guesses, "Control"
    )

    parameters = InitialGuessList()
    if parameter_initial_guesses is not None:
        source_parameters = solution.decision_parameters()
        for phase, target_phase in enumerate(parameter_initial_guesses.options):
            for key, target_guess in target_phase.items():
                if key not in source_parameters:
                    raise KeyError(f"Parameter '{key}' is absent from the solution")
                values = np.asarray(source_parameters[key], dtype=float).reshape((-1, 1))
                parameters.add(key, values, interpolation=target_guess.type, phase=phase)

    algebraic_states = InitialGuessList()
    if algebraic_state_initial_guesses is not None:
        algebraic_states = _adapt_variable_group(
            solution.decision_algebraic_states(to_merge=SolutionMerge.NODES),
            algebraic_state_initial_guesses,
            "Algebraic state",
        )

    return states, controls, parameters, algebraic_states


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
