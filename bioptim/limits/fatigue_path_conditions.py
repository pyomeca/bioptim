import numpy as np

from .path_conditions import InitialGuessList, BoundsList
from ..dynamics.fatigue.fatigue_dynamics import FatigueList
from ..misc.enums import VariableType


def FatigueBounds(fatigue: FatigueList, variable_type=VariableType.STATES, fix_first_frame=False):
    """
    Parameters
    ----------
    fatigue: FatigueList
        The fatigue model sent to the dynamics
    variable_type: VariableType
        The type of variable to prepare
    """

    out = BoundsList()
    for key in fatigue.keys():
        # for to make sure the model is homogenous
        if fatigue[key][0].models.suffix() is not None:
            suffix = fatigue[key][0].models.models[fatigue[key][0].models.suffix()[0]].suffix(variable_type)
        else:
            suffix = []

        for multi in fatigue[key]:
            for m in multi.models.models:
                if multi.models.models[m].suffix(variable_type) != suffix:
                    raise NotImplementedError("Fatigue models cannot be mixed")

        initial = None
        if fix_first_frame:
            initial = FatigueInitialGuess(fatigue, variable_type)

        for index in range(len(fatigue[key][0].models.models)):
            for i, _ in enumerate(suffix):
                min_bounds = []
                max_bounds = []
                for multi in fatigue[key]:
                    min_bound, max_bound = multi.models.default_bounds(index, variable_type)
                    min_bounds = np.concatenate((min_bounds, [min_bound[i]]))
                    max_bounds = np.concatenate((max_bounds, [max_bound[i]]))

                key_name = f"{key}_{suffix[i]}" if suffix[i] else key
                out[key_name] = min_bounds, max_bounds
                if fix_first_frame:
                    out[key_name][:, 0] = initial[key_name].init[:, 0]

            if variable_type == VariableType.CONTROLS and not fatigue[key][0].models.split_controls:
                break

    return out


def FatigueInitialGuess(fatigue: FatigueList, variable_type: VariableType = VariableType.STATES):
    """
    Parameters
    ----------
    fatigue: FatigueList
        The fatigue model sent to the dynamics
    variable_type: VariableType
        The type of variable to prepare
    """

    out = InitialGuessList()
    for key in fatigue.keys():
        # for to make sure the model is homogenous
        suffix = fatigue[key][0].models.models[fatigue[key][0].models.suffix()[0]].suffix(variable_type)
        for dof in fatigue[key]:
            for m in dof.models.models:
                if dof.models.models[m].suffix(variable_type) != suffix:
                    raise NotImplementedError("Fatigue models cannot be mixed")

        for index in range(len(fatigue[key][0].models.models)):
            for i, _ in enumerate(suffix):
                initial_guesses = []
                for multi in fatigue[key]:
                    initial = multi.models.default_initial_guess(index, variable_type)
                    initial_guesses = np.concatenate((initial_guesses, [initial[i]]))

                key_name = f"{key}_{suffix[i]}" if suffix[i] else key
                out[key_name] = initial_guesses
            if variable_type == VariableType.CONTROLS and not fatigue[key][0].models.split_controls:
                break
    return out
