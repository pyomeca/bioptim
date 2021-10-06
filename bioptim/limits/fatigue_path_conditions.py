from .path_conditions import Bounds, InitialGuess
from ..dynamics.fatigue.fatigue_dynamics import FatigueList
from ..misc.enums import VariableType


class FatigueBounds(Bounds):
    """
    Specialized Bounds that reads a model to automatically extract fatigue state bounds
    """

    def __init__(self, fatigue: FatigueList, variable_type=VariableType.STATES, fix_first_frame=False):
        """
        Parameters
        ----------
        fatigue: FatigueList
            The fatigue model sent to the dynamics
        variable_type: VariableType
            The type of variable to prepare
        """

        x_min = []
        x_max = []
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

            for index in range(len(fatigue[key][0].models.models)):
                for i, _ in enumerate(suffix):
                    for multi in fatigue[key]:
                        bound = multi.models.default_bounds(index, variable_type)
                        x_min += [bound[0][i]]
                        x_max += [bound[1][i]]
                if variable_type == VariableType.CONTROLS and not fatigue[key][0].models.split_controls:
                    break

        super(FatigueBounds, self).__init__(min_bound=x_min, max_bound=x_max)

        if fix_first_frame:
            self[:, 0] = FatigueInitialGuess(fatigue, variable_type).init[:, 0]


class FatigueInitialGuess(InitialGuess):
    """
    Specialized InitialGuess that defines initial guess for each fatigue state
    """

    def __init__(self, fatigue: FatigueList, variable_type: VariableType = VariableType.STATES):
        """
        Parameters
        ----------
        fatigue: FatigueList
            The fatigue model sent to the dynamics
        variable_type: VariableType
            The type of variable to prepare
        """

        x_init = []
        for key in fatigue.keys():
            # for to make sure the model is homogenous
            suffix = fatigue[key][0].models.models[fatigue[key][0].models.suffix()[0]].suffix(variable_type)
            for dof in fatigue[key]:
                for m in dof.models.models:
                    if dof.models.models[m].suffix(variable_type) != suffix:
                        raise NotImplementedError("Fatigue models cannot be mixed")

            for index in range(len(fatigue[key][0].models.models)):
                for i, _ in enumerate(suffix):
                    for multi in fatigue[key]:
                        x_init += [multi.models.default_initial_guess(index, variable_type)[i]]
                if variable_type == VariableType.CONTROLS and not fatigue[key][0].models.split_controls:
                    break

        super(FatigueInitialGuess, self).__init__(initial_guess=x_init)
