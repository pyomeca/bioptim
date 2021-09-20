from .path_conditions import Bounds, InitialGuess
from ..dynamics.fatigue.fatigue_dynamics import FatigueList
from ..misc.enums import VariableType


class FatigueBounds(Bounds):
    """
    Specialized Bounds that reads a model to automatically extract fatigue state bounds
    """

    def __init__(self, fatigue: FatigueList, variable_type=VariableType.STATES):
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
        if "tau" in fatigue and len(fatigue["tau"]) > 0:
            sides = fatigue["tau"].suffix
            if len(sides) != 2:
                raise NotImplementedError("The required suffix for tau fatigue is not implemented yet")

            suffix = (
                getattr(fatigue["tau"][0].model, sides[0]).suffix() if variable_type == VariableType.STATES else [0]
            )
            for i, side in enumerate(sides):
                for s in range(len(suffix)):
                    for f in fatigue["tau"]:
                        f_side = getattr(f.model, side)
                        bound = f_side.default_bounds()[s]
                        if variable_type == VariableType.STATES:
                            x_min += [bound[0]]
                            x_max += [bound[1]]
                        else:
                            x_min += [f_side.scale if i == 0 else 0]
                            x_max += [f_side.scale if i == 1 else 0]

        if "muscles" in fatigue and len(fatigue["muscles"]) > 0:
            suffix = fatigue["muscles"].suffix if variable_type == VariableType.STATES else [0]
            for s in range(len(suffix)):
                for f in fatigue["muscles"]:
                    bound = f.model.default_bounds()[s]
                    if variable_type == VariableType.STATES:
                        x_min += [bound[0]]
                        x_max += [bound[1]]
                    else:
                        x_min += [0]
                        x_max += [1]

        super(FatigueBounds, self).__init__(min_bound=x_min, max_bound=x_max)


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
        if "tau" in fatigue and len(fatigue["tau"]) > 0:
            sides = fatigue["tau"][0].model.suffix()
            suffix = (
                getattr(fatigue["tau"][0].model, sides[0]).suffix() if variable_type == VariableType.STATES else [0]
            )
            for side in sides:
                for s in range(len(suffix)):
                    for f in fatigue["tau"]:
                        x_init += [getattr(f.model, side).default_initial_guess()[s]]

        if "muscles" in fatigue and len(fatigue["muscles"]) > 0:
            suffix = fatigue["muscles"].suffix if variable_type == VariableType.STATES else [0]
            for s in range(len(suffix)):
                for f in fatigue["muscles"]:
                    x_init += [f.model.default_initial_guess()[s]]

        super(FatigueInitialGuess, self).__init__(initial_guess=x_init)
