from typing import Any

from ..misc.fcn_enum import FcnEnum
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFunctions
from .objective_functions import ObjectiveFunction


class MultinodeObjective(MultinodePenalty):
    def __init__(self, *args, weight: float = 0, is_stochastic: bool = False, **kwargs):
        super(MultinodeObjective, self).__init__(MultinodeObjectiveFcn, *args, **kwargs)

        self.weight = weight if weight is not None else 0
        self.quadratic = kwargs["quadratic"] if "quadratic" in kwargs else True
        self.base = ObjectiveFunction.MayerFunction
        self.is_stochastic = is_stochastic

    def _get_pool_to_add_penalty(self, ocp, nlp):
        return nlp.J_internal if nlp else ocp.J_internal


class MultinodeObjectiveList(MultinodePenaltyList):
    """
    A list of Multinode Objective

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new MultinodeObjective to the list
    """

    def add(self, multinode_objective: Any, **extra_arguments: Any):
        """
        Add a new MultinodePenalty to the list

        Parameters
        ----------
        multinode_objective: Callable | MultinodeObjectiveFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Objective
        """

        super(MultinodeObjectiveList, self).add(
            option_type=MultinodeObjective,
            multinode_penalty=multinode_objective,
            _multinode_penalty_fcn=MultinodeObjectiveFcn,
            **extra_arguments
        )


class MultinodeObjectiveFunctions(MultinodePenaltyFunctions):
    pass


class MultinodeObjectiveFcn(FcnEnum):
    """
    Selection of valid multinode objective functions
    """

    STATES_EQUALITY = (MultinodeObjectiveFunctions.Functions.states_equality,)
    ALGEBRAIC_STATES_EQUALITY = (MultinodeObjectiveFunctions.Functions.algebraic_states_equality,)
    CONTROLS_EQUALITY = (MultinodeObjectiveFunctions.Functions.controls_equality,)
    CUSTOM = (MultinodeObjectiveFunctions.Functions.custom,)
    COM_EQUALITY = (MultinodeObjectiveFunctions.Functions.com_equality,)
    COM_VELOCITY_EQUALITY = (MultinodeObjectiveFunctions.Functions.com_velocity_equality,)
    TIME_CONSTRAINT = (MultinodeObjectiveFunctions.Functions.time_equality,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodeObjectiveFunctions
