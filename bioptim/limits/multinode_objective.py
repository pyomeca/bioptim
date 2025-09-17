from typing import Any

from ..misc.enums import PenaltyType
from ..misc.fcn_enum import FcnEnum
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFunctions
from .objective_functions import ObjectiveFunction
from .weight import ObjectiveWeight


from ..misc.parameters_types import (
    Bool,
    Float,
    Int,
    IntorNodeIterable,
    IntTuple,
)


class MultinodeObjective(MultinodePenalty):
    def __init__(
        self,
        _multinode_penalty_fcn: Any,
        nodes: IntorNodeIterable,
        nodes_phase: IntTuple,
        weight: Int | Float | ObjectiveWeight,
        is_stochastic: Bool = False,
        **kwargs,
    ):

        if not isinstance(weight, ObjectiveWeight):
            if isinstance(weight, (int, float)):
                weight = ObjectiveWeight(weight)
            else:
                raise ValueError(f"The weight must be a ObjectiveWeight, int or float, not {type(weight)}")

        super(MultinodeObjective, self).__init__(
            _multinode_penalty_fcn=_multinode_penalty_fcn, nodes=nodes, nodes_phase=nodes_phase, weight=weight, **kwargs
        )

        self.quadratic = kwargs["quadratic"] if "quadratic" in kwargs else True
        self.base = ObjectiveFunction.MayerFunction
        self.is_stochastic = is_stochastic

    def _get_pool_to_add_penalty(self, ocp, nlp):
        if self.penalty_type == PenaltyType.INTERNAL:
            pool = nlp.J_internal if nlp else ocp.J_internal
        elif self.penalty_type == PenaltyType.USER:
            pool = nlp.J if nlp else ocp.J
        else:
            raise ValueError(f"Invalid objective type {self.penalty_type}.")

        return pool


class MultinodeObjectiveList(MultinodePenaltyList):
    """
    A list of Multinode Objective

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new MultinodeObjective to the list
    """

    def add(
        self,
        multinode_objective: Any,
        weight: Int | Float | ObjectiveWeight = None,
        **extra_arguments: Any,
    ):
        """
        Add a new MultinodePenalty to the list

        Parameters
        ----------
        multinode_objective: Callable | MultinodeObjectiveFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Objective
        """
        if weight is None:
            weight = ObjectiveWeight()

        if not isinstance(weight, ObjectiveWeight):
            if isinstance(weight, (int, float)):
                weight = ObjectiveWeight(weight)
            else:
                raise ValueError(f"The weight must be a ObjectiveWeight, int or float, not {type(weight)}")

        super(MultinodeObjectiveList, self).add(
            option_type=MultinodeObjective,
            weight=weight,
            multinode_penalty=multinode_objective,
            _multinode_penalty_fcn=MultinodeObjectiveFcn,
            **extra_arguments,
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
