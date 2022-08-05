from typing import Callable, Union, Any

from ..misc.fcn_enum import FcnEnum
from ..misc.enums import Node, PenaltyType
from .objective_functions import Objective, ObjectiveFcn, ObjectiveFunction
from .penalty_node import PenaltyNodeList
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFunctions


# TODO: mirror multinode_constraint.py in here but for objectives
class MultinodeObjective(Objective, MultinodePenalty):
    """
    TODO: docstring
    """

    def __init__(
        self,
        phase_first_idx: int,
        phase_second_idx: int,
        first_node: Union[Node, int],
        second_node: Union[Node, int],
        multinode_objective: Union[Callable, Any] = None,
        custom_function: Callable = None,
        weight: float = 0,
        penalty_type: PenaltyType = PenaltyType.USER,
        **params: Any,
    ):
        """
        Parameters
        ----------
        phase_first_idx: int
            The first index of the phase of concern
        params:
            Generic parameters for options
        """
        if custom_function and not callable(custom_function):
            raise RuntimeError("custom_function must be callable")

        if isinstance(multinode_objective, FcnEnum):
            if MultinodeObjectiveFcn not in multinode_objective.get_fcn_types():
                raise RuntimeError(f"multinode_objective of type '{type(multinode_objective)}' not allowed")
        else:
            if not callable(multinode_objective):
                raise RuntimeError("multinode_objective must be callable")
            custom_function = multinode_objective
            multinode_objective = MultinodeObjectiveFcn.CUSTOM

        super(MultinodeObjective, self).__init__(
            objective=multinode_objective,
            custom_function=custom_function,
            custom_type=ObjectiveFcn.Mayer,
            weight=weight,
            penalty_type=penalty_type,
            **params,
        )
        MultinodePenalty.__init__(
            self,
            phase_first_idx=phase_first_idx,
            phase_second_idx=phase_second_idx,
            first_node=first_node,
            second_node=second_node,
            multinode_penalty=multinode_objective,  # TODO: might not be necessary to store here.
            **params,
        )
        self.base = ObjectiveFunction.MayerFunction

    def _add_penalty_to_pool(self, all_pn: Union[PenaltyNodeList, list, tuple]):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp

        if self.penalty_type == PenaltyType.USER:
            pool = nlp.J if nlp else ocp.J
        elif self.penalty_type == PenaltyType.INTERNAL:
            pool = nlp.J_internal if nlp else ocp.J_internal
        else:
            raise RuntimeError(
                f"penalty_type must be {PenaltyType.USER} or {PenaltyType.INTERNAL}, not {type(self.penalty_type)}"
            )

        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        if self.penalty_type == PenaltyType.USER:
            J_to_add_to = nlp.J if nlp else ocp.J
        elif self.penalty_type == PenaltyType.INTERNAL:
            J_to_add_to = nlp.J_internal if nlp else ocp.J_internal
        else:
            raise RuntimeError(
                f"penalty_type must be {PenaltyType.USER} or {PenaltyType.INTERNAL}, not {type(self.penalty_type)}"
            )

        if self.list_index < 0:
            for i, j in enumerate(J_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                J_to_add_to.append([])
                self.list_index = len(J_to_add_to) - 1
        else:
            while self.list_index >= len(J_to_add_to):
                J_to_add_to.append([])
            J_to_add_to[self.list_index] = []


class MultinodeObjectiveList(MultinodePenaltyList):
    """
    A list of Multi Node Constraint

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
        Add a new MultinodePenalty to the list
    print(self)
        Print the MultinodePenaltyList to the console
    prepare_multinode_penalties(self, ocp) -> list
        Configure all the multinode_objective and put them in a list
    """

    def add(self, multinode_objective: Any, **extra_arguments: Any):
        """
        Add a new MultinodeObjective to the list

        Parameters
        ----------
        multinode_objective: Union[Callable, MultinodePenaltyFcn]
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Penalty
        """

        if isinstance(multinode_objective, MultinodeObjective):
            self.copy(multinode_objective)
        else:
            super(MultinodeObjectiveList, self)._add(
                option_type=MultinodeObjective, multinode_objective=multinode_objective, **extra_arguments
            )


class MultinodeObjectiveFunctions(MultinodePenaltyFunctions, ObjectiveFunction.MayerFunction):
    @staticmethod
    def penalty_nature() -> str:
        return "multinode_objectives"


class MultinodeObjectiveFcn(FcnEnum):
    """
    Selection of valid multinode penalty functions
    """

    EQUALITY = (MultinodeObjectiveFunctions.Functions.equality,)
    CUSTOM = (MultinodeObjectiveFunctions.Functions.custom,)
    COM_EQUALITY = (MultinodeObjectiveFunctions.Functions.com_equality,)
    COM_VELOCITY_EQUALITY = (MultinodeObjectiveFunctions.Functions.com_velocity_equality,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodeObjectiveFunctions

    @staticmethod
    def get_fcn_types():
        """
        Returns the types of the enum
        """
        return (MultinodeObjectiveFcn,) + ObjectiveFcn.Mayer.get_fcn_types()
