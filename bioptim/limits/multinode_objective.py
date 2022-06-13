from typing import Callable, Union, Any

from ..misc.enums import Node
from .objective_functions import Objective
from .penalty_node import PenaltyNodeList
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFcn

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
        min_bound: float = 0,
        max_bound: float = 0,
        weight: float = 0,
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
        super(MultinodeObjective, self).__init__(
            phase_first_idx=phase_first_idx,
            phase_second_idx=phase_second_idx,
            first_node=first_node,
            second_node=second_node,
            multinode_penalty=multinode_objective,
            objective=multinode_objective,
            custom_function=custom_function,
            min_bound=min_bound,
            max_bound=max_bound,
            weight=weight,
            **params,
        )

    def _add_penalty_to_pool(self, all_pn: Union[PenaltyNodeList, list, tuple]):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp

        pool = nlp.J_internal if nlp else ocp.J_internal
        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        g_to_add_to = nlp.J_internal if nlp else ocp.J_internal

        if self.list_index < 0:
            for i, j in enumerate(g_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                g_to_add_to.append([])
                self.list_index = len(g_to_add_to) - 1
        else:
            while self.list_index >= len(g_to_add_to):
                g_to_add_to.append([])
            g_to_add_to[self.list_index] = []


class MultinodeObjectiveList(MultinodePenaltyList):
    """
    A list of Multi Node Constraint

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
        Add a new MultinodePenalty to the list
    print(self)
        Print the MultinodePenaltyList to the console
    prepare_multinode_penalty(self, ocp) -> list
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

        if not isinstance(multinode_objective, MultinodePenaltyFcn):
            extra_arguments["custom_function"] = multinode_objective
            multinode_objective = MultinodePenaltyFcn.CUSTOM
        super(MultinodeObjectiveList, self)._add(
            option_type=MultinodeObjective, multinode_objective=multinode_objective, phase=-1, **extra_arguments
        )


MultinodeObjectiveFcn = MultinodePenaltyFcn
