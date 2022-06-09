from typing import Callable, Union, Any

from .objective_functions import Objective
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFcn

# TODO: mirror multinode_constraint.py in here but for objectives
class MultinodeObjective(MultinodePenalty, Objective):
    """
    TODO: docstring
    """

    pass


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
            option_type=MultinodeObjective, multinode_penalty=multinode_objective, phase=-1, **extra_arguments
        )


MultinodeObjectiveFcn = MultinodePenaltyFcn
