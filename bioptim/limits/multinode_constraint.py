from typing import Callable, Union, Any

from .constraints import Constraint
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFcn

class MultinodeConstraint(MultinodePenalty, Constraint):
    """
    TODO: docstring
    """
    pass


class MultinodeConstraintList(MultinodePenaltyList):
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

    def add(self, multinode_constraint: Any, **extra_arguments: Any):
        """
        Add a new MultinodePenalty to the list

        Parameters
        ----------
        multinode_constraint: Union[Callable, MultinodePenaltyFcn]
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Penalty
        """

        if not isinstance(multinode_constraint, MultinodePenaltyFcn):
            extra_arguments["custom_function"] = multinode_constraint
            multinode_constraint = MultinodePenaltyFcn.CUSTOM
        super(MultinodeConstraintList, self)._add(
            option_type=MultinodeConstraint, multinode_penalty=multinode_constraint, phase=-1, **extra_arguments
        )
