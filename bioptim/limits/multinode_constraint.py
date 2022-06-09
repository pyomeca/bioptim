from typing import Callable, Union, Any

from .constraints import Constraint
from ..misc.enums import Node
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFcn


class MultinodeConstraint(MultinodePenalty, Constraint):  # call MultinodePenalty methods first
    """
        A placeholder for a multi node constraints

        Attributes
        ----------
        min_bound: list
            The minimal bound of the multi node constraints
        max_bound: list
            The maximal bound of the multi node constraints
        bounds: Bounds
            The bounds (will be filled with min_bound/max_bound)
        weight: float
            The weight of the cost function
        quadratic: bool
            If the objective function is quadratic
        phase_first_idx: int
            The first index of the phase of concern
        phase_second_idx: int
            The second index of the phase of concern
        first_node: Node
            The kind of the first node
        second_node: Node
            The kind of the second node
        dt: float
            The delta time
        node_idx: int
            The index of the node in nlp pre
        multinode_constraint: Union[Callable, Any]
            The nature of the cost function is the multi node constraint
        penalty_type: PenaltyType
            If the penalty is from the user or from bioptim (implicit or internal)
        """

    def __init__(
        self,
        phase_first_idx: int,
        phase_second_idx: int,
        first_node: Union[Node, int],
        second_node: Union[Node, int],
        multinode_constraint: Union[Callable, Any] = None,
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
        super(MultinodeConstraint, self).__init__(phase_first_idx=phase_first_idx,
                                                  phase_second_idx=phase_second_idx,
                                                  first_node=first_node,
                                                  second_node=second_node,
                                                  multinode_penalty=multinode_constraint,
                                                  constraint=multinode_constraint,
                                                  custom_function=custom_function,
                                                  min_bound=min_bound,
                                                  max_bound=max_bound,
                                                  weight=weight,
                                                  **params)


class MultinodeConstraintList(MultinodePenaltyList):
    """
    A list of Multi Node Constraint

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
        Add a new MultinodeConstraint to the list
    print(self)
        Print the MultinodeConstraintList to the console
    prepare_multinode_penalty(self, ocp) -> list
        Configure all the multinode_constraint and put them in a list
    """

    def add(self, multinode_constraint: Any, **extra_arguments: Any):
        """
        Add a new MultinodeConstraint to the list

        Parameters
        ----------
        multinode_constraint: Union[Callable, MultinodeConstraintFcn]
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Penalty
        """

        if not isinstance(multinode_constraint, MultinodeConstraintFcn):
            extra_arguments["custom_function"] = multinode_constraint
            multinode_constraint = MultinodeConstraintFcn.CUSTOM
        super(MultinodeConstraintList, self)._add(
            option_type=MultinodeConstraint, multinode_penalty=multinode_constraint, phase=-1, **extra_arguments
        )


MultinodeConstraintFcn = MultinodePenaltyFcn
