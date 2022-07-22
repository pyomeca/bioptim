from typing import Callable, Union, Any

from .constraints import Constraint, ConstraintFcn
from ..misc.enums import Node
from ..misc.options import UniquePerPhaseOptionList
from .penalty_node import PenaltyNodeList
from .multinode_penalty import MultinodePenalty, MultinodePenaltyFunctions


class MultinodeConstraint(Constraint, MultinodePenalty):
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
            raise RuntimeError("custom_function must be callable.")

        if not isinstance(multinode_constraint, MultinodeConstraintFcn):
            custom_function = multinode_constraint
            multinode_constraint = MultinodeConstraintFcn.CUSTOM

        super(MultinodeConstraint, self).__init__(
            constraint=multinode_constraint,
            custom_function=custom_function,
            min_bound=min_bound,
            max_bound=max_bound,
            **params,
        )
        MultinodePenalty.__init__(
            self,
            phase_first_idx=phase_first_idx,
            phase_second_idx=phase_second_idx,
            first_node=first_node,
            second_node=second_node,
            multinode_penalty=multinode_constraint,
            **params,
        )

    def _add_penalty_to_pool(self, all_pn: Union[PenaltyNodeList, list, tuple]):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp

        pool = nlp.g_internal if nlp else ocp.g_internal
        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        pool_to_add_to = nlp.g_internal if nlp else ocp.g_internal

        if self.list_index < 0:
            for i, j in enumerate(pool_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                pool_to_add_to.append([])
                self.list_index = len(pool_to_add_to) - 1
        else:
            while self.list_index >= len(pool_to_add_to):
                pool_to_add_to.append([])
            pool_to_add_to[self.list_index] = []


class MultinodeConstraintList(UniquePerPhaseOptionList):
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

        if isinstance(multinode_constraint, MultinodeConstraint):
            self.copy(multinode_constraint)
        else:
            super(MultinodeConstraintList, self)._add(
                option_type=MultinodeConstraint, multinode_constraint=multinode_constraint, phase=-1, **extra_arguments
            )

    def prepare_multinode_constraints(self, ocp) -> list:
        """
        Configure all the phase transitions and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of all the multi_node constraints prepared
        """
        full_phase_multinode_penalty = []
        existing_phases = []
        for mnc in self:

            idx_phase = mnc.phase_first_idx
            if mnc.phase_first_idx >= ocp.n_phases or mnc.phase_second_idx >= ocp.n_phases:
                raise RuntimeError("Phase index of the multinode_penalty is higher than the number of phases")
            if mnc.phase_first_idx < 0 or mnc.phase_second_idx < 0:
                raise RuntimeError("Phase index of the multinode_penalty need to be positive")
            existing_phases.append(idx_phase)

            full_phase_multinode_penalty.append(mnc)

        return full_phase_multinode_penalty


class MultinodeConstraintFcn(ConstraintFcn):
    """
    Selection of valid multinode penalty functions
    """

    EQUALITY = MultinodePenaltyFunctions.Functions.equality
    CUSTOM = MultinodePenaltyFunctions.Functions.custom
    COM_EQUALITY = MultinodePenaltyFunctions.Functions.com_equality
    COM_VELOCITY_EQUALITY = MultinodePenaltyFunctions.Functions.com_velocity_equality

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodePenaltyFunctions
