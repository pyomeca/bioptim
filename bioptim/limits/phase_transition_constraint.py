from typing import Callable, Union, Any

from .phase_transition import PhaseTransitionFunctions
from .multinode_penalty import MultinodePenaltyFunctions
from .multinode_constraint import MultinodeConstraint, MultinodeConstraintFcn
from .path_conditions import Bounds
from .objective_functions import ObjectiveFunction
from ..misc.enums import Node, PenaltyType
from ..misc.options import UniquePerPhaseOptionList


class PhaseTransitionConstraint(MultinodeConstraint):
    """
    A placeholder for a transition of state

    Attributes
    ----------
    min_bound: list
        The minimal bound of the phase transition
    max_bound: list
        The maximal bound of the phase transition
    bounds: Bounds
        The bounds (will be filled with min_bound/max_bound)
    weight: float
        The weight of the cost function
    quadratic: bool
        If the objective function is quadratic
    phase_pre_idx: int
        The index of the phase right before the transition
    phase_post_idx: int
        The index of the phase right after the transition
    node: Node
        The kind of node
    dt: float
        The delta time
    node_idx: int
        The index of the node in nlp pre
    transition: bool
        The nature of the cost function is transition
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    """

    def __init__(
        self,
        phase_pre_idx: int = None,
        transition: Union[Callable, Any] = None,
        weight: float = 0,
        custom_function: Callable = None,
        min_bound: float = 0,
        max_bound: float = 0,
        **params: Any,
    ):
        if custom_function and not callable(custom_function):
            raise RuntimeError("custom_function must be callable.")

        if not isinstance(transition, PhaseTransitionConstraintFcn):
            custom_function = transition
            transition = PhaseTransitionConstraintFcn.CUSTOM

        super(PhaseTransitionConstraint, self).__init__(
            phase_first_idx=phase_pre_idx,
            phase_second_idx=None,
            first_node=Node.END,
            second_node=Node.START,
            multinode_constraint=transition,
            custom_function=custom_function,
            min_bound=min_bound,
            max_bound=max_bound,
            weight=weight,
            **params,
        )

        self.node = Node.TRANSITION
        self.transition = True


class PhaseTransitionConstraintList(UniquePerPhaseOptionList):
    """
    A list of PhaseTransitionConstraint

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
        Add a new PhaseTransitionConstraint to the list
    print(self)
        Print the PhaseTransitionList to the console
    prepare_phase_transitions(self, ocp) -> list
        Configure all the phase transitions and put them in a list
    """

    def add(self, transition: Any, **extra_arguments: Any):
        """
        Add a new PhaseTransitionConstraint to the list

        Parameters
        ----------
        transition: Union[Callable, PhaseTransitionFcn]
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(transition, PhaseTransitionConstraint):
            self.copy(transition)
        else:
            super(PhaseTransitionConstraintList, self)._add(
                option_type=PhaseTransitionConstraint, transition=transition, phase=-1, **extra_arguments
            )

    def print(self):
        """
        Print the PhaseTransitionList to the console
        """
        raise NotImplementedError("Printing of PhaseTransitionList is not ready yet")

    def prepare_phase_transitions(self, ocp) -> list:
        """
        Configure all the added phase transition constraints and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of the added transition constraints prepared
        """

        # By default it assume Continuous. It can be change later TODO: not sure it is a good idea or that even relevant anymore
        # TODO: this here and in PhaseTransitionObective must go somewhere else! Otherwise there are always PhaseTransitionConstraint even when only objectives are wanted

        phase_transitions = []
        for pt in self:
            if (
                pt.phase_pre_idx is None and pt.type == PhaseTransitionConstraintFcn.CYCLIC
            ):  # what if phase_pre_idx is None and not CYCLIC?
                pt.phase_pre_idx = ocp.n_phases - 1
            pt.phase_post_idx = (pt.phase_pre_idx + 1) % ocp.n_phases

            idx_phase = pt.phase_pre_idx
            if idx_phase >= ocp.n_phases:
                raise RuntimeError("Phase index of the phase transition is higher than the number of phases")

            # TODO: wierd stuff was happening here with the weight

            if idx_phase == ocp.n_phases - 1:
                # Add a cyclic constraint or objective
                phase_transitions.append(pt)
            else:
                phase_transitions[idx_phase] = pt
        return phase_transitions

    @staticmethod
    def prepare_continuity(ocp) -> list:
        """
        Configure inter-phase transition continuity constraints and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of the inter-phase transition continuity constraints prepared
        """
        phase_transitions_continuity = [
            PhaseTransitionConstraint(phase_pre_idx=i, transition=PhaseTransitionConstraintFcn.CONTINUOUS)
            for i in range(ocp.n_phases - 1)
        ]
        for pt in phase_transitions_continuity:
            pt.phase_post_idx = (pt.phase_pre_idx + 1) % ocp.n_phases

        return phase_transitions_continuity


class PhaseTransitionConstraintFcn(MultinodeConstraintFcn):
    """
    Selection of valid phase transition functions
    """

    CONTINUOUS = PhaseTransitionFunctions.Functions.continuous
    IMPACT = PhaseTransitionFunctions.Functions.impact
    CYCLIC = PhaseTransitionFunctions.Functions.cyclic
    CUSTOM = MultinodePenaltyFunctions.Functions.custom

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return PhaseTransitionFunctions
