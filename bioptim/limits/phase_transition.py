from typing import Callable, Any
from warnings import warn

from casadi import vertcat, MX

from .multinode_constraint import BinodeConstraint, BinodeConstraintFunctions
from .path_conditions import Bounds
from .objective_functions import ObjectiveFunction
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyController
from ..misc.enums import Node, PenaltyType
from ..misc.fcn_enum import FcnEnum
from ..misc.options import UniquePerPhaseOptionList


class PhaseTransition(BinodeConstraint):
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
        transition: Callable | Any = None,
        weight: float = 0,
        custom_function: Callable = None,
        min_bound: float = 0,
        max_bound: float = 0,
        **params: Any,
    ):
        if not isinstance(transition, PhaseTransitionFcn):
            custom_function = transition
            transition = PhaseTransitionFcn.CUSTOM
        super(PhaseTransition, self).__init__(
            phase_first_idx=phase_pre_idx,
            phase_second_idx=None,
            first_node=Node.END,
            second_node=Node.START,
            binode_constraint=transition,
            custom_function=custom_function,
            min_bound=min_bound,
            max_bound=max_bound,
            weight=weight if weight is not None else 0,
            force_binode=True,
            **params,
        )

        self.node = Node.TRANSITION
        self.transition = True


class PhaseTransitionList(UniquePerPhaseOptionList):
    """
    A list of PhaseTransition

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new PhaseTransition to the list
    print(self)
        Print the PhaseTransitionList to the console
    prepare_phase_transitions(self, ocp) -> list
        Configure all the phase transitions and put them in a list
    """

    def add(self, transition: Any, **extra_arguments: Any):
        """
        Add a new PhaseTransition to the list

        Parameters
        ----------
        transition: Callable | PhaseTransitionFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(transition, PhaseTransitionFcn):
            extra_arguments["custom_function"] = transition
            transition = PhaseTransitionFcn.CUSTOM
        super(PhaseTransitionList, self)._add(
            option_type=PhaseTransition, transition=transition, phase=-1, **extra_arguments
        )

    def print(self):
        """
        Print the PhaseTransitionList to the console
        """
        raise NotImplementedError("Printing of PhaseTransitionList is not ready yet")

    def prepare_phase_transitions(self, ocp, continuity_weight: float = None) -> list:
        """
        Configure all the phase transitions and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of all the transitions prepared
        """

        # By default it assume Continuous. It can be change later
        full_phase_transitions = [
            PhaseTransition(phase_pre_idx=i, transition=PhaseTransitionFcn.CONTINUOUS, weight=continuity_weight)
            for i in range(ocp.n_phases - 1)
        ]
        for pt in full_phase_transitions:
            pt.phase_post_idx = (pt.phase_pre_idx + 1) % ocp.n_phases

        existing_phases = []

        for pt in self:
            if pt.phase_pre_idx is None:
                if pt.type == PhaseTransitionFcn.CYCLIC:
                    pt.phase_pre_idx = ocp.n_phases - 1
            else:
                pt.phase_post_idx = (pt.phase_pre_idx + 1) % ocp.n_phases

            idx_phase = pt.phase_pre_idx
            if idx_phase >= ocp.n_phases:
                raise RuntimeError("Phase index of the phase transition is higher than the number of phases")
            existing_phases.append(idx_phase)

            if pt.weight:
                pt.base = ObjectiveFunction.MayerFunction

            if idx_phase == ocp.n_phases - 1:
                # Add a cyclic constraint or objective
                full_phase_transitions.append(pt)
            else:
                full_phase_transitions[idx_phase] = pt
        return full_phase_transitions


class PhaseTransitionFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """

    class Functions:
        """
        Implementation of all the phase transitions
        """

        @staticmethod
        def continuous(transition, controllers: list[PenaltyController, PenaltyController]):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            return BinodeConstraintFunctions.Functions.states_equality(transition, controllers, "all")

        @staticmethod
        def discontinuous(transition, controllers: list[PenaltyController, PenaltyController]):
            """
            There is no continuity constraints on the states

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            return MX.zeros(0, 0)

        @staticmethod
        def cyclic(transition, controllers: list[PenaltyController, PenaltyController]) -> MX:
            """
            The continuity function applied to the last to first node

            Parameters
            ----------
            transition: PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node
            """

            return BinodeConstraintFunctions.Functions.states_equality(transition, controllers, "all")

        @staticmethod
        def impact(transition, controllers: list[PenaltyController, PenaltyController]):
            """
            A discontinuous function that simulates an inelastic impact of a new contact point

            Parameters
            ----------
            transition: PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node after applying the impulse equations
            """

            ocp = controllers[0].ocp
            if ocp.nlp[transition.phase_pre_idx].states.shape != ocp.nlp[transition.phase_post_idx].states.shape:
                raise RuntimeError(
                    "Impact transition without same nx is not possible, please provide a custom phase transition"
                )

            # Aliases
            pre, post = controllers

            # A new model is loaded here so we can use pre Qdot with post model, this is a hack and should be dealt
            # a better way (e.g. create a supplementary variable in v that link the pre and post phase with a
            # constraint. The transition would therefore apply to node_0 and node_1 (with an augmented ns)
            # EDIT 1: using multinode constraint this should work now
            model = post.model.copy()

            if post.model.nb_rigid_contacts == 0:
                warn("The chosen model does not have any rigid contact")

            # Todo scaled?
            q_pre = pre.states["q"].mx
            qdot_pre = pre.states["qdot"].mx
            qdot_impact = model.qdot_from_impact(q_pre, qdot_pre)

            val = []
            cx_start = []
            cx_end = []
            for key in pre.states:
                cx_end = vertcat(cx_end, pre.states[key].mapping.to_second.map(pre.states[key].cx_end))
                cx_start = vertcat(cx_start, post.states[key].mapping.to_second.map(post.states[key].cx_start))
                post_mx = post.states[key].mx
                continuity = post.states["qdot"].mapping.to_first.map(
                    qdot_impact - post_mx if key == "qdot" else pre.states[key].mx - post_mx
                )
                val = vertcat(val, continuity)

            name = f"PHASE_TRANSITION_{pre.phase_idx}_{post.phase_idx}"
            func = pre.to_casadi_func(name, val, pre.states.mx, post.states.mx)(cx_end, cx_start)
            return func


class PhaseTransitionFcn(FcnEnum):
    """
    Selection of valid phase transition functions
    """

    CONTINUOUS = (PhaseTransitionFunctions.Functions.continuous,)
    DISCONTINUOUS = (PhaseTransitionFunctions.Functions.discontinuous,)
    IMPACT = (PhaseTransitionFunctions.Functions.impact,)
    CYCLIC = (PhaseTransitionFunctions.Functions.cyclic,)
    CUSTOM = (BinodeConstraintFunctions.Functions.custom,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return PhaseTransitionFunctions
