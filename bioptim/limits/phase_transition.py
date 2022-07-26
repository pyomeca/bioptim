from typing import Callable, Union, Any
from warnings import warn

import biorbd_casadi as biorbd
from casadi import vertcat, MX

from .penalty_option import PenaltyOption
from .multinode_penalty import MultinodePenaltyFunctions
from .path_conditions import Bounds
from .objective_functions import ObjectiveFunction
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyNodeList
from ..misc.enums import Node, PenaltyType, InterpolationType
from ..misc.fcn_enum import FcnEnum, Fcn
from ..misc.options import UniquePerPhaseOptionList


class PhaseTransition(PenaltyOption):
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
        weight: float = None,
        custom_function: Callable = None,
        min_bound: float = 0,
        max_bound: float = 0,
        relaxed: bool = False,
        **params: Any,
    ):

        if custom_function and not callable(custom_function):
            raise RuntimeError("custom_function must be callable")

        if isinstance(transition, Fcn):
            if PhaseTransitionFcn not in transition.get_fcn_types():
                raise RuntimeError(f"transition of type '{type(transition)}' not allowed")
        else:
            custom_function = transition
            transition = PhaseTransitionFcn.CUSTOM

        if not relaxed and weight is not None:
            raise RuntimeError("if relaxed=False, weight must be None")

        if relaxed and weight is None:
            raise RuntimeError("if phase transition is relaxed, weight must be a float")

        super(PhaseTransition, self).__init__(
            penalty=transition,
            custom_function=custom_function,
            **params,
        )
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(interpolation=InterpolationType.CONSTANT)

        self.first_node = Node.END
        self.second_node = Node.START
        self.weight = weight
        self.quadratic = True
        self.phase_pre_idx = phase_pre_idx
        self.phase_post_idx = None
        self.node = Node.TRANSITION
        self.dt = 1
        self.node_idx = [0]
        self.transition = True
        self.relaxed = relaxed

    def _add_penalty_to_pool(self, all_pn: Union[PenaltyNodeList, list, tuple]):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp

        if self.relaxed and isinstance(self.weight, (int, float)):
            pool = nlp.J_internal if nlp else ocp.J_internal
        else:
            pool = nlp.g_internal if nlp else ocp.g_internal

        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        if self.relaxed and isinstance(self.weight, (int, float)):
            pool_to_add_to = nlp.J_internal if nlp else ocp.J_internal
        else:
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


class PhaseTransitionList(UniquePerPhaseOptionList):
    """
    A list of PhaseTransition

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
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
        transition: Union[Callable, PhaseTransitionFcn]
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

    def prepare_phase_transitions(self, ocp, relax_continuity: bool, continuity_weight=None) -> list:
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
            PhaseTransition(
                phase_pre_idx=i,
                transition=PhaseTransitionFcn.CONTINUOUS,
                relaxed=relax_continuity,
                weight=continuity_weight,
            )
            for i in range(ocp.n_phases - 1)
        ]
        for pt in full_phase_transitions:
            pt.phase_post_idx = (pt.phase_pre_idx + 1) % ocp.n_phases

        for pt in self:
            if pt.phase_pre_idx is None and pt.type == PhaseTransitionFcn.CYCLIC:
                pt.phase_pre_idx = ocp.n_phases - 1
            pt.phase_post_idx = (pt.phase_pre_idx + 1) % ocp.n_phases

            idx_phase = pt.phase_pre_idx
            if idx_phase >= ocp.n_phases:
                raise RuntimeError("Phase index of the phase transition is higher than the number of phases")

            if pt.relaxed:
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
        def continuous(transition, all_pn):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            return MultinodePenaltyFunctions.Functions.equality(transition, all_pn)

        @staticmethod
        def cyclic(transition, all_pn) -> MX:
            """
            The continuity function applied to the last to first node

            Parameters
            ----------
            transition: PhaseTransition
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node
            """

            return MultinodePenaltyFunctions.Functions.equality(transition, all_pn)

        @staticmethod
        def impact(transition, all_pn):
            """
            A discontinuous function that simulates an inelastic impact of a new contact point

            Parameters
            ----------
            transition: PhaseTransition
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node after applying the impulse equations
            """

            ocp = all_pn[0].ocp
            if ocp.nlp[transition.phase_pre_idx].states.shape != ocp.nlp[transition.phase_post_idx].states.shape:
                raise RuntimeError(
                    "Impact transition without same nx is not possible, please provide a custom phase transition"
                )

            # Aliases
            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp

            # A new model is loaded here so we can use pre Qdot with post model, this is a hack and should be dealt
            # a better way (e.g. create a supplementary variable in v that link the pre and post phase with a
            # constraint. The transition would therefore apply to node_0 and node_1 (with an augmented ns)
            model = biorbd.Model(nlp_post.model.path().absolutePath().to_string())

            if nlp_post.model.nbContacts() == 0:
                warn("The chosen model does not have any contact")
            q_pre = nlp_pre.states["q"].mx
            qdot_pre = nlp_pre.states["qdot"].mx
            qdot_impact = model.ComputeConstraintImpulsesDirect(q_pre, qdot_pre).to_mx()

            val = []
            cx_end = []
            cx = []
            for key in nlp_pre.states:
                cx_end = vertcat(cx_end, nlp_pre.states[key].mapping.to_second.map(nlp_pre.states[key].cx_end))
                cx = vertcat(cx, nlp_post.states[key].mapping.to_second.map(nlp_post.states[key].cx))
                post_mx = nlp_post.states[key].mx
                continuity = nlp_post.states["qdot"].mapping.to_first.map(
                    qdot_impact - post_mx if key == "qdot" else nlp_pre.states[key].mx - post_mx
                )
                val = vertcat(val, continuity)

            name = f"PHASE_TRANSITION_{nlp_pre.phase_idx}_{nlp_post.phase_idx}"
            func = biorbd.to_casadi_func(name, val, nlp_pre.states.mx, nlp_post.states.mx)(cx_end, cx)
            return func


class PhaseTransitionFcn(FcnEnum):
    """
    Selection of valid phase transition functions
    """

    CONTINUOUS = Fcn(PhaseTransitionFunctions.Functions.continuous)
    IMPACT = Fcn(PhaseTransitionFunctions.Functions.impact)
    CYCLIC = Fcn(PhaseTransitionFunctions.Functions.cyclic)
    CUSTOM = Fcn(MultinodePenaltyFunctions.Functions.custom)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return PhaseTransitionFunctions

    @staticmethod
    def get_fcn_types():
        """
        Returns the types of the enum
        """
        return (PhaseTransitionFcn,)
