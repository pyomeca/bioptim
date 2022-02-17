from typing import Callable, Union, Any
from warnings import warn
from enum import Enum

import biorbd_casadi as biorbd
from casadi import vertcat, MX

from .constraints import Constraint
from .path_conditions import Bounds
from .objective_functions import ObjectiveFunction
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyNodeList
from ..misc.enums import Node, InterpolationType, ConstraintType
from ..misc.options import UniquePerPhaseOptionList


class MultiNodeConstraint(Constraint):
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
    multi_node_constraint: Union[Callable, Any]
        The nature of the cost function is the multi node constraint
    constraint_type: ConstraintType
        If the penalty is from the user or from bioptim (implicit or internal)


    """

    def __init__(
        self,
        phase_first_idx: int = None,
        phase_second_idx: int = None,
        first_node: Node = None,
        second_node: Node = None,
        multi_node_constraint: Union[Callable, Any] = None,
        weight: float = 0,
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

        if not isinstance(multi_node_constraint, MultiNodeConstraintFcn):
            custom_function = multi_node_constraint
            transition = MultiNodeConstraintFcn.CUSTOM
        super(Constraint, self).__init__(penalty=multi_node_constraint, custom_function=custom_function, **params)

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(interpolation=InterpolationType.CONSTANT)

        self.weight = weight
        self.quadratic = True
        self.phase_first_idx = phase_first_idx
        self.phase_second_idx = phase_second_idx
        self.phase_pre_idx = phase_second_idx
        self.phase_post_idx = phase_first_idx
        self.first_node = first_node
        self.second_node = second_node
        self.node = [self.first_node, self.second_node]
        self.dt = 1
        self.node_idx = [0]
        self.multi_node_constraint = True
        self.constraint_type = ConstraintType.INTERNAL

    def _add_penalty_to_pool(self, all_pn: Union[PenaltyNodeList, list, tuple]):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp
        if self.weight == 0:
            pool = nlp.g_internal if nlp else ocp.g_internal
        else:
            pool = nlp.J_internal if nlp else ocp.J_internal
        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        if self.weight == 0:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal
        else:
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


class MultiNodeConstraintList(UniquePerPhaseOptionList):
    """
    A list of Multi Node Constraint

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
        Add a new MultiNodeConstraint to the list
    print(self)
        Print the MultiNodeConstraintList to the console
    prepare_multi_node_constraint(self, ocp) -> list
        Configure all the multi_node_constraint and put them in a list
    """

    def add(self, multi_node_constraint: Any, **extra_arguments: Any):
        """
        Add a new MultiNodeConstraint to the list

        Parameters
        ----------
        multi_node_constraint: Union[Callable, MultiNodeConstraintFcn]
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(multi_node_constraint, MultiNodeConstraintFcn):
            extra_arguments["custom_function"] = multi_node_constraint
            multi_node_constraint = MultiNodeConstraintFcn.CUSTOM
        super(MultiNodeConstraintList, self)._add(
            option_type=MultiNodeConstraint, multi_node_constraint=multi_node_constraint, phase=-1, **extra_arguments
        )

    def print(self):
        """
        Print the MultiNodeConstraintList to the console
        """
        raise NotImplementedError("Printing of MultiNodeConstraintList is not ready yet")

    def prepare_multi_node_constraints(self, ocp) -> list:
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
        full_phase_multi_node_constraint = []
        existing_phases = []
        for mnc in self:

            idx_phase = mnc.phase_first_idx
            if mnc.phase_first_idx >= ocp.n_phases or mnc.phase_second_idx >= ocp.n_phases:
                raise RuntimeError("Phase index of the multi_node_constraint is higher than the number of phases")
            existing_phases.append(idx_phase)

            if mnc.weight:
                mnc.base = ObjectiveFunction.MayerFunction

            full_phase_multi_node_constraint.append(mnc)

        return full_phase_multi_node_constraint


class MultiNodeConstraintFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """

    class Functions:
        """
        Implementation of all the Multi Node Constraint
        """

        @staticmethod
        def continuous(multi_node_constraint, all_pn):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            multi_node_constraint : MultiNodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            states_pre = multi_node_constraint.states_mapping.to_second.map(nlp_pre.states.cx_end)
            states_post = multi_node_constraint.states_mapping.to_first.map(nlp_post.states.cx)

            if states_pre.shape != states_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {states_pre.shape} in the "
                    f"pre-transition phase and {states_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            return states_pre - states_post

        @staticmethod
        def cyclic(multi_node_constraint, all_pn) -> MX:
            """
            The continuity function applied to the last to first node

            Parameters
            ----------
            multi_node_constraint : MultiNodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node
            """
            return MultiNodeConstraintFunctions.Functions.continuous(multi_node_constraint, all_pn)

        @staticmethod
        def impact(multi_node_constraint, all_pn):
            """
            A discontinuous function that simulates an inelastic impact of a new contact point

            Parameters
            ----------
            multi_node_constraint : MultiNodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node after applying the impulse equations
            """

            ocp = all_pn[0].ocp
            if (
                ocp.nlp[multi_node_constraint.phase_pre_idx].states.shape
                != ocp.nlp[multi_node_constraint.phase_post_idx].states.shape
            ):
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

        @staticmethod
        def custom(multi_node_constraint, all_pn, **extra_params):
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            multi_node_constraint: MultiNodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            return multi_node_constraint.custom_function(
                multi_node_constraint, nlp_pre.states, nlp_post.states, **extra_params
            )


class MultiNodeConstraintFcn(Enum):
    """
    Selection of valid phase transition functions
    """

    CONTINUOUS = (MultiNodeConstraintFunctions.Functions.continuous,)
    IMPACT = (MultiNodeConstraintFunctions.Functions.impact,)
    CYCLIC = (MultiNodeConstraintFunctions.Functions.cyclic,)
    CUSTOM = (MultiNodeConstraintFunctions.Functions.custom,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultiNodeConstraintFunctions
