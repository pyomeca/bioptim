from typing import Callable, Union
from warnings import warn
from enum import Enum

import biorbd
from casadi import vertcat, MX

from .constraints import ConstraintFunction
from .objective_functions import ObjectiveFunction
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric


class StateTransition(OptionGeneric):
    """
    A placeholder for a transition of state

    Attributes
    ----------
    base: ConstraintFunction
        The type of penalty the state transition is (Constraint if no weight, Mayer otherwise)
    weight: float
        The weight of the objective function. The transition is a constraint if weight is not specified
    quadratic: bool
        If the objective function is quadratic
    phase_pre_idx: int
        The index of the phase right before the transition
    custom_function: function
        The function to call if a custom transition function is provided
    """

    def __init__(self, phase_pre_idx: int = None, weight: float = None, custom_function: Callable = None, **params):
        """
        Parameters
        ----------
        phase_pre_idx: int
            The index of the phase right before the transition
        custom_function: function
            The function to call if a custom transition function is provided
        params:
            Generic parameters for options
        """

        super(StateTransition, self).__init__(**params)
        self.base = ConstraintFunction
        self.weight = weight
        self.quadratic = True
        self.phase_pre_idx = phase_pre_idx
        self.custom_function = custom_function


class StateTransitionList(UniquePerPhaseOptionList):
    """
    A list of StateTransition

    Methods
    -------
    add(self, transition: Union[Callable, "StateTransitionFcn"], phase: int = -1, **extra_arguments)
        Add a new StateTransition to the list
    """

    def add(self, transition: Union[Callable, "StateTransitionFcn"], **extra_arguments):
        """
        Add a new StateTransition to the list

        Parameters
        ----------
        transition: Union[Callable, "StateTransitionFcn"]
            The chosen state transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(transition, StateTransitionFcn):
            extra_arguments["custom_function"] = transition
            transition = StateTransitionFcn.CUSTOM
        super(StateTransitionList, self)._add(option_type=StateTransition, type=transition, phase=-1, **extra_arguments)


class StateTransitionFunctions:
    """
    Internal implementation of the state transitions

    Methods
    -------
    prepare_state_transitions(ocp: OptimalControlProgram, state_transitions: StateTransitionList) -> list
        Configure all the state transitions and put them in a list
    """

    class Functions:
        """
        Implementation of all the state transitions

        Methods
        -------
        continuous(ocp: OptimalControlProgram, transition: StateTransition)
            The most common continuity function, that is state before equals state after
        cyclic(ocp: OptimalControlProgram" transition: StateTransition)
            The continuity function applied to the last to first node
        impact(ocp: OptimalControlProgram, transition: StateTransition)
            A discontinuous function that simulates an inelastic impact of a new contact point
        custom(ocp: OptimalControlProgram, transition: StateTransition)
            Calls the custom transition function provided by the user
        __get_nlp_pre_and_post(ocp: OptimalControlProgram, phase_pre_idx: int)
            Get two consecutive nlp. If the "pre" phase is the last, then the next one is the first (circular)
        """

        @staticmethod
        def continuous(ocp, transition: StateTransition) -> MX:
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: StateTransition
                A reference to the state transition

            Returns
            -------
            The difference between the state after and before
            """

            if ocp.nlp[transition.phase_pre_idx].nx != ocp.nlp[(transition.phase_pre_idx + 1) % ocp.nb_phases].nx:
                raise RuntimeError(
                    "Continuous state transitions without same nx is not possible, please provide a custom state transition"
                )
            nlp_pre, nlp_post = StateTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            return nlp_pre.X[-1] - nlp_post.X[0]

        @staticmethod
        def cyclic(ocp, transition: StateTransition) -> MX:
            """
            The continuity function applied to the last to first node

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: StateTransition
                A reference to the state transition

            Returns
            -------
            The difference between the last and first node
            """
            return StateTransitionFunctions.Functions.continuous(ocp, transition)

        @staticmethod
        def impact(ocp, transition: StateTransition) -> MX:
            """
            A discontinuous function that simulates an inelastic impact of a new contact point

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: StateTransition
                A reference to the state transition

            Returns
            -------
            The difference between the last and first node after applying the impulse equations
            """

            if ocp.nlp[transition.phase_pre_idx].nx != ocp.nlp[(transition.phase_pre_idx + 1) % ocp.nb_phases].nx:
                raise RuntimeError(
                    "Impact transition without same nx is not possible, please provide a custom state transition"
                )

            # Aliases
            nlp_pre, nlp_post = StateTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            nbQ = nlp_pre.shape["q"]
            nbQdot = nlp_pre.shape["q_dot"]
            q = nlp_pre.mapping["q"].to_second.map(nlp_pre.X[-1][:nbQ])
            qdot_pre = nlp_pre.mapping["q_dot"].to_second.map(nlp_pre.X[-1][nbQ : nbQ + nbQdot])

            if nlp_post.model.nbContacts() == 0:
                warn("The chosen model does not have any contact")
            # A new model is loaded here so we can use pre Qdot with post model, this is a hack and should be dealt
            # a better way (e.g. create a supplementary variable in V that link the pre and post phase with a
            # constraint. The transition would therefore apply to node_0 and node_1 (with an augmented ns)
            model = biorbd.Model(nlp_post.model.path().absolutePath().to_string())
            func = biorbd.to_casadi_func(
                "impulse_direct", model.ComputeConstraintImpulsesDirect, nlp_pre.q, nlp_pre.q_dot
            )
            qdot_post = func(q, qdot_pre)
            qdot_post = nlp_post.mapping["q_dot"].to_first.map(qdot_post)

            val = nlp_pre.X[-1][:nbQ] - nlp_post.X[0][:nbQ]
            val = vertcat(val, qdot_post - nlp_post.X[0][nbQ : nbQ + nbQdot])
            return val

        @staticmethod
        def custom(ocp, transition: StateTransition) -> MX:
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: StateTransition
                A reference to the state transition

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            nlp_pre, nlp_post = StateTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            return transition.custom_function(nlp_pre.X[-1], nlp_post.X[0], **transition.params)

        @staticmethod
        def __get_nlp_pre_and_post(ocp, phase_pre_idx: int) -> tuple:
            """
            Get two consecutive nlp. If the "pre" phase is the last, then the next one is the first (circular)

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            phase_pre_idx: int
                The index of the phase right before the transition

            Returns
            -------
            The nlp before and after the transition
            """

            return ocp.nlp[phase_pre_idx], ocp.nlp[(phase_pre_idx + 1) % ocp.nb_phases]

    @staticmethod
    def prepare_state_transitions(ocp, state_transitions: StateTransitionList) -> list:
        """
        Configure all the state transitions and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        state_transitions: StateTransitionList
            The list of all the state transitions

        Returns
        -------
        The list of all the transitions prepared
        """

        # By default it assume Continuous. It can be change later
        full_state_transitions = [
            StateTransition(type=StateTransitionFcn.CONTINUOUS, phase_pre_idx=i) for i in range(ocp.nb_phases - 1)
        ]

        existing_phases = []
        for pt in state_transitions:
            if pt.phase_pre_idx is None and pt.type == StateTransitionFcn.CYCLIC:
                pt.phase_pre_idx = ocp.nb_phases - 1

            idx_phase = pt.phase_pre_idx
            if idx_phase in existing_phases:
                raise RuntimeError("It is not possible to define two state transitions for the same phase")
            if idx_phase >= ocp.nb_phases:
                raise RuntimeError("Phase index of the state transition is higher than the number of phases")
            existing_phases.append(idx_phase)

            if pt.weight:
                pt.base = ObjectiveFunction.MayerFunction

            if idx_phase == ocp.nb_phases - 1:
                # Add a cyclic constraint or objective
                full_state_transitions.append(pt)
            else:
                full_state_transitions[idx_phase] = pt
        return full_state_transitions


class ContinuityFunctions:
    """
    Interface between continuity and constraint
    """

    @staticmethod
    def continuity(ocp):
        """
        The declaration of inner- and inter-phase continuity constraints

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        ConstraintFunction.inner_phase_continuity(ocp)

        # Dynamics must be respected between phases
        for pt in ocp.state_transitions:
            pt.base.inter_phase_continuity(ocp, pt)


class StateTransitionFcn(Enum):
    """
    Selection of valid state transition functions
    """

    CONTINUOUS = (StateTransitionFunctions.Functions.continuous,)
    IMPACT = (StateTransitionFunctions.Functions.impact,)
    CYCLIC = (StateTransitionFunctions.Functions.cyclic,)
    CUSTOM = (StateTransitionFunctions.Functions.custom,)
