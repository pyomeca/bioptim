from typing import Callable, Union, Any
from warnings import warn
from enum import Enum

import biorbd
from casadi import vertcat, MX, Function

from .constraints import ConstraintFunction
from .objective_functions import ObjectiveFunction
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric


class PhaseTransition(OptionGeneric):
    """
    A placeholder for a transition of state

    Attributes
    ----------
    base: ConstraintFunction
        The type of penalty the phase transition is (Constraint if no weight, Mayer otherwise)
    casadi_function: Function
        The casadi function of the cost function
    custom_function: Callable
        The function to call if a custom transition function is provided
    phase_pre_idx: int
        The index of the phase right before the transition
    quadratic: bool
        If the objective function is quadratic
    weight: float
        The weight of the objective function. The transition is a constraint if weight is not specified
    """

    def __init__(
        self, phase_pre_idx: int = None, weight: float = None, custom_function: Callable = None, **params: Any
    ):
        """
        Parameters
        ----------
        phase_pre_idx: int
            The index of the phase right before the transition
        custom_function: Callable
            The function to call if a custom transition function is provided
        params:
            Generic parameters for options
        """

        super(PhaseTransition, self).__init__(**params)
        self.base = ConstraintFunction
        self.weight = weight
        self.quadratic = True
        self.phase_pre_idx = phase_pre_idx
        self.custom_function = custom_function
        self.casadi_function = None


class PhaseTransitionList(UniquePerPhaseOptionList):
    """
    A list of PhaseTransition

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
        Add a new PhaseTransition to the list
    print(self)
        Print the PhaseTransitionList to the console
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
        super(PhaseTransitionList, self)._add(option_type=PhaseTransition, type=transition, phase=-1, **extra_arguments)

    def print(self):
        """
        Print the PhaseTransitionList to the console
        """
        raise NotImplementedError("Printing of PhaseTransitionList is not ready yet")

    @ staticmethod
    def get_phase_transitions(ocp):
        list_phase_transitions = []
        for g in ocp.g:
            if "PHASE_TRANSITION" in g[0]['constraint'].name:
                list_phase_transitions.append(g[0]['bounds'].type.name)
        return list_phase_transitions


class PhaseTransitionFunctions:
    """
    Internal implementation of the phase transitions

    Methods
    -------
    prepare_phase_transitions(ocp: OptimalControlProgram, phase_transitions: PhaseTransitionList) -> list
        Configure all the phase transitions and put them in a list
    """

    class Functions:
        """
        Implementation of all the phase transitions

        Methods
        -------
        continuous(ocp: OptimalControlProgram, transition: PhaseTransition)
            The most common continuity function, that is state before equals state after
        cyclic(ocp: OptimalControlProgram" transition: PhaseTransition)
            The continuity function applied to the last to first node
        impact(ocp: OptimalControlProgram, transition: PhaseTransition)
            A discontinuous function that simulates an inelastic impact of a new contact point
        custom(ocp: OptimalControlProgram, transition: PhaseTransition)
            Calls the custom transition function provided by the user
        __get_nlp_pre_and_post(ocp: OptimalControlProgram, phase_pre_idx: int)
            Get two consecutive nlp. If the "pre" phase is the last, then the next one is the first (circular)
        """

        @staticmethod
        def continuous(ocp, transition: PhaseTransition) -> MX:
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: PhaseTransition
                A reference to the phase transition

            Returns
            -------
            The difference between the state after and before
            """

            if ocp.nlp[transition.phase_pre_idx].nx != ocp.nlp[(transition.phase_pre_idx + 1) % ocp.n_phases].nx:
                raise RuntimeError(
                    "Continuous phase transition without same number of states is not possible, "
                    "please provide a custom phase transition"
                )
            nlp_pre, nlp_post = PhaseTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            return nlp_pre.X[-1] - nlp_post.X[0]

        @staticmethod
        def cyclic(ocp, transition: PhaseTransition) -> MX:
            """
            The continuity function applied to the last to first node

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: PhaseTransition
                A reference to the phase transition

            Returns
            -------
            The difference between the last and first node
            """
            return PhaseTransitionFunctions.Functions.continuous(ocp, transition)

        @staticmethod
        def impact(ocp, transition: PhaseTransition) -> MX:
            """
            A discontinuous function that simulates an inelastic impact of a new contact point

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: PhaseTransition
                A reference to the phase transition

            Returns
            -------
            The difference between the last and first node after applying the impulse equations
            """

            if ocp.nlp[transition.phase_pre_idx].nx != ocp.nlp[(transition.phase_pre_idx + 1) % ocp.n_phases].nx:
                raise RuntimeError(
                    "Impact transition without same nx is not possible, please provide a custom phase transition"
                )

            # Aliases
            nlp_pre, nlp_post = PhaseTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            n_q = nlp_pre.shape["q"]
            n_qdot = nlp_pre.shape["qdot"]
            q = nlp_pre.mapping["q"].to_second.map(nlp_pre.X[-1][:n_q])
            qdot_pre = nlp_pre.mapping["qdot"].to_second.map(nlp_pre.X[-1][n_q : n_q + n_qdot])

            if nlp_post.model.nbContacts() == 0:
                warn("The chosen model does not have any contact")
            # A new model is loaded here so we can use pre Qdot with post model, this is a hack and should be dealt
            # a better way (e.g. create a supplementary variable in v that link the pre and post phase with a
            # constraint. The transition would therefore apply to node_0 and node_1 (with an augmented ns)
            model = biorbd.Model(nlp_post.model.path().absolutePath().to_string())
            func = biorbd.to_casadi_func(
                "impulse_direct", model.ComputeConstraintImpulsesDirect, nlp_pre.q, nlp_pre.qdot
            )
            qdot_post = func(q, qdot_pre)
            qdot_post = nlp_post.mapping["qdot"].to_first.map(qdot_post)

            val = nlp_pre.X[-1][:n_q] - nlp_post.X[0][:n_q]
            val = vertcat(val, qdot_post - nlp_post.X[0][n_q : n_q + n_qdot])
            return val

        @staticmethod
        def custom(ocp, transition: PhaseTransition) -> MX:
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            transition: PhaseTransition
                A reference to the phase transition

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            nlp_pre, nlp_post = PhaseTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
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

            return ocp.nlp[phase_pre_idx], ocp.nlp[(phase_pre_idx + 1) % ocp.n_phases]

    @staticmethod
    def prepare_phase_transitions(ocp, phase_transitions: PhaseTransitionList) -> list:
        """
        Configure all the phase transitions and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        phase_transitions: PhaseTransitionList
            The list of all the phase transitions

        Returns
        -------
        The list of all the transitions prepared
        """

        # By default it assume Continuous. It can be change later
        full_phase_transitions = [
            PhaseTransition(type=PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=i) for i in range(ocp.n_phases - 1)
        ]

        existing_phases = []
        for pt in phase_transitions:
            if pt.phase_pre_idx is None and pt.type == PhaseTransitionFcn.CYCLIC:
                pt.phase_pre_idx = ocp.n_phases - 1

            idx_phase = pt.phase_pre_idx
            if idx_phase in existing_phases:
                raise RuntimeError("It is not possible to define two phase transitions for the same phase")
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


class PhaseTransitionFcn(Enum):
    """
    Selection of valid phase transition functions
    """

    CONTINUOUS = (PhaseTransitionFunctions.Functions.continuous,)
    IMPACT = (PhaseTransitionFunctions.Functions.impact,)
    CYCLIC = (PhaseTransitionFunctions.Functions.cyclic,)
    CUSTOM = (PhaseTransitionFunctions.Functions.custom,)
