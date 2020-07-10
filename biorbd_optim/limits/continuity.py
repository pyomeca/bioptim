from warnings import warn
from enum import Enum

import biorbd
from casadi import vertcat

from .constraints import ConstraintFunction
from .objective_functions import ObjectiveFunction
from .penalty import PenaltyFunctionAbstract
from ..misc.options_lists import UniquePerPhaseOptionList, OptionGeneric


class StateTransitionOption(OptionGeneric):
    def __init__(self, phase_pre_idx=None, weight=0, custom_function=None, **params):
        super(StateTransitionOption, self).__init__(**params)
        self.base = ConstraintFunction
        self.phase_pre_idx = phase_pre_idx

        self.weight = weight
        self.quadratic = True
        self.custom_function = custom_function


class StateTransitionList(UniquePerPhaseOptionList):
    def add(self, transition, phase=-1, **extra_arguments):
        if not isinstance(transition, StateTransition):
            extra_arguments["custom_function"] = transition
            transition = StateTransition.CUSTOM
        super(StateTransitionList, self)._add(
            option_type=StateTransitionOption, type=transition, phase=phase, **extra_arguments
        )


class StateTransitionFunctions:
    class Functions:
        @staticmethod
        def continuous(ocp, transition):
            """
            TODO
            """
            if ocp.nlp[transition.phase_pre_idx]["nx"] != ocp.nlp[(transition.phase_pre_idx + 1) % ocp.nb_phases]["nx"]:
                raise RuntimeError(
                    "Continuous state transitions without same nx is not possible, please provide a custom state transition"
                )
            nlp_pre, nlp_post = StateTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            return nlp_pre["X"][-1] - nlp_post["X"][0]

        @staticmethod
        def cyclic(ocp, transition):
            """
            TODO
            """
            return StateTransitionFunctions.Functions.continuous(ocp, transition)

        @staticmethod
        def impact(ocp, transition):
            """
            TODO
            """
            if ocp.nlp[transition.phase_pre_idx]["nx"] != ocp.nlp[(transition.phase_pre_idx + 1) % ocp.nb_phases]["nx"]:
                raise RuntimeError(
                    "Impact transition without same nx is not possible, please provide a custom state transition"
                )

            # Aliases
            nlp_pre, nlp_post = StateTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            nbQ = nlp_pre["nbQ"]
            nbQdot = nlp_pre["nbQdot"]
            q = nlp_pre["q_mapping"].expand.map(nlp_pre["X"][-1][:nbQ])
            qdot_pre = nlp_pre["q_dot_mapping"].expand.map(nlp_pre["X"][-1][nbQ : nbQ + nbQdot])

            if nlp_post["model"].nbContacts() == 0:
                warn("The chosen model does not have any contact")
            # A new model is loaded here so we can use pre Qdot with post model, this is a hack and should be dealt
            # a better way (e.g. create a supplementary variable in V that link the pre and post phase with a
            # constraint. The transition would therefore apply to node_0 and node_1 (with an augmented ns)
            model = biorbd.Model(nlp_post["model"].path().absolutePath().to_string())
            func = biorbd.to_casadi_func(
                "impulse_direct", model.ComputeConstraintImpulsesDirect, nlp_pre["q"], nlp_pre["qdot"]
            )
            qdot_post = func(q, qdot_pre)
            qdot_post = nlp_post["q_dot_mapping"].reduce.map(qdot_post)

            val = nlp_pre["X"][-1][:nbQ] - nlp_post["X"][0][:nbQ]
            val = vertcat(val, qdot_post - nlp_post["X"][0][nbQ : nbQ + nbQdot])
            return val

        @staticmethod
        def custom(ocp, transition):
            nlp_pre, nlp_post = StateTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, transition.phase_pre_idx)
            return transition.custom_function(nlp_pre["X"][-1], nlp_post["X"][0], **transition.params)

        @staticmethod
        def __get_nlp_pre_and_post(ocp, phase_pre_idx):
            return ocp.nlp[phase_pre_idx], ocp.nlp[(phase_pre_idx + 1) % ocp.nb_phases]

    @staticmethod
    def prepare_state_transitions(ocp, state_transitions):
        # By default it assume Continuous. It can be change later
        full_state_transitions = [
            StateTransitionOption(type=StateTransition.CONTINUOUS, phase_pre_idx=i) for i in range(ocp.nb_phases - 1)
        ]

        existing_phases = []
        for pt in state_transitions:
            if pt.phase_pre_idx is None and pt.type == StateTransition.CYCLIC:
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
    @staticmethod
    def continuity(ocp):
        ConstraintFunction.inner_phase_continuity(ocp)

        # Dynamics must be respected between phases
        for pt in ocp.state_transitions:
            pt.base.inter_phase_continuity(ocp, pt)


class StateTransition(Enum):
    """
    Different types of state transitions.
    """

    CONTINUOUS = (StateTransitionFunctions.Functions.continuous,)
    IMPACT = (StateTransitionFunctions.Functions.impact,)
    CYCLIC = (StateTransitionFunctions.Functions.cyclic,)
    CUSTOM = (StateTransitionFunctions.Functions.custom,)
