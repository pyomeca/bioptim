from warnings import warn
from enum import Enum

from casadi import vertcat

from .constraints import ConstraintFunction
from .objective_functions import ObjectiveFunction
from .penalty import PenaltyFunctionAbstract


class PhaseTransitionFunctions:
    class Functions:
        @staticmethod
        def continuous(ocp, phase_pre_idx, **unused):
            """
            TODO
            """
            if ocp.nlp[phase_pre_idx]["nx"] != ocp.nlp[(phase_pre_idx + 1) % ocp.nb_phases]["nx"]:
                raise RuntimeError(
                    "Continuous phase constraints without same nx is not possible, "
                    "please provide a custom phase transition"
                )
            nlp_pre, nlp_post = PhaseTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, phase_pre_idx)
            return nlp_pre["X"][-1] - nlp_post["X"][0]

        @staticmethod
        def cyclic(ocp, **kwargs):
            """
            TODO
            """
            return PhaseTransitionFunctions.Functions.continuous(ocp, **kwargs)

        @staticmethod
        def impact(ocp, phase_pre_idx, **unused):
            """
            TODO
            """
            if ocp.nlp[phase_pre_idx]["nx"] != ocp.nlp[(phase_pre_idx + 1) % ocp.nb_phases]["nx"]:
                raise RuntimeError(
                    "Impact phase constraints without same nx is not possible, "
                    "please provide a custom phase transition"
                )

            # Aliases
            nlp_pre, nlp_post = PhaseTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, phase_pre_idx)
            nbQ_reduced = nlp_pre["nbQ"]
            q_reduced = nlp_post["X"][0][:nbQ_reduced]
            qdot_pre_reduced = nlp_pre["X"][-1][nbQ_reduced:]
            q = nlp_post["q_mapping"].expand.map(q_reduced)
            qdot_pre = nlp_pre["q_dot_mapping"].expand.map(qdot_pre_reduced)

            if nlp_post["model"].nbContacts() == 0:
                warn("The chosen model does not have any contact")
            qdot_post = nlp_post["model"].ComputeConstraintImpulsesDirect(q, qdot_pre).to_mx()
            qdot_post_reduced = nlp_post["q_dot_mapping"].reduce.map(qdot_post)

            val = nlp_pre["X"][-1][:nbQ_reduced] - q_reduced
            val = vertcat(val, nlp_post["X"][0][nbQ_reduced:] - qdot_post_reduced)
            return val

        @staticmethod
        def custom(ocp, phase_pre_idx, **parameters):
            func = parameters["function"]
            del parameters["function"]
            del parameters["type"]
            del parameters["base"]
            nlp_pre, nlp_post = PhaseTransitionFunctions.Functions.__get_nlp_pre_and_post(ocp, phase_pre_idx)
            return func(nlp_pre["X"][-1], nlp_post["X"][0], **parameters)

        @staticmethod
        def __get_nlp_pre_and_post(ocp, phase_pre_idx):
            return ocp.nlp[phase_pre_idx], ocp.nlp[(phase_pre_idx + 1) % ocp.nb_phases]

    @staticmethod
    def prepare_phase_transitions(ocp, phase_transitions):
        # By default it assume Continuous. It can be change later
        full_phase_transitions = [
            {"type": PhaseTransition.CONTINUOUS, "phase_pre_idx": i, "base": ConstraintFunction}
            for i in range(ocp.nb_phases - 1)
        ]

        existing_phases = []
        for pt in phase_transitions:
            if "phase_pre_idx" not in pt and pt["type"] == PhaseTransition.CYCLIC:
                pt["phase_pre_idx"] = ocp.nb_phases - 1

            idx_phase = pt["phase_pre_idx"]
            if idx_phase in existing_phases:
                raise RuntimeError("It is not possible to define two phase continuity constraints for the same phase")
            if idx_phase >= ocp.nb_phases:
                raise RuntimeError("Phase index of the transition constraint is higher than the number of phases")
            existing_phases.append(idx_phase)

            pt["base"] = ConstraintFunction
            if "weight" in pt:
                if pt["weight"]:
                    pt["base"] = ObjectiveFunction.MayerFunction
                    pt["quadratic"] = True

            if idx_phase == ocp.nb_phases - 1:
                # Add a cyclic constraint or objective
                full_phase_transitions.append(pt)
            else:
                full_phase_transitions[idx_phase] = pt
        return full_phase_transitions


class ContinuityFunctions:
    @staticmethod
    def continuity(ocp):
        ConstraintFunction.continuity(ocp)  # Inner phase continuity
        PenaltyFunctionAbstract.continuity(ocp)  # Inter phase continuity


class PhaseTransition(Enum):
    """
    Different transitions between nlp phases.
    """

    CONTINUOUS = PhaseTransitionFunctions.Functions.continuous
    IMPACT = PhaseTransitionFunctions.Functions.impact
    CYCLIC = PhaseTransitionFunctions.Functions.cyclic
    CUSTOM = PhaseTransitionFunctions.Functions.custom
