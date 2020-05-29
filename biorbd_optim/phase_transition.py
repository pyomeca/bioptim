from enum import Enum

import biorbd
from casadi import vertcat


class PhaseTransitionFunctions:

    class Functions:
        @staticmethod
        def continuous(ocp, phase_before_idx):
            """
            TODO
            """
            return ocp.nlp[phase_before_idx]["X"][-1] - ocp.nlp[(phase_before_idx + 1) % ocp.nb_phases]["X"][0]

        @staticmethod
        def impact(ocp, phase_before_idx):
            """
            TODO
            """
            # Aliases
            nlp_pre = ocp.nlp[phase_before_idx]
            nlp_post = ocp.nlp[(phase_before_idx + 1) % ocp.nb_phases]
            nbQ = nlp_pre["nbQ"]
            q = nlp_post["X"][0][:nbQ]
            qdot_pre = nlp_pre["X"][-1][nbQ:]
            # qdot_post = nlp_post["model"].ComputeConstraintImpulsesDirect(q, qdot_pre)

            # As a temporary replacement for ComputeConstraintImpulsesDirect:
            qdot_post = nlp_post["X"][-1][nbQ:]

            val = nlp_pre["X"][-1][:nbQ] - q
            val = vertcat(val, nlp_post["X"][0][nbQ:] - qdot_post)
            return val

        @staticmethod
        def custom(ocp, phase_before_idx):
            pass


class PhaseTransition(Enum):
    """
    Different transitions between nlp phases.
    """

    CONTINUOUS = PhaseTransitionFunctions.Functions.continuous
    IMPACT = PhaseTransitionFunctions.Functions.impact
    CUSTOM = PhaseTransitionFunctions.Functions.custom
