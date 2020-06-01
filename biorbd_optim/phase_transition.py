from copy import deepcopy
from enum import Enum

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
            raise NotImplementedError("Custom transitions constraints are not implemented yet")

    @staticmethod
    def prepare_phase_transitions(ocp, phase_transitions):
        # By default it assume Continuous. It can be change later
        full_phase_transitions = [
            {"type": PhaseTransition.CONTINUOUS, "phase_pre_idx": i} for i in range(ocp.nb_phases - 1)
        ]

        existing_phases = []
        for pt in phase_transitions:
            idx_phase = pt["phase_pre_idx"]
            if idx_phase in existing_phases:
                raise RuntimeError("It is not possible to define two phase continuity constraints for the same phase")
            if idx_phase >= ocp.nb_phases:
                raise RuntimeError("Phase index of the transition constraint is higher than the number of phases")
            existing_phases.append(idx_phase)
            if idx_phase == ocp.nb_phases - 1:
                # Add a cyclic constraint
                full_phase_transitions.append(pt)
            else:
                full_phase_transitions[idx_phase] = pt
        return full_phase_transitions


class PhaseTransition(Enum):
    """
    Different transitions between nlp phases.
    """

    CONTINUOUS = PhaseTransitionFunctions.Functions.continuous
    IMPACT = PhaseTransitionFunctions.Functions.impact
    CUSTOM = PhaseTransitionFunctions.Functions.custom
