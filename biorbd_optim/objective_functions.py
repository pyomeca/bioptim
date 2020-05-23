from enum import Enum

import casadi

from .penalty import PenaltyType, PenaltyFunctionAbstract
from .enums import Instant


class ObjectiveFunction:
    class LagrangeFunction(PenaltyFunctionAbstract):
        """
        Different conditions between biorbd geometric structures.
        """

        class Functions:
            @staticmethod
            def minimize_time(penalty_type, ocp, nlp, t, x, u, **extra_param):
                val = 1
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def _add_to_penalty(ocp, nlp, val, penalty_idx, weight=1, quadratic=False, **extra_param):
            if quadratic:
                J = casadi.dot(val, val) * weight * nlp["dt"]
            else:
                J = casadi.sum1(val) * weight * nlp["dt"]
            ObjectiveFunction._add_to_penalty(ocp, nlp, J, penalty_idx)

        @staticmethod
        def _reset_penalty(ocp, nlp, penalty_idx):
            return ObjectiveFunction._reset_penalty(ocp, nlp, penalty_idx)

        @staticmethod
        def _parameter_modifier(penalty_function, parameters):
            # Everything that should change the entry parameters depending on the penalty can be added here
            PenaltyFunctionAbstract._parameter_modifier(penalty_function, parameters)
            if penalty_function == Objective.Lagrange.MINIMIZE_TIME.value[0]:
                if "quadratic" not in parameters.keys():
                    parameters["quadratic"] = True

        @staticmethod
        def _span_checker(penalty_function, instant, nlp):
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(penalty_function, instant, nlp)

    class MayerFunction(PenaltyFunctionAbstract):
        """
        Different conditions between biorbd geometric structures.
        """

        class Functions:
            @staticmethod
            def minimize_time(penalty_type, ocp, nlp, t, x, u, **extra_param):
                val = nlp["tf"]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def _add_to_penalty(ocp, nlp, val, penalty_idx, weight=1, quadratic=False, **parameters):
            if quadratic:
                J = casadi.dot(val, val) * weight
            else:
                J = casadi.sum1(val) * weight
            ObjectiveFunction._add_to_penalty(ocp, nlp, J, penalty_idx)

        @staticmethod
        def _reset_penalty(ocp, nlp, penalty_idx):
            return ObjectiveFunction._reset_penalty(ocp, nlp, penalty_idx)

        @staticmethod
        def _parameter_modifier(penalty_function, parameters):
            # Everything that should change the entry parameters depending on the penalty can be added here
            PenaltyFunctionAbstract._parameter_modifier(penalty_function, parameters)

        @staticmethod
        def _span_checker(penalty_function, instant, nlp):
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(penalty_function, instant, nlp)

    @staticmethod
    def add_or_replace(ocp, nlp, objective, penalty_idx):
        if objective["type"]._get_type() == ObjectiveFunction.LagrangeFunction:
            if "instant" in objective.keys() and objective["instant"] != Instant.ALL:
                raise RuntimeError("Lagrange objective are for Instant.ALL, did you mean Mayer?")
            objective["instant"] = Instant.ALL
        elif objective["type"]._get_type() == ObjectiveFunction.MayerFunction:
            if "instant" not in objective.keys():
                objective["instant"] = Instant.END
        else:
            raise RuntimeError("Objective function Type must be either a Lagrange or Mayer type")
        PenaltyFunctionAbstract.add_or_replace(ocp, nlp, objective, penalty_idx)
    #
    # @staticmethod
    # def cyclic(ocp, weight=1):
    #
    #     if ocp.nlp[0]["nx"] != ocp.nlp[-1]["nx"]:
    #         raise RuntimeError("Cyclic constraint without same nx is not supported yet")
    #
    #     ocp.J += (
    #         casadi.dot(ocp.nlp[-1]["X"][-1] - ocp.nlp[0]["X"][0], ocp.nlp[-1]["X"][-1] - ocp.nlp[0]["X"][0]) * weight
    #     )

    @staticmethod
    def _add_to_penalty(ocp, nlp, J, penalty_idx):
        if nlp:
            nlp["J"][penalty_idx].append(J)
        else:
            ocp.J[penalty_idx].append(J)

    @staticmethod
    def _reset_penalty(ocp, nlp, penalty_idx):
        if nlp:
            J_to_add_to = nlp["J"]
        else:
            J_to_add_to = ocp.J

        if penalty_idx < 0:
            J_to_add_to.append([])
            return len(J_to_add_to) - 1
        else:
            J_to_add_to[penalty_idx] = []
            return penalty_idx


class Objective:
    class Lagrange(Enum):
        """
        Different conditions between biorbd geometric structures.
        """

        MINIMIZE_TIME = (ObjectiveFunction.LagrangeFunction.Functions.minimize_time,)
        MINIMIZE_STATE = (PenaltyType.MINIMIZE_STATE,)
        TRACK_STATE = (PenaltyType.TRACK_STATE,)
        MINIMIZE_MARKERS = (PenaltyType.MINIMIZE_MARKERS,)
        TRACK_MARKERS = (PenaltyType.TRACK_MARKERS,)
        MINIMIZE_MARKERS_DISPLACEMENT = (PenaltyType.MINIMIZE_MARKERS_DISPLACEMENT,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyType.MINIMIZE_MARKERS_VELOCITY,)
        TRACK_MARKERS_VELOCITY = (PenaltyType.TRACK_MARKERS_VELOCITY,)
        ALIGN_MARKERS = (PenaltyType.ALIGN_MARKERS,)
        PROPORTIONAL_STATE = (PenaltyType.PROPORTIONAL_STATE,)
        PROPORTIONAL_CONTROL = (PenaltyType.PROPORTIONAL_CONTROL,)
        MINIMIZE_TORQUE = (PenaltyType.MINIMIZE_TORQUE,)
        TRACK_TORQUE = (PenaltyType.TRACK_TORQUE,)
        MINIMIZE_MUSCLES_CONTROL = (PenaltyType.MINIMIZE_MUSCLES_CONTROL,)
        TRACK_MUSCLES_CONTROL = (PenaltyType.TRACK_MUSCLES_CONTROL,)
        MINIMIZE_ALL_CONTROLS = (PenaltyType.MINIMIZE_ALL_CONTROLS,)
        TRACK_ALL_CONTROLS = (PenaltyType.TRACK_ALL_CONTROLS,)
        MINIMIZE_CONTACT_FORCES = (PenaltyType.MINIMIZE_CONTACT_FORCES,)
        TRACK_CONTACT_FORCES = (PenaltyType.TRACK_CONTACT_FORCES,)
        ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
        ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
        CUSTOM = (PenaltyType.CUSTOM,)

        @staticmethod
        def _get_type():
            return ObjectiveFunction.LagrangeFunction

    class Mayer(Enum):
        """
        Different conditions between biorbd geometric structures.
        """

        MINIMIZE_TIME = (ObjectiveFunction.MayerFunction.Functions.minimize_time,)
        MINIMIZE_STATE = (PenaltyType.MINIMIZE_STATE,)
        TRACK_STATE = (PenaltyType.TRACK_STATE,)
        MINIMIZE_MARKERS = (PenaltyType.MINIMIZE_MARKERS,)
        TRACK_MARKERS = (PenaltyType.TRACK_MARKERS,)
        MINIMIZE_MARKERS_DISPLACEMENT = (PenaltyType.MINIMIZE_MARKERS_DISPLACEMENT,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyType.MINIMIZE_MARKERS_VELOCITY,)
        TRACK_MARKERS_VELOCITY = (PenaltyType.TRACK_MARKERS_VELOCITY,)
        ALIGN_MARKERS = (PenaltyType.ALIGN_MARKERS,)
        PROPORTIONAL_STATE = (PenaltyType.PROPORTIONAL_STATE,)
        PROPORTIONAL_CONTROL = (PenaltyType.PROPORTIONAL_CONTROL,)
        MINIMIZE_TORQUE = (PenaltyType.MINIMIZE_TORQUE,)
        TRACK_TORQUE = (PenaltyType.TRACK_TORQUE,)
        MINIMIZE_MUSCLES_CONTROL = (PenaltyType.MINIMIZE_MUSCLES_CONTROL,)
        TRACK_MUSCLES_CONTROL = (PenaltyType.TRACK_MUSCLES_CONTROL,)
        MINIMIZE_ALL_CONTROLS = (PenaltyType.MINIMIZE_ALL_CONTROLS,)
        TRACK_ALL_CONTROLS = (PenaltyType.TRACK_ALL_CONTROLS,)
        MINIMIZE_CONTACT_FORCES = (PenaltyType.MINIMIZE_CONTACT_FORCES,)
        TRACK_CONTACT_FORCES = (PenaltyType.TRACK_CONTACT_FORCES,)
        MINIMIZE_PREDICTED_COM_HEIGHT = (PenaltyType.MINIMIZE_PREDICTED_COM_HEIGHT,)
        ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
        ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
        CUSTOM = (PenaltyType.CUSTOM,)

        @staticmethod
        def _get_type():
            return ObjectiveFunction.MayerFunction
