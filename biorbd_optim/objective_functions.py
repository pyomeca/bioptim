from enum import Enum

import casadi

from .penalty import PenaltyFunctionAbstract
from .enums import Instant


class ObjectiveFunction(PenaltyFunctionAbstract):
    class Lagrange(PenaltyFunctionAbstract, Enum):
        """
        Different conditions between biorbd geometric structures.
        """

        MINIMIZE_STATE = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_STATE.value
        TRACK_STATE = MINIMIZE_STATE
        MINIMIZE_MARKERS = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS.value
        TRACK_MARKERS = MINIMIZE_MARKERS
        MINIMIZE_MARKERS_DISPLACEMENT = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS_DISPLACEMENT.value
        MINIMIZE_MARKERS_VELOCITY = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS_VELOCITY.value
        TRACK_MARKERS_VELOCITY = MINIMIZE_MARKERS_VELOCITY
        ALIGN_MARKERS = "l_" + PenaltyFunctionAbstract.Type.ALIGN_MARKERS.value
        PROPORTIONAL_STATE = "l_" + PenaltyFunctionAbstract.Type.PROPORTIONAL_STATE.value
        PROPORTIONAL_CONTROL = "l_" + PenaltyFunctionAbstract.Type.PROPORTIONAL_CONTROL.value
        MINIMIZE_TORQUE = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_TORQUE.value
        TRACK_TORQUE = MINIMIZE_TORQUE
        MINIMIZE_MUSCLES = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_MUSCLES.value
        TRACK_MUSCLES = MINIMIZE_MUSCLES
        MINIMIZE_ALL_CONTROLS = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_ALL_CONTROLS.value
        TRACK_ALL_CONTROLS = MINIMIZE_ALL_CONTROLS
        MINIMIZE_PREDICTED_COM_HEIGHT = "l_" + PenaltyFunctionAbstract.Type.MINIMIZE_PREDICTED_COM_HEIGHT.value
        ALIGN_SEGMENT_WITH_CUSTOM_RT = "l_" + PenaltyFunctionAbstract.Type.ALIGN_SEGMENT_WITH_CUSTOM_RT.value
        ALIGN_MARKER_WITH_SEGMENT_AXIS = "l_" + PenaltyFunctionAbstract.Type.ALIGN_MARKER_WITH_SEGMENT_AXIS.value
        CUSTOM = "l_" + PenaltyFunctionAbstract.Type.CUSTOM.value

        @staticmethod
        def _add_to_goal(ocp, nlp, val, weight):
            ocp.J += (
                    casadi.dot(val, val)
                    * nlp["dt"]
                    * nlp["dt"]
                    * weight
            )

        @staticmethod
        def _get_type():
            return ObjectiveFunction.Lagrange

    class Mayer(PenaltyFunctionAbstract, Enum):
        """
        Different conditions between biorbd geometric structures.
        """

        MINIMIZE_STATE = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_STATE.value
        TRACK_STATE = MINIMIZE_STATE
        MINIMIZE_MARKERS = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS.value
        TRACK_MARKERS = MINIMIZE_MARKERS
        MINIMIZE_MARKERS_DISPLACEMENT = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS_DISPLACEMENT.value
        MINIMIZE_MARKERS_VELOCITY = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS_VELOCITY.value
        TRACK_MARKERS_VELOCITY = MINIMIZE_MARKERS_VELOCITY
        ALIGN_MARKERS = "m_" + PenaltyFunctionAbstract.Type.ALIGN_MARKERS.value
        PROPORTIONAL_STATE = "m_" + PenaltyFunctionAbstract.Type.PROPORTIONAL_STATE.value
        PROPORTIONAL_CONTROL = "m_" + PenaltyFunctionAbstract.Type.PROPORTIONAL_CONTROL.value
        MINIMIZE_TORQUE = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_TORQUE.value
        TRACK_TORQUE = MINIMIZE_TORQUE
        MINIMIZE_MUSCLES = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_MUSCLES.value
        TRACK_MUSCLES = MINIMIZE_MUSCLES
        MINIMIZE_ALL_CONTROLS = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_ALL_CONTROLS.value
        TRACK_ALL_CONTROLS = MINIMIZE_ALL_CONTROLS
        MINIMIZE_PREDICTED_COM_HEIGHT = "m_" + PenaltyFunctionAbstract.Type.MINIMIZE_PREDICTED_COM_HEIGHT.value
        ALIGN_SEGMENT_WITH_CUSTOM_RT = "m_" + PenaltyFunctionAbstract.Type.ALIGN_SEGMENT_WITH_CUSTOM_RT.value
        ALIGN_MARKER_WITH_SEGMENT_AXIS = "m_" + PenaltyFunctionAbstract.Type.ALIGN_MARKER_WITH_SEGMENT_AXIS.value
        CUSTOM = "m_" + PenaltyFunctionAbstract.Type.CUSTOM.value

        @staticmethod
        def _add_to_goal(ocp, nlp, val, weight):
            ocp.J += val * weight

        @staticmethod
        def _get_type():
            return ObjectiveFunction.Mayer

    @staticmethod
    def add(ocp, nlp):
        for objective in nlp["objective_functions"]:
            if objective["type"].value[:2] == "l_":
                if "instant" in objective.keys() and objective["instant"] != Instant.ALL:
                    raise RuntimeError("Lagrange objective are for Instant.ALL, did you mean Mayer?")
                objective["instant"] = Instant.ALL
            elif objective["type"].value[:2] == "m_":
                if "instant" not in objective.keys():
                    objective["instant"] = Instant.END
            else:
                raise RuntimeError("Objective function Type must be either a Lagrange or Mayer type")

        for _type, instant, t, x, u, objective in PenaltyFunctionAbstract._add(ocp, nlp, "objective_functions"):
            # Put here ObjectiveFunctions only
            raise RuntimeError(
                f"{_type} is not a valid objective function, take a look at ObjectiveFunction.Lagrange or ObjectiveFunction.Mayer")

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