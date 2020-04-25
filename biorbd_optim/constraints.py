from math import inf
from enum import Enum

from casadi import vertcat, horzsplit, Function, fabs

from .enums import Instant
from .penalty import PenaltyFunctionAbstract
from .dynamics import Dynamics

# TODO: Convert the constraint in CasADi function?


class Constraint(PenaltyFunctionAbstract, Enum):
    """
    Different conditions between biorbd geometric structures.
    """

    MINIMIZE_STATE = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_STATE.value
    TRACK_STATE = MINIMIZE_STATE
    MINIMIZE_MARKERS = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS.value
    TRACK_MARKERS = MINIMIZE_MARKERS
    MINIMIZE_MARKERS_DISPLACEMENT = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS_DISPLACEMENT.value
    MINIMIZE_MARKERS_VELOCITY = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_MARKERS_VELOCITY.value
    TRACK_MARKERS_VELOCITY = MINIMIZE_MARKERS_VELOCITY
    ALIGN_MARKERS = "c_" + PenaltyFunctionAbstract.Type.ALIGN_MARKERS.value
    PROPORTIONAL_STATE = "c_" + PenaltyFunctionAbstract.Type.PROPORTIONAL_STATE.value
    PROPORTIONAL_CONTROL = "c_" + PenaltyFunctionAbstract.Type.PROPORTIONAL_CONTROL.value
    MINIMIZE_TORQUE = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_TORQUE.value
    TRACK_TORQUE = MINIMIZE_TORQUE
    MINIMIZE_MUSCLES = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_MUSCLES.value
    TRACK_MUSCLES = MINIMIZE_MUSCLES
    MINIMIZE_ALL_CONTROLS = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_ALL_CONTROLS.value
    TRACK_ALL_CONTROLS = MINIMIZE_ALL_CONTROLS
    MINIMIZE_PREDICTED_COM_HEIGHT = "c_" + PenaltyFunctionAbstract.Type.MINIMIZE_PREDICTED_COM_HEIGHT.value
    CONTACT_FORCE_GREATER_THAN = "c_contact_force_greater_than"
    CONTACT_FORCE_LESSER_THAN = "c_contact_force_lesser_than"
    NON_SLIPPING = "c_non_slipping"
    ALIGN_SEGMENT_WITH_CUSTOM_RT = "c_" + PenaltyFunctionAbstract.Type.ALIGN_SEGMENT_WITH_CUSTOM_RT.value
    ALIGN_MARKER_WITH_SEGMENT_AXIS = "c_" + PenaltyFunctionAbstract.Type.ALIGN_MARKER_WITH_SEGMENT_AXIS.value
    CUSTOM = "c_" + PenaltyFunctionAbstract.Type.CUSTOM.value

    @staticmethod
    def add(ocp, nlp):
        """
        Adds constraints to the requested nodes in (nlp.g) and (nlp.g_bounds).
        :param ocp: An OptimalControlProgram class.
        """
        for _type, instant, t, x, u, constraint in PenaltyFunctionAbstract._add(ocp, nlp, "constraints"):
            # Put here ConstraintsFunction only
            if _type == Constraint.CONTACT_FORCE_GREATER_THAN:
                if instant == Instant.END or instant == nlp["ns"]:
                    raise RuntimeError("No control u at last node")
                Constraint.__contact_force_inequality(ocp, nlp, x, u, "GREATER_THAN", **constraint)

            elif _type == Constraint.CONTACT_FORCE_LESSER_THAN:
                if instant == Instant.END or instant == nlp["ns"]:
                    raise RuntimeError("No control u at last node")
                Constraint.__contact_force_inequality(ocp, nlp, x, u, "LESSER_THAN", **constraint)

            elif _type == Constraint.NON_SLIPPING:
                if instant == Instant.END or instant == nlp["ns"]:
                    raise RuntimeError("No control u at last node")
                Constraint.__non_slipping(ocp, nlp, x, u, **constraint)

            else:
                raise RuntimeError(f"{_type} is not a valid constraint, take a look in Constraint.Type class")

    @staticmethod
    def __contact_force_inequality(ocp, nlp, X, U, _type, idx, boundary):
        """
        To be completed when this function will be fully developed, in particular the fact that policy is either a tuple/list or a tuple of tuples/list of lists,
        with in the 1st index the number of the contact force and in the 2nd index the associated bound.
        """
        # To be modified later so that it can handle something other than lower bounds for greater than
        CS_func = Function(
            "Contact_force_inequality",
            [ocp.symbolic_states, ocp.symbolic_controls],
            [Dynamics.forces_from_forward_dynamics_with_contact(ocp.symbolic_states, ocp.symbolic_controls, nlp)],
            ["x", "u"],
            ["CS"],
        ).expand()

        X, U = horzsplit(X, 1), horzsplit(U, 1)
        for i in range(len(U)):
            ocp.g = vertcat(ocp.g, CS_func(X[i], U[i])[idx])
            if _type == "GREATER_THAN":
                ocp.g_bounds.min.append(boundary)
                ocp.g_bounds.max.append(inf)
            elif _type == "LESSER_THAN":
                ocp.g_bounds.min.append(-inf)
                ocp.g_bounds.max.append(boundary)

    @staticmethod
    def __non_slipping(ocp, nlp, x, u, normal_component_idx, tangential_component_idx,
                       static_friction_coefficient):
        """
        :param coeff: It is the coefficient of static friction.
        """
        CS_func = Function(
            "Contact_force_inequality",
            [ocp.symbolic_states, ocp.symbolic_controls],
            [Dynamics.forces_from_forward_dynamics_with_contact(ocp.symbolic_states, ocp.symbolic_controls, nlp)],
            ["x", "u"],
            ["CS"],
        ).expand()

        mu = static_friction_coefficient
        X, U = horzsplit(x, 1), horzsplit(u, 1)
        for i in range(len(U)):
            normal_contact_force = tangential_contact_force = 0
            for idx in normal_component_idx:
                normal_contact_force += CS_func(X[i], U[i])[idx]
            for idx in tangential_component_idx:
                normal_contact_force += CS_func(X[i], U[i])[idx]

            # Proposal : only case normal_contact_force >= 0 and with two ocp.g
            ocp.g = vertcat(ocp.g, mu * fabs(normal_contact_force) + fabs(tangential_contact_force))
            ocp.g_bounds.min.append(0)
            ocp.g_bounds.max.append(inf)

    @staticmethod
    def continuity_constraint(ocp):
        """
        Adds continuity constraints between each nodes and its neighbours. It is possible to add a continuity
        constraint between first and last nodes to have a loop (nlp.is_cyclic_constraint).
        :param ocp: An OptimalControlProgram class.
        """
        # Dynamics must be sound within phases
        for nlp in ocp.nlp:
            # Loop over shooting nodes
            for k in range(nlp["ns"]):
                # Create an evaluation node
                end_node = nlp["dynamics"].call({"x0": nlp["X"][k], "p": nlp["U"][k]})["xf"]

                # Save continuity constraints
                val = end_node - nlp["X"][k + 1]
                Constraint._add_to_goal(ocp, None, val, None)

        # Dynamics must be continuous between phases
        for i in range(len(ocp.nlp) - 1):
            if ocp.nlp[i]["nx"] != ocp.nlp[i + 1]["nx"]:
                raise RuntimeError("Phase constraints without same nx is not supported yet")

            val = ocp.nlp[i]["X"][-1] - ocp.nlp[i + 1]["X"][0]
            Constraint._add_to_goal(ocp, None, val, None)

        if ocp.is_cyclic_constraint:
            # Save continuity constraints between final integration and first node
            if ocp.nlp[0]["nx"] != ocp.nlp[-1]["nx"]:
                raise RuntimeError("Cyclic constraint without same nx is not supported yet")

            val = ocp.nlp[-1]["X"][-1][1:] - ocp.nlp[0]["X"][0][1:]
            Constraint._add_to_goal(ocp, None, val, None)

    @staticmethod
    def _get_type():
        return Constraint

    @staticmethod
    def _add_to_goal(ocp, nlp, val, weight):
        ocp.g = vertcat(ocp.g, val)
        for _ in range(val.rows()):
            ocp.g_bounds.min.append(0)
            ocp.g_bounds.max.append(0)
