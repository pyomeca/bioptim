from math import inf
from enum import Enum

from casadi import vertcat, Function, fabs

from .enums import Instant
from .penalty import PenaltyType, PenaltyFunctionAbstract
from .dynamics import Dynamics

# TODO: Convert the constraint in CasADi function?


class ConstraintFunction(PenaltyFunctionAbstract):
    """
    Different conditions between biorbd geometric structures.
    """

    class Functions:
        @staticmethod
        def contact_force_inequality(constraint_type, ocp, nlp, t, x, u, direction, contact_force_idx, boundary):
            """
            To be completed when this function will be fully developed, in particular the fact that policy is either a
            tuple/list or a tuple of tuples/list of lists,
            with in the 1st index the number of the contact force and in the 2nd index the associated bound.
            """
            # To be modified later so that it can handle something other than lower bounds for greater than
            CS_func = Function(
                "Contact_force_inequality",
                [ocp.symbolic_states, ocp.symbolic_controls],
                [nlp["contact_forces_func"](ocp.symbolic_states, ocp.symbolic_controls, nlp)],
                ["x", "u"],
                ["CS"],
            ).expand()

            for i in range(len(u)):
                ocp.g = vertcat(ocp.g, CS_func(x[i], u[i])[contact_force_idx])
                if direction == "GREATER_THAN":
                    ocp.g_bounds.min.append(boundary)
                    ocp.g_bounds.max.append(inf)
                elif direction == "LESSER_THAN":
                    ocp.g_bounds.min.append(-inf)
                    ocp.g_bounds.max.append(boundary)
                else:
                    raise RuntimeError(
                        "direction parameter of contact_force_inequality must either be GREATER_THAN or LESSER_THAN"
                    )

        @staticmethod
        def non_slipping(
            constraint_type,
            ocp,
            nlp,
            t,
            x,
            u,
            tangential_component_idx,
            normal_component_idx,
            static_friction_coefficient,
        ):
            """
            :param coeff: It is the coefficient of static friction.
            """
            if not isinstance(tangential_component_idx, int):
                raise RuntimeError("tangential_component_idx must be a unique integer")

            CS_func = Function(
                "Contact_force_inequality",
                [ocp.symbolic_states, ocp.symbolic_controls],
                [Dynamics.forces_from_forward_dynamics_with_contact(ocp.symbolic_states, ocp.symbolic_controls, nlp)],
                ["x", "u"],
                ["CS"],
            ).expand()

            if isinstance(normal_component_idx, int):
                normal_component_idx = [normal_component_idx]

            mu = static_friction_coefficient
            for i in range(len(u)):
                normal_contact_force = tangential_contact_force = 0
                for idx in normal_component_idx:
                    normal_contact_force += CS_func(x[i], u[i])[idx]
                tangential_contact_force += CS_func(x[i], u[i])[tangential_component_idx]

                # Proposal : only case normal_contact_force >= 0 and with two ocp.g
                ocp.g = vertcat(ocp.g, mu * fabs(normal_contact_force) - fabs(tangential_contact_force))
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(inf)

    @staticmethod
    def add(ocp, nlp):
        """
        Adds constraints to the requested nodes in (nlp.g) and (nlp.g_bounds).
        :param ocp: An OptimalControlProgram class.
        """
        PenaltyFunctionAbstract._add(ocp, nlp, "constraints")

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
                ConstraintFunction._add_to_penalty(ocp, None, val)

        # Dynamics must be continuous between phases
        for i in range(len(ocp.nlp) - 1):
            if ocp.nlp[i]["nx"] != ocp.nlp[i + 1]["nx"]:
                raise RuntimeError("Phase constraints without same nx is not supported yet")

            val = ocp.nlp[i]["X"][-1] - ocp.nlp[i + 1]["X"][0]
            ConstraintFunction._add_to_penalty(ocp, None, val)

        if ocp.is_cyclic_constraint:
            # Save continuity constraints between final integration and first node
            if ocp.nlp[0]["nx"] != ocp.nlp[-1]["nx"]:
                raise RuntimeError("Cyclic constraint without same nx is not supported yet")

            val = ocp.nlp[-1]["X"][-1][1:] - ocp.nlp[0]["X"][0][1:]
            ConstraintFunction._add_to_penalty(ocp, None, val)

    @staticmethod
    def _add_to_penalty(ocp, nlp, val, inf_bound=0, max_bound=0, **extra_param):
        ocp.g = vertcat(ocp.g, val)
        for _ in range(val.rows()):
            ocp.g_bounds.min.append(inf_bound)
            ocp.g_bounds.max.append(max_bound)

    @staticmethod
    def _parameter_modifier(constraint_function, parameters):
        # Everything that should change the entry parameters depending on the penalty can be added here
        super(ConstraintFunction, ConstraintFunction)._parameter_modifier(constraint_function, parameters)

    @staticmethod
    def _span_checker(constraint_function, instant, nlp):
        # Everything that is suspicious in terms of the span of the penalty function ca be checked here
        super(ConstraintFunction, ConstraintFunction)._span_checker(constraint_function, instant, nlp)
        if (
            constraint_function == Constraint.CONTACT_FORCE_INEQUALITY.value[0]
            or constraint_function == Constraint.NON_SLIPPING.value[0]
        ):
            if instant == Instant.END or instant == nlp["ns"]:
                raise RuntimeError("No control u at last node")


class Constraint(Enum):
    """
    Different conditions between biorbd geometric structures.
    """

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
    MINIMIZE_PREDICTED_COM_HEIGHT = (PenaltyType.MINIMIZE_PREDICTED_COM_HEIGHT,)
    ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
    ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
    CUSTOM = (PenaltyType.CUSTOM,)
    CONTACT_FORCE_INEQUALITY = (ConstraintFunction.Functions.contact_force_inequality,)
    NON_SLIPPING = (ConstraintFunction.Functions.non_slipping,)

    @staticmethod
    def _get_type():
        return ConstraintFunction
