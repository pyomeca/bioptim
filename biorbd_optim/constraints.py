import enum
import numpy as np
from math import inf

import biorbd


from casadi import vertcat, MX, Function

from .dynamics import Dynamics


# TODO: Convert the constraint in CasADi function?


class Constraint:
    @staticmethod
    class Type(enum.Enum):
        """
        Different conditions between biorbd geometric structures.
        """

        MARKERS_TO_PAIR = 0
        ALIGN_WITH_CUSTOM_RT = 1
        PROPORTIONAL_Q = 2
        PROPORTIONAL_CONTROL = 3
        CONTACT_FORCE_GREATER_THAN = 4
        # TODO: PAUL = Add lesser than
        # TODO: PAUL = Add frictional cone

    @staticmethod
    class Instant(enum.Enum):
        """
        Five groups of nodes.
        START: first node only.
        MID: middle node only.
        INTERMEDIATES: all nodes except first and last.
        END: last node only.
        ALL: obvious.
        """

        START = 0
        MID = 1
        INTERMEDIATES = 2
        END = 3
        ALL = 4

    @staticmethod
    def add_constraints(ocp, nlp):
        """
        Adds constraints to the requested nodes in (nlp.g) and (nlp.g_bounds).
        :param ocp: An OptimalControlProgram class.
        """
        if nlp["constraints"] is None:
            return
        for elem in nlp["constraints"]:
            if elem[1] == Constraint.Instant.START:
                x = [nlp["X"][0]]
                u = [nlp["U"][0]]
            elif elem[1] == Constraint.Instant.MID:
                if nlp["ns"] % 2 == 0:
                    raise (ValueError("Number of shooting points must be odd to use MID"))
                x = [nlp["X"][nlp["ns"] // 2 + 1]]
                u = [nlp["U"][nlp["ns"] // 2 + 1]]
            elif elem[1] == Constraint.Instant.INTERMEDIATES:
                x = nlp["X"][1 : nlp["ns"] - 1]
                u = nlp["U"][1 : nlp["ns"] - 1]
            elif elem[1] == Constraint.Instant.END:
                x = [nlp["X"][nlp["ns"]]]
                u = []
            elif elem[1] == Constraint.Instant.ALL:
                x = nlp["X"]
                u = nlp["U"]
            else:
                continue

            if elem[0] == Constraint.Type.MARKERS_TO_PAIR:
                Constraint.__markers_to_pair(ocp, nlp, x, elem[2])

            elif elem[0] == Constraint.Type.ALIGN_WITH_CUSTOM_RT:
                Constraint.__align_with_custom_rt(ocp, nlp, x, elem[2])

            elif elem[0] == Constraint.Type.PROPORTIONAL_Q:
                Constraint.__proportional_variable(ocp, nlp, x, elem[2])

            elif elem[0] == Constraint.Type.PROPORTIONAL_CONTROL:
                if elem[1] == Constraint.Instant.END:
                    raise RuntimeError("Instant.END is used even though there is no control u at last node")
                Constraint.__proportional_variable(ocp, nlp, u, elem[2])

            elif elem[0] == Constraint.Type.CONTACT_FORCE_GREATER_THAN:
                if elem[1] == Constraint.Instant.END:
                    raise RuntimeError("Instant.END is used even though there is no control u at last node")
                Constraint.__contact_force_inequality(ocp, nlp, x, u, elem[2], "GREATER_THAN")

    @staticmethod
    def __markers_to_pair(ocp, nlp, X, policy):
        """
        Adds the constraint that the two markers must be coincided at the desired instant(s).
        :param nlp: An OptimalControlProgram class.
        :param X: List of instant(s).
        :param policy: Tuple of indices of two markers.
        """
        nq = nlp["q_mapping"].nb_reduced
        for x in X:
            q = nlp["q_mapping"].expand(x[:nq])
            marker1 = nlp["model"].marker(q, policy[0]).to_mx()
            marker2 = nlp["model"].marker(q, policy[1]).to_mx()
            ocp.g = vertcat(ocp.g, marker1 - marker2)
            for i in range(3):
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(0)

    @staticmethod
    def __align_with_custom_rt(ocp, nlp, X, policy):
        """
        Adds the constraint that the RT and the segment must be aligned at the desired instant(s).
        :param nlp: An OptimalControlProgram class.
        :param X: List of instant(s).
        :param policy: Tuple of indices of segment and rt.
        """
        nq = nlp["dof_mapping"].nb_reduced
        for x in X:
            q = nlp["dof_mapping"].expand(x[:nq])
            r_seg = np.array(nlp["model"].globalJCS(q, policy[0]).rot())
            r_rt = np.array(nlp["model"].RT(q, policy[1]).rot())
            constraint = biorbd.Rotation_toEulerAngles(r_seg.T * r_rt, "zyx")
            ocp.g = vertcat(ocp.g, constraint)
            for i in range(constraint.row()):
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(0)

    @staticmethod
    def __proportional_variable(ocp, nlp, V, policy):
        """
        Adds proportionality constraint between the elements (states or controls) chosen.
        :param nlp: An instance of the OptimalControlProgram class.
        :param V: List of states or controls at instants on which this constraint must be applied.
        :param policy: A tuple or a tuple of tuples (also works with lists) whose first two elements
        are the indexes of elements to be linked proportionally.
        The third element of each tuple (policy[i][2]) is the proportionality coefficient.
        """
        if not isinstance(policy[0], (tuple, list)):
            policy = [policy]
        for elem in policy:
            if not isinstance(elem, (tuple, list)):
                raise RuntimeError(
                    "A mix of tuples/lists and non tuples/lists cannot be used for defining proportionality constraints"
                )
            for v in V:
                v = nlp["q_mapping"].expand(v)
                ocp.g = vertcat(ocp.g, v[elem[0]] - elem[2] * v[elem[1]])
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(0)

    @staticmethod
    def __contact_force_inequality(ocp, nlp, X, U, policy, type):
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

        if not isinstance(policy[0], (tuple, list)):
            policy = [policy]

        for i in range(len(U)):
            contact_forces = CS_func(X[i], U[i])
            contact_forces = contact_forces[: nlp["model"].nbContacts()]

            for elem in policy:
                ocp.g = vertcat(ocp.g, contact_forces[elem[0]])
                if type == "GREATER_THAN":
                    ocp.g_bounds.min.append(elem[1])
                    ocp.g_bounds.max.append(inf)
                elif type == "LESSER_THAN":
                    ocp.g_bounds.min.append(-inf)
                    ocp.g_bounds.max.append(elem[1])

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
                ocp.g = vertcat(ocp.g, end_node - nlp["X"][k + 1])
                for _ in range(nlp["nx"]):
                    ocp.g_bounds.min.append(0)
                    ocp.g_bounds.max.append(0)

        # Dynamics must be continuous between phases
        for i in range(len(ocp.nlp) - 1):
            if ocp.nlp[i]["nx"] != ocp.nlp[i + 1]["nx"]:
                raise RuntimeError("Phase constraints without same nx is not supported yet")

            ocp.g = vertcat(ocp.g, ocp.nlp[i]["X"][-1] - ocp.nlp[i + 1]["X"][0])
            for _ in range(ocp.nlp[i]["nx"]):
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(0)

        if ocp.is_cyclic_constraint:
            # Save continuity constraints between final integration and first node
            if ocp.nlp[0]["nx"] != ocp.nlp[-1]["nx"]:
                raise RuntimeError("Cyclic constraint without same nx is not supported yet")
            ocp.g = vertcat(ocp.g, ocp.nlp[-1]["X"][-1] - ocp.nlp[0]["X"][0])
            for i in range(ocp.nlp[0]["nx"]):
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(0)
