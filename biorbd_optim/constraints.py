import enum
from math import inf

import biorbd
from casadi import vertcat, horzcat, MX, Function, horzsplit

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
        for constraint in nlp["constraints"]:
            x, u, last_node = Constraint.__get_instant(nlp, constraint)
            type = constraint["type"]
            del constraint["instant"], constraint["type"]

            if type == Constraint.Type.MARKERS_TO_PAIR:
                Constraint.__markers_to_pair(ocp, nlp, x, **constraint)

            elif type == Constraint.Type.ALIGN_WITH_CUSTOM_RT:
                Constraint.__align_with_custom_rt(ocp, nlp, x, **constraint)

            elif type == Constraint.Type.PROPORTIONAL_Q:
                Constraint.__proportional_variable(ocp, nlp, x, **constraint)

            elif type == Constraint.Type.PROPORTIONAL_CONTROL:
                if last_node:
                    raise RuntimeError("No control u at last node")
                Constraint.__proportional_variable(ocp, nlp, u, **constraint)

            elif type == Constraint.Type.CONTACT_FORCE_GREATER_THAN:
                if last_node:
                    raise RuntimeError("No control u at last node")
                Constraint.__contact_force_inequality(ocp, nlp, x, u, "GREATER_THAN", **constraint)

            else:
                raise RuntimeError(constraint + "is not a valid constraint, take a look in Constraint.Type class")

    @staticmethod
    def __get_instant(nlp, constraint):
        if not isinstance(constraint["instant"], (list, tuple)):
            constraint["instant"] = (constraint["instant"],)
        x = MX()
        u = MX()
        last_node = False
        for node in constraint["instant"]:
            if isinstance(node, int):
                if node < 0 or node > nlp["ns"]:
                    raise RuntimeError("Invalid instant, " + str(node) + " must be between 0 and " + str(nlp["ns"]))
                if node == nlp["ns"]:
                    last_node = True
                x = horzcat(x, nlp["X"][node])
                u = horzcat(u, nlp["U"][node])

            elif node == Constraint.Instant.START:
                x = horzcat(x, nlp["X"][0])
                u = horzcat(u, nlp["U"][0])

            elif node == Constraint.Instant.MID:
                if nlp["ns"] % 2 == 0:
                    raise (ValueError("Number of shooting points must be odd to use MID"))
                x = horzcat(x, nlp["X"][nlp["ns"] // 2 + 1])
                u = horzcat(u, nlp["U"][nlp["ns"] // 2 + 1])

            elif node == Constraint.Instant.INTERMEDIATES:
                x = horzcat(x, nlp["X"][1 : nlp["ns"] - 1])
                u = horzcat(u, nlp["U"][1 : nlp["ns"] - 1])

            elif node == Constraint.Instant.END:
                x = horzcat(x, nlp["X"][nlp["ns"]])

            elif node == Constraint.Instant.ALL:
                for i in range(nlp["ns"]):
                    x = horzcat(x, nlp["X"][i])
                    u = horzcat(u, nlp["U"][i])
                x = horzcat(x, nlp["X"][nlp["ns"]])
                last_node = True
            else:
                raise RuntimeError(" is not a valid instant")
        return x, u, last_node

    @staticmethod
    def __markers_to_pair(ocp, nlp, X, first_marker, second_marker):
        """
        Adds the constraint that the two markers must be coincided at the desired instant(s).
        :param nlp: An OptimalControlProgram class.
        :param X: List of instant(s).
        :param policy: Tuple of indices of two markers.
        """
        Correct.parameters("marker", [first_marker, second_marker], nlp["model"].nbMarkers())

        nq = nlp["q_mapping"].reduce.len
        for x in horzsplit(X, 1):
            q = nlp["q_mapping"].expand.map(x[:nq])
            marker1 = nlp["model"].marker(q, first_marker).to_mx()
            marker2 = nlp["model"].marker(q, second_marker).to_mx()
            ocp.g = vertcat(ocp.g, marker1 - marker2)
            for i in range(3):
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(0)

    @staticmethod
    def __align_with_custom_rt(ocp, nlp, X, segment, rt):
        """
        Adds the constraint that the RT and the segment must be aligned at the desired instant(s).
        :param nlp: An OptimalControlProgram class.
        :param X: List of instant(s).
        :param policy: Tuple of indices of segment and rt.
        """
        Correct.parameters("segment", segment, nlp["model"].nbSegment())
        Correct.parameters("rt", rt, nlp["model"].nbRTs())

        nq = nlp["q_mapping"].nb_reduced
        for x in horzsplit(X, 1):
            q = nlp["q_mapping"].expand(x[:nq])
            r_seg = nlp["model"].globalJCS(q, segment).rot()
            r_rt = nlp["model"].RT(q, rt).rot()
            constraint = biorbd.Rotation_toEulerAngles(r_seg.transpose() * r_rt, "zyx").to_mx()
            ocp.g = vertcat(ocp.g, constraint)
            for i in range(constraint.rows()):
                ocp.g_bounds.min.append(0)
                ocp.g_bounds.max.append(0)

    @staticmethod
    def __proportional_variable(ocp, nlp, UX, first_dof, second_dof, coef):
        """
        Adds proportionality constraint between the elements (states or controls) chosen.
        :param nlp: An instance of the OptimalControlProgram class.
        :param V: List of states or controls at instants on which this constraint must be applied.
        :param policy: A tuple or a tuple of tuples (also works with lists) whose first two elements
        are the indexes of elements to be linked proportionally.
        The third element of each tuple (policy[i][2]) is the proportionality coefficient.
        """
        Correct.parameters("dof", (first_dof, second_dof), UX.rows())
        if not isinstance(coef, (int, float)):
            raise RuntimeError("coef must be a coeff")

        for v in horzsplit(UX, 1):
            v = nlp["q_mapping"].expand.map(v)
            ocp.g = vertcat(ocp.g, v[first_dof] - coef * v[second_dof])
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

        X, U = horzsplit(X, 1), horzsplit(U, 1)
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


class Correct:
    @staticmethod
    def parameters(name, elements, nb):
        if not isinstance(elements, (list, tuple)):
            elements = (elements,)
        for element in elements:
            if not isinstance(element, int):
                raise RuntimeError(str(element) + " is not a valid index for " + name + ", it must be a " + str(type))
            if element < 0 or element > nb:
                raise RuntimeError(
                    str(element) + " is not a valid index for " + name + ", it must be between 0 and " + str(nb - 1)
                )

    # TODO: same security in objective_function
