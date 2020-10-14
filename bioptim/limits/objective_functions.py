from enum import Enum

import numpy as np
import casadi
from casadi import dot, sum1

from .penalty import PenaltyType, PenaltyFunctionAbstract
from ..misc.enums import Instant
from ..misc.options_lists import OptionList, OptionGeneric


class ObjectiveOption(OptionGeneric):
    def __init__(
        self, objective, instant=Instant.DEFAULT, quadratic=None, weight=1, custom_type=None, phase=0, **params
    ):
        custom_function = None
        if not isinstance(objective, Objective.Lagrange) and not isinstance(objective, Objective.Mayer):
            custom_function = objective

            if custom_type is None:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is missing. "
                    "It should either be Objective.Mayer or Objective.Lagrange"
                )
            objective = custom_type(custom_type.CUSTOM)
            if isinstance(objective, Objective.Lagrange):
                pass
            elif isinstance(objective, Objective.Mayer):
                pass
            elif isinstance(objective, Objective.Parameter):
                pass
            else:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is invalid. "
                    "It should either be Objective.Mayer or Objective.Lagrange"
                )

        super(ObjectiveOption, self).__init__(type=objective, phase=phase, **params)
        self.instant = instant
        self.quadratic = quadratic
        self.weight = weight
        self.custom_function = custom_function


class ObjectiveList(OptionList):
    def add(self, objective, **extra_arguments):
        if isinstance(objective, ObjectiveOption):
            self.copy(objective)
        else:
            super(ObjectiveList, self)._add(objective=objective, option_type=ObjectiveOption, **extra_arguments)


class ObjectiveFunction:
    """
    Different conditions between biorbd geometric structures.
    """

    class LagrangeFunction(PenaltyFunctionAbstract):
        """
        Lagrange type objectives. (integral of the objective over the optimized movement duration)
        """

        class Functions:
            """
            Biomechanical objectives
            """

            @staticmethod
            def minimize_time(penalty, ocp, nlp, t, x, u, p, **extra_param):
                """Minimizes the duration of the movement (Lagrange)."""
                val = 1
                ObjectiveFunction.LagrangeFunction.add_to_penalty(ocp, nlp, val, penalty, **extra_param)

        @staticmethod
        def add_to_penalty(ocp, nlp, val, penalty, target=None, **extra_arguments):
            """
            Adds an objective.
            :param val: Value to be optimized. (MX.sym from CasADi)
            :param penalty: Index of the objective. (integer)
            :param weight: Weight of the objective. (float)
            :param quadratic: If True, value is squared. (bool)
            """
            ObjectiveFunction.add_to_penalty(ocp, nlp, val, penalty, dt=nlp.dt, target=target)

        @staticmethod
        def clear_penalty(ocp, nlp, penalty):
            """
            Resets specified penalty.
            """
            return ObjectiveFunction.clear_penalty(ocp, nlp, penalty)

        @staticmethod
        def _parameter_modifier(penalty_function, parameters):
            """Modification of parameters"""
            # Everything that should change the entry parameters depending on the penalty can be added here
            if penalty_function == Objective.Lagrange.MINIMIZE_TIME.value[0]:
                if not parameters.quadratic:
                    parameters.quadratic = True
            PenaltyFunctionAbstract._parameter_modifier(penalty_function, parameters)

        @staticmethod
        def _span_checker(penalty_function, instant, nlp):
            """Raises errors on the span of penalty functions"""
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(penalty_function, instant, nlp)

    class MayerFunction(PenaltyFunctionAbstract):
        """
        Mayer type objectives. (value of the objective at one time point, usually the end)
        """

        class Functions:
            """
            Biomechanical objectives
            """

            @staticmethod
            def minimize_time(penalty, ocp, nlp, t, x, u, p, **extra_param):
                """Minimizes the duration of the movement (Mayer)."""
                val = nlp.tf
                ObjectiveFunction.MayerFunction.add_to_penalty(ocp, nlp, val, penalty, **extra_param)

        @staticmethod
        def inter_phase_continuity(ocp, pt):
            # Dynamics must be respected between phases
            penalty = OptionGeneric()
            penalty.idx = -1
            penalty.quadratic = pt.quadratic
            penalty.weight = pt.weight
            pt.base.clear_penalty(ocp, None, penalty)
            val = pt.type.value[0](ocp, pt)
            pt.base.add_to_penalty(ocp, None, val, penalty, **pt.params)

        @staticmethod
        def add_to_penalty(ocp, nlp, val, penalty, target=None, **extra_param):
            """
            Adds an objective.
            :param val: Value to be optimized. (MX.sym from CasADi)
            :param penalty: Index of the objective. (integer)
            :param weight: Weight of the objective. (float)
            :param quadratic: If True, value is squared (bool)
            """
            ObjectiveFunction.add_to_penalty(ocp, nlp, val, penalty, dt=1, target=target)

        @staticmethod
        def clear_penalty(ocp, nlp, penalty_idx):
            """
            Resets specified penalty.
            """
            return ObjectiveFunction.clear_penalty(ocp, nlp, penalty_idx)

        @staticmethod
        def _parameter_modifier(penalty_function, parameters):
            """Modification of parameters"""
            # Everything that should change the entry parameters depending on the penalty can be added here
            PenaltyFunctionAbstract._parameter_modifier(penalty_function, parameters)

        @staticmethod
        def _span_checker(penalty_function, instant, nlp):
            """Raises errors on the span of penalty functions"""
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(penalty_function, instant, nlp)

    class ParameterFunction(PenaltyFunctionAbstract):
        """
        Mayer type objectives. (value of the objective at one time point, usually the end)
        """

        class Functions:
            """
            Biomechanical objectives
            """

            pass

        @staticmethod
        def add_to_penalty(ocp, _, val, penalty, target=None, **extra_param):
            """
            Adds an objective.
            :param val: Value to be optimized. (MX.sym from CasADi)
            :param penalty: Index of the objective. (integer)
            :param weight: Weight of the objective. (float)
            :param quadratic: If True, value is squared (bool)
            """
            ObjectiveFunction.add_to_penalty(ocp, None, val, penalty, dt=1, target=target)

        @staticmethod
        def clear_penalty(ocp, _, penalty_idx):
            """
            Resets specified penalty.
            """
            return ObjectiveFunction.clear_penalty(ocp, None, penalty_idx)

        @staticmethod
        def _parameter_modifier(penalty_function, parameters):
            """Modification of parameters"""
            # Everything that should change the entry parameters depending on the penalty can be added here
            PenaltyFunctionAbstract._parameter_modifier(penalty_function, parameters)

        @staticmethod
        def _span_checker(penalty_function, instant, nlp):
            """Raises errors on the span of penalty functions"""
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(penalty_function, instant, nlp)

    @staticmethod
    def add_or_replace(ocp, nlp, objective):
        """
        Modifies or raises errors if user provided Instant does not match the objective type.
        :param objective: New objective to replace with. (dictionary)
        """
        if objective.type.get_type() == ObjectiveFunction.LagrangeFunction:
            if objective.instant != Instant.ALL and objective.instant != Instant.DEFAULT:
                raise RuntimeError("Lagrange objective are for Instant.ALL, did you mean Mayer?")
            objective.instant = Instant.ALL
        elif objective.type.get_type() == ObjectiveFunction.MayerFunction:
            if objective.instant == Instant.DEFAULT:
                objective.instant = Instant.END

        else:
            raise RuntimeError("Objective function Type must be either a Lagrange or Mayer type")
        PenaltyFunctionAbstract.add_or_replace(ocp, nlp, objective)

    @staticmethod
    def cyclic(ocp, weight=1):

        if ocp.nlp[0].nx != ocp.nlp[-1].nx:
            raise RuntimeError("Cyclic constraint without same nx is not supported yet")

        ocp.J += casadi.dot(ocp.nlp[-1].X[-1] - ocp.nlp[0].X[0], ocp.nlp[-1].X[-1] - ocp.nlp[0].X[0]) * weight

    @staticmethod
    def add_to_penalty(ocp, nlp, val, penalty, dt=0, target=None):
        """
        Adds objective J to objective array nlp.J[penalty] or ocp.J[penalty] at index penalty.
        :param J: Objective. (dict of [val, target, weight, is_quadratic])
        :param penalty: Index of the objective. (integer)
        """
        J = {"objective": penalty, "val": val, "target": target, "dt": dt}

        if nlp:
            nlp.J[penalty.idx].append(J)
        else:
            ocp.J[penalty.idx].append(J)

    @staticmethod
    def clear_penalty(ocp, nlp, penalty):
        """
        Resets specified objective.
        Negative penalty index leads to enlargement of the array by one empty space.
        """
        if nlp:
            J_to_add_to = nlp.J
        else:
            J_to_add_to = ocp.J

        if penalty.idx < 0:
            # Add a new one
            for i, j in enumerate(J_to_add_to):
                if not j:
                    penalty.idx = i
                    return
            else:
                J_to_add_to.append([])
                penalty.idx = len(J_to_add_to) - 1
        else:
            while penalty.idx >= len(J_to_add_to):
                J_to_add_to.append([])
            J_to_add_to[penalty.idx] = []


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
        MINIMIZE_TORQUE_DERIVATIVE = (PenaltyType.MINIMIZE_TORQUE_DERIVATIVE,)
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
        def get_type():
            """Returns the type of the objective function"""
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
        def get_type():
            """Returns the type of the objective function"""
            return ObjectiveFunction.MayerFunction

    class Parameter(Enum):
        CUSTOM = (PenaltyType.CUSTOM,)

    class Printer:
        def __init__(self, ocp, sol_obj):
            self.ocp = ocp
            self.sol_obj = sol_obj

        def by_function(self):
            for idx_phase, phase in enumerate(self.sol_obj):
                print(f"********** Phase {idx_phase} **********")
                for idx_obj in range(phase.shape[0]):
                    print(
                        f"{self.ocp.original_values['objective_functions'][idx_phase][idx_phase + idx_obj].type.name} : {np.nansum(phase[idx_obj])}"
                    )

        def by_nodes(self):
            for idx_phase, phase in enumerate(self.sol_obj):
                print(f"********** Phase {idx_phase} **********")
                for idx_node in range(phase.shape[1]):
                    print(f"Node {idx_node} : {np.nansum(phase[:, idx_node])}")

        def mean(self):
            m = 0
            for idx_phase, phase in enumerate(self.sol_obj):
                m += np.nansum(phase)
            return m / len(self.sol_obj)


def get_objective_values(ocp, sol):
    def __get_instant(instants, nlp):
        nodes = []
        for node in instants:
            if isinstance(node, int):
                if node < 0 or node > nlp.ns:
                    raise RuntimeError(f"Invalid instant, {node} must be between 0 and {nlp.ns}")
                nodes.append(node)

            elif node == Instant.START:
                nodes.append(0)

            elif node == Instant.MID:
                if nlp.ns % 2 == 1:
                    raise (ValueError("Number of shooting points must be even to use MID"))
                nodes.append(nlp.ns // 2)

            elif node == Instant.INTERMEDIATES:
                for i in range(1, nlp.ns - 1):
                    nodes.append(i)

            elif node == Instant.END:
                nodes.append(nlp.ns - 1)

            elif node == Instant.ALL:
                for i in range(nlp.ns):
                    nodes.append(i)
        return nodes

    sol = sol["x"]
    out = []
    for nlp in ocp.nlp:
        nJ = len(nlp.J)
        out.append(np.ndarray((nJ, nlp.ns)))
        out[-1][:, :] = np.nan
        for idx_obj_func in range(nJ):
            nodes = __get_instant(nlp.J[idx_obj_func][0]["objective"].instant, nlp)
            nodes = nodes[: len(nlp.J[idx_obj_func])]
            for node, idx_node in enumerate(nodes):
                obj = casadi.Function("obj", [ocp.V], [get_objective_value(nlp.J[idx_obj_func][node])])
                out[-1][idx_obj_func, idx_node] = obj(sol)
    return out


def get_objective_value(j_dict):
    val = j_dict["val"]
    if j_dict["target"] is not None:
        val -= j_dict["target"]

    if j_dict["objective"].quadratic:
        val = dot(val, val)
    else:
        val = sum1(val)

    val *= j_dict["objective"].weight * j_dict["dt"]
    return val
