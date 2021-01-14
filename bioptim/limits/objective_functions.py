from enum import Enum

import numpy as np
import casadi

from .penalty import PenaltyType, PenaltyFunctionAbstract, PenaltyOption
from ..misc.enums import Node
from ..misc.options_lists import OptionList, OptionGeneric


class Objective(PenaltyOption):
    def __init__(self, objective, weight=1, custom_type=None, phase=0, **params):
        custom_function = None
        if not isinstance(objective, ObjectiveFcn.Lagrange) and not isinstance(objective, ObjectiveFcn.Mayer):
            custom_function = objective

            if custom_type is None:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is missing. "
                    "It should either be ObjectiveFcn.Mayer or ObjectiveFcn.Lagrange"
                )
            objective = custom_type(custom_type.CUSTOM)
            if isinstance(objective, ObjectiveFcn.Lagrange):
                pass
            elif isinstance(objective, ObjectiveFcn.Mayer):
                pass
            elif isinstance(objective, ObjectiveFcn.Parameter):
                pass
            else:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is invalid. "
                    "It should either be ObjectiveFcn.Mayer or ObjectiveFcn.Lagrange"
                )

        super(Objective, self).__init__(penalty=objective, phase=phase, custom_function=custom_function, **params)
        self.weight = weight


class ObjectiveList(OptionList):
    def add(self, objective, **extra_arguments):
        if isinstance(objective, Objective):
            self.copy(objective)
        else:
            super(ObjectiveList, self)._add(option_type=Objective, objective=objective, **extra_arguments)


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
            def minimize_time(penalty, ocp, nlp, t, x, u, p):
                """Minimizes the duration of the movement (Lagrange)."""
                val = 1
                ObjectiveFunction.LagrangeFunction.add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def add_to_penalty(ocp, nlp, val, penalty):
            """
            Adds an objective.
            :param val: Value to be optimized. (MX.sym from CasADi)
            :param penalty: Index of the objective. (integer)
            :param weight: Weight of the objective. (float)
            :param quadratic: If True, value is squared. (bool)
            """
            ObjectiveFunction.add_to_penalty(ocp, nlp, val, penalty, dt=nlp.dt)

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
            if penalty_function == ObjectiveFcn.Lagrange.MINIMIZE_TIME.value[0]:
                # max_bound ans min_bound are already dealt with in OptimalControlProgram.__define_parameters_phase_time
                if "min_bound" in parameters.params:
                    raise RuntimeError(
                        "ObjectiveFcn.Lagrange.MINIMIZE_TIME cannot have min_bound. "
                        "Please either use MAYER or constraint"
                    )
                if "max_bound" in parameters.params:
                    raise RuntimeError(
                        "ObjectiveFcn.Lagrange.MINIMIZE_TIME cannot have max_bound. "
                        "Please either use MAYER or constraint"
                    )
                if not parameters.quadratic:
                    parameters.quadratic = True
            PenaltyFunctionAbstract._parameter_modifier(penalty_function, parameters)

        @staticmethod
        def _span_checker(objective_function, node, nlp):
            """Raises errors on the span of penalty functions"""
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(objective_function, node, nlp)

    class MayerFunction(PenaltyFunctionAbstract):
        """
        Mayer type objectives. (value of the objective at one time point, usually the end)
        """

        class Functions:
            """
            Biomechanical objectives
            """

            @staticmethod
            def minimize_time(penalty, ocp, nlp, t, x, u, p):
                """Minimizes the duration of the movement (Mayer)."""
                val = nlp.tf
                ObjectiveFunction.MayerFunction.add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def inter_phase_continuity(ocp, pt):
            # Dynamics must be respected between phases
            penalty = OptionGeneric()
            penalty.list_index = -1
            penalty.quadratic = pt.quadratic
            penalty.weight = pt.weight
            penalty.sliced_target = None
            pt.base.clear_penalty(ocp, None, penalty)
            val = pt.type.value[0](ocp, pt)
            pt.base.add_to_penalty(ocp, None, val, penalty)

        @staticmethod
        def add_to_penalty(ocp, nlp, val, penalty):
            """
            Adds an objective.
            :param val: Value to be optimized. (MX.sym from CasADi)
            :param penalty: Index of the objective. (integer)
            :param weight: Weight of the objective. (float)
            :param quadratic: If True, value is squared (bool)
            """
            ObjectiveFunction.add_to_penalty(ocp, nlp, val, penalty, dt=1)

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
            if penalty_function == ObjectiveFcn.Mayer.MINIMIZE_TIME.value[0]:
                # max_bound ans min_bound are already dealt with in OptimalControlProgram.__define_parameters_phase_time
                if "min_bound" in parameters.params:
                    del parameters.params["min_bound"]
                if "max_bound" in parameters.params:
                    del parameters.params["max_bound"]

            PenaltyFunctionAbstract._parameter_modifier(penalty_function, parameters)

        @staticmethod
        def _span_checker(penalty_function, node, nlp):
            """Raises errors on the span of penalty functions"""
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(penalty_function, node, nlp)

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
        def add_to_penalty(ocp, _, val, penalty):
            """
            Adds an objective.
            :param val: Value to be optimized. (MX.sym from CasADi)
            :param penalty: Index of the objective. (integer)
            :param weight: Weight of the objective. (float)
            :param quadratic: If True, value is squared (bool)
            """
            ObjectiveFunction.add_to_penalty(ocp, None, val, penalty, dt=1)

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
        def _span_checker(penalty_function, node, nlp):
            """Raises errors on the span of penalty functions"""
            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(penalty_function, node, nlp)

    @staticmethod
    def add_or_replace(ocp, nlp, objective):
        """
        Modifies or raises errors if user provided Node does not match the objective type.
        :param objective: New objective to replace with. (dictionary)
        """
        if objective.type.get_type() == ObjectiveFunction.LagrangeFunction:
            if objective.node != Node.ALL and objective.node != Node.DEFAULT:
                raise RuntimeError("Lagrange objective are for Node.ALL, did you mean Mayer?")
            objective.node = Node.ALL
        elif objective.type.get_type() == ObjectiveFunction.MayerFunction:
            if objective.node == Node.DEFAULT:
                objective.node = Node.END

        else:
            raise RuntimeError("ObjectiveFcn function Type must be either a Lagrange or Mayer type")
        PenaltyFunctionAbstract.add_or_replace(ocp, nlp, objective)

    @staticmethod
    def cyclic(ocp, weight=1):

        if ocp.nlp[0].nx != ocp.nlp[-1].nx:
            raise RuntimeError("Cyclic constraint without same nx is not supported yet")

        ocp.J += casadi.dot(ocp.nlp[-1].X[-1] - ocp.nlp[0].X[0], ocp.nlp[-1].X[-1] - ocp.nlp[0].X[0]) * weight

    @staticmethod
    def add_to_penalty(ocp, nlp, val, penalty, dt=0):
        """
        Adds objective J to objective array nlp.J[penalty] or ocp.J[penalty] at index penalty.
        :param J: ObjectiveFcn. (dict of [val, target, weight, is_quadratic])
        :param penalty: Index of the objective. (integer)
        """
        J = {"objective": penalty, "val": val, "target": penalty.sliced_target, "dt": dt}

        if nlp:
            nlp.J[penalty.list_index].append(J)
        else:
            ocp.J[penalty.list_index].append(J)

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

        if penalty.list_index < 0:
            # Add a new one
            for i, j in enumerate(J_to_add_to):
                if not j:
                    penalty.list_index = i
                    return
            else:
                J_to_add_to.append([])
                penalty.list_index = len(J_to_add_to) - 1
        else:
            while penalty.list_index >= len(J_to_add_to):
                J_to_add_to.append([])
            J_to_add_to[penalty.list_index] = []


class ObjectivePrinter:
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


class ObjectiveFcn:
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
        MINIMIZE_COM_POSITION = (PenaltyType.MINIMIZE_COM_POSITION,)
        MINIMIZE_COM_VELOCITY = (PenaltyType.MINIMIZE_COM_VELOCITY,)
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
        MINIMIZE_COM_POSITION = (PenaltyType.MINIMIZE_COM_POSITION,)
        MINIMIZE_COM_VELOCITY = (PenaltyType.MINIMIZE_COM_VELOCITY,)
        ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
        ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
        CUSTOM = (PenaltyType.CUSTOM,)

        @staticmethod
        def get_type():
            """Returns the type of the objective function"""
            return ObjectiveFunction.MayerFunction

    class Parameter(Enum):
        CUSTOM = (PenaltyType.CUSTOM,)

        @staticmethod
        def get_type():
            """Returns the type of the objective function"""
            return ObjectiveFunction.ParameterFunction
