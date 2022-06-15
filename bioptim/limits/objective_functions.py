from typing import Callable, Union, Any
from enum import Enum

from .penalty import PenaltyFunctionAbstract, PenaltyOption
from .penalty_node import PenaltyNodeList
from ..misc.enums import Node, IntegralApproximation, PenaltyType
from ..misc.options import OptionList


class Objective(PenaltyOption):
    """
    A placeholder for an objective function
    """

    def __init__(self, objective: Any, custom_type: Any = None, phase: int = -1, **params: Any):
        """
        Parameters
        ----------
        objective: Union[ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, Callable[OptimalControlProgram, MX]]
            The chosen objective function
        custom_type: Union[ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, Callable]
            When objective is a custom defined function, one must specify if the custom_type is Mayer or Lagrange
        phase: int
            At which phase this objective function must be applied
        params: dict
            Generic parameters for options
        """

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

        # sanity check on the integration method
        if isinstance(objective, ObjectiveFcn.Lagrange):
            if "integration_rule" not in params.keys() or params["integration_rule"] == IntegralApproximation.DEFAULT:
                params["integration_rule"] = IntegralApproximation.RECTANGLE
        elif isinstance(objective, ObjectiveFcn.Mayer):
            if "integration_rule" in params.keys() and params["integration_rule"] != IntegralApproximation.DEFAULT:
                raise ValueError(
                    "Mayer objective functions cannot be integrated, "
                    "remove the argument "
                    "integration_rule"
                    " or use a Lagrange objective function"
                )
        elif isinstance(objective, ObjectiveFcn.Parameter):
            pass

        super(Objective, self).__init__(penalty=objective, phase=phase, custom_function=custom_function, **params)

    def _add_penalty_to_pool(self, all_pn: Union[PenaltyNodeList, list, tuple]):
        if isinstance(all_pn, (list, tuple)):
            pool = all_pn[0].nlp.J if all_pn[0] is not None and all_pn[0].nlp else all_pn[0].ocp.J
        else:
            pool = all_pn.nlp.J if all_pn is not None and all_pn.nlp else all_pn.ocp.J
        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        """
        Resets a objective function. A negative penalty index creates a new empty objective function.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        if nlp:
            j_to_add_to = nlp.J
        else:
            j_to_add_to = ocp.J

        if self.list_index < 0:
            # Add a new one
            for i, j in enumerate(j_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                j_to_add_to.append([])
                self.list_index = len(j_to_add_to) - 1
        else:
            while self.list_index >= len(j_to_add_to):
                j_to_add_to.append([])
            j_to_add_to[self.list_index] = []

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        if self.type.get_type() == ObjectiveFunction.LagrangeFunction:
            if self.node != Node.ALL_SHOOTING and self.node != Node.ALL and self.node != Node.DEFAULT:
                raise RuntimeError("Lagrange objective are for Node.ALL_SHOOTING or Node.ALL, did you mean Mayer?")
            if self.node == Node.DEFAULT:
                self.node = Node.ALL_SHOOTING
        elif self.type.get_type() == ObjectiveFunction.MayerFunction:
            if self.node == Node.DEFAULT:
                self.node = Node.END
        else:
            raise RuntimeError("Objective is not Mayer or Lagrange")

        super(Objective, self).add_or_replace_to_penalty_pool(ocp, nlp)


class ObjectiveList(OptionList):
    """
    A list of Constraint if more than one is required

    Methods
    -------
    add(self, constraint: Union[Callable, "ConstraintFcn"], **extra_arguments)
        Add a new Constraint to the list
    print(self):
        Print the ObjectiveList to the console
    """

    def add(self, objective: Union[Callable, Objective, Any], **extra_arguments: Any):
        """
        Add a new objective function to the list

        Parameters
        ----------
        objective: Union[Callable, Objective, ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer]
            The chosen objective function
        extra_arguments: dict
            Any parameters to pass to ObjectiveFcn
        """

        if isinstance(objective, Objective):
            self.copy(objective)
        else:
            super(ObjectiveList, self)._add(option_type=Objective, objective=objective, **extra_arguments)

    def print(self):
        raise NotImplementedError("Printing of ObjectiveList is not ready yet")


class ObjectiveFunction:
    """
    Internal (re)implementation of the penalty functions

    Methods
    -------
    update_target(ocp_or_nlp: Any, list_index: int, new_target: Any)
        Update a specific target
    """

    class LagrangeFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions
        """

        class Functions:
            """
            Implementation of all the Lagrange objective functions
            """

            @staticmethod
            def minimize_time(_: Objective, all_pn: PenaltyNodeList):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                _: Objective,
                    The actual constraint to declare
                all_pn: PenaltyNodeList
                    The penalty node elements
                """

                return all_pn.nlp.cx().ones(1, 1)

        @staticmethod
        def get_dt(nlp):
            return nlp.dt

        @staticmethod
        def penalty_nature() -> str:
            return "objective_functions"

    class MayerFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions
        """

        class Functions:
            """
            Implementation of all the Mayer objective functions
            """

            @staticmethod
            def minimize_time(
                _: Objective,
                all_pn: PenaltyNodeList,
                min_bound: float = None,
                max_bound: float = None,
            ):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                _: Objective,
                    The actual constraint to declare
                all_pn: PenaltyNodeList
                    The penalty node elements
                min_bound: float
                    The minimum value the time can take (this is ignored here, but
                    taken into account elsewhere in the code)
                max_bound: float
                    The maximal value the time can take (this is ignored here, but
                    taken into account elsewhere in the code)
                """

                return all_pn.nlp.tf

        @staticmethod
        def get_dt(_):
            return 1

        @staticmethod
        def penalty_nature() -> str:
            return "objective_functions"

    class ParameterFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions
        """

        class Functions:
            """
            Implementation of all the parameters objective functions
            """

            pass

        @staticmethod
        def penalty_nature() -> str:
            return "parameters"

    @staticmethod
    def update_target(ocp_or_nlp: Any, list_index: int, new_target: Any):
        """
        Update a specific target

        Parameters
        ----------
        ocp_or_nlp: Union[OptimalControlProgram, NonLinearProgram]
            The reference to where to find J
        list_index: int
            The index in J
        new_target
            The target to modify
        """

        if list_index >= len(ocp_or_nlp.J) or list_index < 0:
            raise ValueError("'list_index' must be defined properly")

        ocp_or_nlp.J[list_index].target = [new_target] if not isinstance(new_target, Union[list, tuple]) else new_target

    @staticmethod
    def inner_phase_continuity(ocp):
        """
        Add continuity objectives between each nodes of a phase.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        # Dynamics must be sound within phases
        for i, nlp in enumerate(ocp.nlp):
            penalty = Objective(
                ObjectiveFcn.Lagrange.CONTINUITY,
                node=Node.ALL_SHOOTING,
                penalty_type=PenaltyType.INTERNAL,
            )
            penalty.add_or_replace_to_penalty_pool(ocp, nlp)

    @staticmethod
    def inter_phase_continuity(ocp):
        """
        Add phase transition objectives between two phases.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """
        for i, pt in enumerate(ocp.phase_transitions):  # TODO: is it constraint?
            # Dynamics must be respected between phases
            pt.name = f"PHASE_TRANSITION {pt.phase_pre_idx}->{pt.phase_post_idx}"
            pt.list_index = -1
            pt.add_or_replace_to_penalty_pool(ocp, ocp.nlp[pt.phase_pre_idx])

    @staticmethod
    def node_equalities(ocp):
        """
        Add multi node objectives between chosen phases.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """
        for i, mnc in enumerate(ocp.multinode_objectives):
            # Equality constraint between nodes
            first_node_name = f"idx {str(mnc.first_node)}" if isinstance(mnc.first_node, int) else mnc.first_node.name
            second_node_name = (
                f"idx {str(mnc.second_node)}" if isinstance(mnc.second_node, int) else mnc.second_node.name
            )
            mnc.name = (
                f"NODE_EQUALITY "
                f"Phase {mnc.phase_first_idx} Node {first_node_name}"
                f"->Phase {mnc.phase_second_idx} Node {second_node_name}"
            )
            mnc.list_index = -1
            mnc.add_or_replace_to_penalty_pool(ocp, ocp.nlp[mnc.phase_first_idx])


class ObjectiveFcn:
    """
    Selection of valid objective functions
    """

    class Lagrange(Enum):
        """
        Selection of valid Lagrange objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        CONTINUITY = (PenaltyFunctionAbstract.Functions.continuity,)
        MINIMIZE_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        TRACK_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        MINIMIZE_FATIGUE = (PenaltyFunctionAbstract.Functions.minimize_fatigue,)
        MINIMIZE_CONTROL = (PenaltyFunctionAbstract.Functions.minimize_controls,)
        TRACK_CONTROL = (PenaltyFunctionAbstract.Functions.minimize_controls,)
        SUPERIMPOSE_MARKERS = (PenaltyFunctionAbstract.Functions.superimpose_markers,)
        MINIMIZE_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        TRACK_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        MINIMIZE_TIME = (ObjectiveFunction.LagrangeFunction.Functions.minimize_time,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        TRACK_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        PROPORTIONAL_STATE = (PenaltyFunctionAbstract.Functions.proportional_states,)
        PROPORTIONAL_CONTROL = (PenaltyFunctionAbstract.Functions.proportional_controls,)
        MINIMIZE_QDDOT = (PenaltyFunctionAbstract.Functions.minimize_qddot,)
        MINIMIZE_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_contact_forces,)
        TRACK_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_contact_forces,)
        MINIMIZE_SOFT_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_soft_contact_forces,)
        TRACK_SOFT_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_soft_contact_forces,)
        MINIMIZE_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
        MINIMIZE_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
        MINIMIZE_COM_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_com_acceleration,)
        MINIMIZE_ANGULAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_angular_momentum,)
        MINIMIZE_LINEAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_linear_momentum,)
        TRACK_SEGMENT_WITH_CUSTOM_RT = (PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt,)
        TRACK_MARKER_WITH_SEGMENT_AXIS = (PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis,)
        CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.LagrangeFunction

    class Mayer(Enum):
        """
        Selection of valid Mayer objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        MINIMIZE_TIME = (ObjectiveFunction.MayerFunction.Functions.minimize_time,)
        MINIMIZE_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        TRACK_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        MINIMIZE_FATIGUE = (PenaltyFunctionAbstract.Functions.minimize_fatigue,)
        MINIMIZE_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        TRACK_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        TRACK_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        SUPERIMPOSE_MARKERS = (PenaltyFunctionAbstract.Functions.superimpose_markers,)
        PROPORTIONAL_STATE = (PenaltyFunctionAbstract.Functions.proportional_states,)
        MINIMIZE_PREDICTED_COM_HEIGHT = (PenaltyFunctionAbstract.Functions.minimize_predicted_com_height,)
        MINIMIZE_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
        MINIMIZE_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
        MINIMIZE_COM_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_com_acceleration,)
        MINIMIZE_ANGULAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_angular_momentum,)
        MINIMIZE_LINEAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_linear_momentum,)
        TRACK_SEGMENT_WITH_CUSTOM_RT = (PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt,)
        TRACK_MARKER_WITH_SEGMENT_AXIS = (PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis,)
        CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.MayerFunction

    class Parameter(Enum):
        """
        Selection of valid Parameters objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.ParameterFunction


class ContinuityObjectiveFunctions:
    """
    Interface between continuity and objective
    """

    @staticmethod
    def continuity(ocp):
        """
        The declaration of inner- and inter-phase continuity objectives

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        ObjectiveFunction.inner_phase_continuity(ocp)

        # Dynamics must be respected between phases
        ObjectiveFunction.inter_phase_continuity(ocp)

        if ocp.multinode_constraints:  # TODO: change for objective
            ObjectiveFunction.node_equalities(ocp)
