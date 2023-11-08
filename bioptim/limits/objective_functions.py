from typing import Callable, Any

from .penalty import PenaltyFunctionAbstract, PenaltyOption
from .penalty_controller import PenaltyController
from ..misc.enums import Node, QuadratureRule, PenaltyType
from ..misc.fcn_enum import FcnEnum
from ..misc.options import OptionList


class Objective(PenaltyOption):
    """
    A placeholder for an objective function
    """

    def __init__(
        self, objective: Any, custom_type: Any = None, phase: int = -1, is_stochastic: bool = False, **params: Any
    ):
        """
        Parameters
        ----------
        objective: ObjectiveFcn.Lagrange | ObjectiveFcn.Mayer | Callable[OptimalControlProgram, MX]
            The chosen objective function
        custom_type: ObjectiveFcn.Lagrange | ObjectiveFcn.Mayer | Callable
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
            if "integration_rule" not in params.keys() or params["integration_rule"] == QuadratureRule.DEFAULT:
                params["integration_rule"] = QuadratureRule.RECTANGLE_LEFT
            if params["integration_rule"] not in (
                QuadratureRule.RECTANGLE_LEFT,
                QuadratureRule.TRAPEZOIDAL,
                QuadratureRule.APPROXIMATE_TRAPEZOIDAL,
            ):
                raise NotImplementedError(
                    f"{params['integration_rule']} has not been implemented yet for objective functions."
                )
        elif isinstance(objective, ObjectiveFcn.Mayer):
            if "integration_rule" in params.keys() and params["integration_rule"] != QuadratureRule.DEFAULT:
                raise ValueError(
                    "Mayer objective functions cannot be integrated, "
                    "remove the argument "
                    "integration_rule"
                    " or use a Lagrange objective function"
                )
        elif isinstance(objective, ObjectiveFcn.Parameter):
            pass

        super(Objective, self).__init__(
            penalty=objective, phase=phase, custom_function=custom_function, is_stochastic=is_stochastic, **params
        )

    def _add_penalty_to_pool(self, controller: PenaltyController):
        if isinstance(controller, (list, tuple)):
            controller = controller[0]  # This is a special case of Node.TRANSITION

        if self.penalty_type == PenaltyType.INTERNAL:
            pool = (
                controller.get_nlp.J_internal
                if controller is not None and controller.get_nlp
                else controller.ocp.J_internal
            )
        elif self.penalty_type == PenaltyType.USER:
            pool = controller.get_nlp.J if controller is not None and controller.get_nlp else controller.ocp.J
        else:
            raise ValueError(f"Invalid objective type {self.penalty_type}.")
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        """
        Resets an objective function. A negative penalty index creates a new empty objective function.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        if self.penalty_type == PenaltyType.INTERNAL:
            J_to_add_to = nlp.J_internal if nlp else ocp.J_internal
        elif self.penalty_type == PenaltyType.USER:
            J_to_add_to = nlp.J if nlp else ocp.J
        else:
            raise ValueError(f"Invalid Type of objective {self.penalty_type}")

        if self.list_index < 0:
            # Add a new one
            for i, j in enumerate(J_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                J_to_add_to.append([])
                self.list_index = len(J_to_add_to) - 1
        else:
            while self.list_index >= len(J_to_add_to):
                J_to_add_to.append([])
            J_to_add_to[self.list_index] = []

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
    add(self, constraint: Callable | "ConstraintFcn", **extra_arguments)
        Add a new Constraint to the list
    print(self):
        Print the ObjectiveList to the console
    """

    def add(self, objective: Callable | Objective | Any, **extra_arguments: Any):
        """
        Add a new objective function to the list

        Parameters
        ----------
        objective: Callable | Objective | ObjectiveFcn.Lagrange | ObjectiveFcn.Mayer
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
            def minimize_time(_: Objective, controller: PenaltyController):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                _: Objective,
                    The actual constraint to declare
                controller: PenaltyController
                    The penalty node elements
                """

                return controller.cx().ones(1, 1)

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
                controller: PenaltyController,
                min_bound: float = None,
                max_bound: float = None,
            ):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                _: Objective,
                    The actual constraint to declare
                controller: PenaltyController
                    The penalty node elements
                min_bound: float
                    The minimum value the time can take (this is ignored here, but
                    taken into account elsewhere in the code)
                max_bound: float
                    The maximal value the time can take (this is ignored here, but
                    taken into account elsewhere in the code)
                """

                return controller.tf

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
            return "parameter_objectives"

    @staticmethod
    def update_target(ocp_or_nlp: Any, list_index: int, new_target: Any):
        """
        Update a specific target

        Parameters
        ----------
        ocp_or_nlp: OptimalControlProgram  NonLinearProgram
            The reference to where to find J
        list_index: int
            The index in J
        new_target
            The target to modify
        """

        if list_index >= len(ocp_or_nlp.J) or list_index < 0:
            raise ValueError("'list_index' must be defined properly")

        ocp_or_nlp.J[list_index].target = [new_target] if not isinstance(new_target, list | tuple) else new_target


class ObjectiveFcn:
    """
    Selection of valid objective functions
    """

    class Lagrange(FcnEnum):
        """
        Selection of valid Lagrange objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)
        MINIMIZE_ANGULAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_angular_momentum,)
        MINIMIZE_COM_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_com_acceleration,)
        MINIMIZE_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
        MINIMIZE_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
        MINIMIZE_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_contact_forces,)
        MINIMIZE_CONTROL = (PenaltyFunctionAbstract.Functions.minimize_controls,)
        MINIMIZE_FATIGUE = (PenaltyFunctionAbstract.Functions.minimize_fatigue,)
        MINIMIZE_LINEAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_linear_momentum,)
        MINIMIZE_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        MINIMIZE_MARKERS_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_markers_acceleration,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        MINIMIZE_POWER = (PenaltyFunctionAbstract.Functions.minimize_power,)
        MINIMIZE_QDDOT = (PenaltyFunctionAbstract.Functions.minimize_qddot,)
        MINIMIZE_SEGMENT_ROTATION = (PenaltyFunctionAbstract.Functions.minimize_segment_rotation,)
        MINIMIZE_SEGMENT_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_segment_velocity,)
        MINIMIZE_SOFT_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_soft_contact_forces,)
        MINIMIZE_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        MINIMIZE_TIME = (ObjectiveFunction.LagrangeFunction.Functions.minimize_time,)
        PROPORTIONAL_CONTROL = (PenaltyFunctionAbstract.Functions.proportional_controls,)
        PROPORTIONAL_STATE = (PenaltyFunctionAbstract.Functions.proportional_states,)
        STOCHASTIC_MINIMIZE_VARIABLE = (PenaltyFunctionAbstract.Functions.stochastic_minimize_variables,)
        STOCHASTIC_MINIMIZE_EXPECTED_FEEDBACK_EFFORTS = (
            PenaltyFunctionAbstract.Functions.stochastic_minimize_expected_feedback_efforts,
        )
        SUPERIMPOSE_MARKERS = (PenaltyFunctionAbstract.Functions.superimpose_markers,)
        TRACK_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_contact_forces,)
        TRACK_CONTROL = (PenaltyFunctionAbstract.Functions.minimize_controls,)
        TRACK_MARKER_WITH_SEGMENT_AXIS = (PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis,)
        TRACK_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        TRACK_MARKERS_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_markers_acceleration,)
        TRACK_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        TRACK_POWER = (PenaltyFunctionAbstract.Functions.minimize_power,)
        TRACK_SEGMENT_WITH_CUSTOM_RT = (PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt,)
        TRACK_SOFT_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_soft_contact_forces,)
        TRACK_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS = (
            PenaltyFunctionAbstract.Functions.track_vector_orientations_from_markers,
        )

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.LagrangeFunction

    class Mayer(FcnEnum):
        """
        Selection of valid Mayer objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        STATE_CONTINUITY = (PenaltyFunctionAbstract.Functions.state_continuity,)
        CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)
        MINIMIZE_ANGULAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_angular_momentum,)
        MINIMIZE_COM_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_com_acceleration,)
        MINIMIZE_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
        MINIMIZE_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
        MINIMIZE_FATIGUE = (PenaltyFunctionAbstract.Functions.minimize_fatigue,)
        MINIMIZE_LINEAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_linear_momentum,)
        MINIMIZE_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        MINIMIZE_MARKERS_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_markers_acceleration,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        MINIMIZE_POWER = (PenaltyFunctionAbstract.Functions.minimize_power,)
        MINIMIZE_PREDICTED_COM_HEIGHT = (PenaltyFunctionAbstract.Functions.minimize_predicted_com_height,)
        MINIMIZE_QDDOT = (PenaltyFunctionAbstract.Functions.minimize_qddot,)
        MINIMIZE_SEGMENT_ROTATION = (PenaltyFunctionAbstract.Functions.minimize_segment_rotation,)
        MINIMIZE_SEGMENT_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_segment_velocity,)
        MINIMIZE_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        MINIMIZE_TIME = (ObjectiveFunction.MayerFunction.Functions.minimize_time,)
        PROPORTIONAL_STATE = (PenaltyFunctionAbstract.Functions.proportional_states,)
        STOCHASTIC_MINIMIZE_VARIABLE = (PenaltyFunctionAbstract.Functions.stochastic_minimize_variables,)
        SUPERIMPOSE_MARKERS = (PenaltyFunctionAbstract.Functions.superimpose_markers,)
        SUPERIMPOSE_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.superimpose_markers_velocity,)
        TRACK_MARKER_WITH_SEGMENT_AXIS = (PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis,)
        TRACK_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        TRACK_MARKERS_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_markers_acceleration,)
        TRACK_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        TRACK_POWER = (PenaltyFunctionAbstract.Functions.minimize_power,)
        TRACK_SEGMENT_WITH_CUSTOM_RT = (PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt,)
        TRACK_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
        TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS = (
            PenaltyFunctionAbstract.Functions.track_vector_orientations_from_markers,
        )

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.MayerFunction

    class Multinode(FcnEnum):
        """
        Selection of valid Multinode objective functions

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
            return ObjectiveFunction.MultinodeFunction

    class Parameter(FcnEnum):
        """
        Selection of valid Parameters objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        MINIMIZE_PARAMETER = (PenaltyFunctionAbstract.Functions.minimize_parameter,)
        CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.ParameterFunction


class ParameterObjective(PenaltyOption):
    """
    A placeholder for an objective function
    """

    def __init__(self, parameter_objective: Any, custom_type: Any = None, **params: Any):
        """
        Parameters
        ----------
        parameter_objective: ObjectiveFcn.Lagrange | ObjectiveFcn.Mayer | Callable[OptimalControlProgram, MX]
            The chosen objective function
        custom_type: ObjectiveFcn.Parameter | Callable
            The custom parameter objective function
        params: dict
            Generic parameters for options
        """
        custom_function = None
        if not isinstance(parameter_objective, ObjectiveFcn.Parameter):
            custom_function = parameter_objective

            if custom_type is None:
                raise RuntimeError(
                    "Custom parameter objective function detected, but custom_function is missing. "
                    "It should ObjectiveFcn.Parameter"
                )
            parameter_objective = custom_type(custom_type.CUSTOM)
            if isinstance(parameter_objective, ObjectiveFcn.Parameter):
                pass
            else:
                raise RuntimeError(
                    "Custom parameter objective function detected, but custom_function is invalid. "
                    "It should either be ObjectiveFcn.Parameter"
                )

        super(ParameterObjective, self).__init__(penalty=parameter_objective, custom_function=custom_function, **params)

    def _add_penalty_to_pool(self, controller: PenaltyController):
        if isinstance(controller, (list, tuple)):
            controller = controller[0]  # This is a special case of Node.TRANSITION

        if self.penalty_type == PenaltyType.INTERNAL:
            pool = (
                controller.get_nlp.J_internal
                if controller is not None and controller.get_nlp
                else controller.ocp.J_internal
            )
        elif self.penalty_type == PenaltyType.USER:
            pool = controller.get_nlp.J if controller is not None and controller.get_nlp else controller.ocp.J
        else:
            raise ValueError(f"Invalid objective type {self.penalty_type}.")
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        """
        Resets a objective function. A negative penalty index creates a new empty objective function.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        if self.penalty_type == PenaltyType.INTERNAL:
            J_to_add_to = nlp.J_internal if nlp else ocp.J_internal
        elif self.penalty_type == PenaltyType.USER:
            J_to_add_to = nlp.J if nlp else ocp.J
        else:
            raise ValueError(f"Invalid Type of objective {self.penalty_type}")

        if self.list_index < 0:
            # Add a new one
            for i, j in enumerate(J_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                J_to_add_to.append([])
                self.list_index = len(J_to_add_to) - 1
        else:
            while self.list_index >= len(J_to_add_to):
                J_to_add_to.append([])
            J_to_add_to[self.list_index] = []

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        super(ParameterObjective, self).add_or_replace_to_penalty_pool(ocp, nlp)

    @staticmethod
    def get_type():
        return "parameter_objective"


class ParameterObjectiveList(OptionList):
    """
    A list of objectives which apply for all phases at once if more than one is required

    Methods
    -------
    add(self, parameter_objective: Callable | "ParameterObjectiveFcn", **extra_arguments)
        Add a new ParameterObjective to the list
    print(self):
        Print the ParameterObjectiveList to the console
    """

    def add(self, parameter_objective: Callable | ParameterObjective | Any, **extra_arguments: Any):
        """
        Add a new parameter objective function to the list

        Parameters
        ----------
        parameter_objective: Callable | ParameterObjective
            The chosen parameter objective function
        extra_arguments: dict
            Any parameters to pass to ParameterObjectiveFcn
        """

        if isinstance(parameter_objective, ParameterObjective):
            self.copy(parameter_objective)
        else:
            super(ParameterObjectiveList, self)._add(
                option_type=ParameterObjective, parameter_objective=parameter_objective, **extra_arguments
            )

    def print(self):
        raise NotImplementedError("Printing of ParameterObjectiveList is not ready yet")
