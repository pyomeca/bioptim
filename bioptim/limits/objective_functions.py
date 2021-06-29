from typing import Callable, Union, Any
from enum import Enum

from .penalty import PenaltyFunctionAbstract, PenaltyOption
from .penalty_node import PenaltyNodeList
from ..misc.enums import Node
from ..misc.options import OptionList, OptionGeneric


class Objective(PenaltyOption):
    """
    A placeholder for an objective function

    Attributes
    ----------
    weight: float
        The weighting applied to this specific objective function

    Functions
    ---------
    clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Objective)
        Resets a objective function. A negative penalty index creates a new empty objective function.
    add_or_replace_to_penalty_pool(ocp: OptimalControlProgram, nlp: NonLinearProgram, objective: Objective)
        Add the objective function to the objective pool
    """

    def __init__(self, objective: Any, custom_type: Any = None, phase: int = 0, **params: Any):
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

        super(Objective, self).__init__(penalty=objective, phase=phase, custom_function=custom_function, **params)

    def get_penalty_pool(self, all_pn: PenaltyNodeList):
        """
        Add the objective function to the objective pool

        Parameters
        ----------
        all_pn: PenaltyNodeList
                The penalty node elements
        """

        return all_pn.nlp.J if all_pn is not None and all_pn.nlp else all_pn.ocp.J

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
            J_to_add_to = nlp.J
        else:
            J_to_add_to = ocp.J

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
            if self.node != Node.ALL_SHOOTING and self.node != Node.DEFAULT:
                raise RuntimeError("Lagrange objective are for Node.ALL_SHOOTING, did you mean Mayer?")
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
        """
        Print the ObjectiveList to the console
        """
        raise NotImplementedError("Printing of ObjectiveList is not ready yet")


class ObjectiveFunction:
    """
    Internal (re)implementation of the penalty functions

    Methods
    -------
    add_to_penalty(ocp: OptimalControlProgram, pn: PenaltyNodeList, val: Union[MX, SX], penalty: Objective, dt:float=0)
        Add the objective function to the objective pool
    """

    class LagrangeFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions

        Methods
        -------
        _parameter_modifier(objective: Objective)
            Apply some default parameters
        _span_checker(objective: Objective, pn: PenaltyNodeList)
            Check for any non sense in the requested times for the constraint. Raises an error if so
        penalty_nature() -> str
            Get the nature of the penalty
        """

        class Functions:
            """
            Implementation of all the Lagrange objective functions

            Methods
            -------
            minimize_time(penalty: ObjectiveFcn.Lagrange, pn: PenaltyNodeList)
                Minimizes the duration of the phase
            """

            @staticmethod
            def minimize_time(penalty: Objective, all_pn: PenaltyNodeList):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                penalty: Objective,
                    The actual constraint to declare
                all_pn: PenaltyNodeList
                    The penalty node elements
                """

                val = 1
                raise NotImplementedError()
                # # max_bound ans min_bound are already dealt with in OptimalControlProgram.__define_parameters_phase_time
                # if "min_bound" in objective.params:
                #     raise RuntimeError(
                #         "ObjectiveFcn.Lagrange.MINIMIZE_TIME cannot have min_bound. "
                #         "Please either use MAYER or constraint"
                #     )
                # if "max_bound" in objective.params:
                #     raise RuntimeError(
                #         "ObjectiveFcn.Lagrange.MINIMIZE_TIME cannot have max_bound. "
                #         "Please either use MAYER or constraint"
                #     )
                # if not objective.quadratic:
                #     objective.quadratic = True
                # ObjectiveFunction.add_to_penalty(all_pn.ocp, all_pn)

        @staticmethod
        def get_dt(nlp):
            return nlp.dt

        @staticmethod
        def penalty_nature() -> str:
            """
            Get the nature of the penalty

            Returns
            -------
            The nature of the penalty
            """

            return "objective_functions"

    class MayerFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions

        Methods
        -------
        inter_phase_continuity(ocp: OptimalControlProgram, pt: "PhaseTransition")
            Add phase transition objective between two phases.
        _parameter_modifier(objective: Objective)
            Apply some default parameters
        _span_checker(objective: Objective, pn: PenaltyNodeList)
            Check for any non sense in the requested times for the constraint. Raises an error if so
        penalty_nature() -> str
            Get the nature of the penalty
        """

        class Functions:
            """
            Implementation of all the Mayer objective functions

            Methods
            -------
            minimize_time(penalty: "ObjectiveFcn.Lagrange", pn: PenaltyNodeList)
                Minimizes the duration of the phase
            """

            @staticmethod
            def minimize_time(
                penalty: Objective,
                pn: PenaltyNodeList,
            ):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                penalty: Objective,
                    The actual constraint to declare
                pn: PenaltyNodeList
                    The penalty node elements
                """

                val = pn.nlp.tf
                # penalty.quadratic = True
                # if "min_bound" in objective.params:
                #     del objective.params["min_bound"]
                # if "max_bound" in objective.params:
                #     del objective.params["max_bound"]
                # ObjectiveFunction.MayerFunction.add_to_penalty(pn.ocp, pn, val, penalty)

        @staticmethod
        def get_dt(_):
            return 1

        @staticmethod
        def penalty_nature() -> str:
            """
            Get the nature of the penalty

            Returns
            -------
            The nature of the penalty
            """

            return "objective_functions"

    class ParameterFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions

        add_to_penalty(ocp: OptimalControlProgram, _, val: Union[MX, SX], penalty: Objective)
            Add the objective function to the objective pool
        _parameter_modifier(objective: Objective)
            Apply some default parameters
        _span_checker(objective: Objective, pn: PenaltyNodeList)
            Check for any non sense in the requested times for the constraint. Raises an error if so
        penalty_nature() -> str
            Get the nature of the penalty
        """

        class Functions:
            """
            Implementation of all the parameters objective functions
            """

            pass

        @staticmethod
        def penalty_nature() -> str:
            """
            Get the nature of the penalty

            Returns
            -------
            The nature of the penalty
            """

            return "parameters"

    @staticmethod
    def update_target(ocp_or_nlp, list_index, new_target):
        if list_index >= len(ocp_or_nlp.J) or list_index < 0:
            raise ValueError("'list_index' must be defined properly")

        for i, j in enumerate(ocp_or_nlp.J[list_index]):
            j["target"] = new_target[..., i]


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

        MINIMIZE_STATE = (PenaltyFunctionAbstract.Functions.minimize_states, )
        TRACK_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
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
        MINIMIZE_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
        MINIMIZE_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
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
        MINIMIZE_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        TRACK_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        TRACK_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
        SUPERIMPOSE_MARKERS = (PenaltyFunctionAbstract.Functions.superimpose_markers,)
        PROPORTIONAL_STATE = (PenaltyFunctionAbstract.Functions.proportional_states,)
        MINIMIZE_PREDICTED_COM_HEIGHT = (PenaltyFunctionAbstract.Functions.minimize_predicted_com_height,)
        MINIMIZE_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
        MINIMIZE_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
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
