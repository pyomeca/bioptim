from typing import Callable, Union, Any
from enum import Enum

import numpy as np
from casadi import sum1, horzcat, if_else, vertcat, lt, MX, SX, Function
import biorbd

from .path_conditions import Bounds
from .penalty import PenaltyFunctionAbstract, PenaltyOption, PenaltyNodeList
from ..dynamics.ode_solver import OdeSolver
from ..misc.enums import Node, ControlType, InterpolationType
from ..misc.options import OptionList, OptionGeneric


class Constraint(PenaltyOption):
    """
    A placeholder for a constraint

    Attributes
    ----------
    min_bound: np.ndarray
        The vector of minimum bound of the constraint. Default is 0
    max_bound: np.ndarray
        The vector of maximal bound of the constraint. Default is 0
    """

    def __init__(
        self,
        constraint: Any,
        min_bound: Union[np.ndarray, float] = None,
        max_bound: Union[np.ndarray, float] = None,
        quadratic: bool = False,
        phase: int = 0,
        **params: Any,
    ):
        """
        Parameters
        ----------
        constraint: ConstraintFcn
            The chosen constraint
        min_bound: np.ndarray
            The vector of minimum bound of the constraint. Default is 0
        max_bound: np.ndarray
            The vector of maximal bound of the constraint. Default is 0
        phase: int
            The index of the phase to apply the constraint
        quadratic: bool
            If the penalty is quadratic
        params:
            Generic parameters for options
        """
        custom_function = None
        if not isinstance(constraint, ConstraintFcn):
            custom_function = constraint
            constraint = ConstraintFcn.CUSTOM

        super(Constraint, self).__init__(penalty=constraint, phase=phase, quadratic=quadratic, custom_function=custom_function, **params)
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(interpolation=InterpolationType.CONSTANT)

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        """
        Doing some configuration before calling the super.add_or_replace function that prepares the adding of the
        constraint to the constraint pool

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        if self.type == ConstraintFcn.TIME_CONSTRAINT:
            self.node = Node.END

        super(Constraint, self).add_or_replace_to_penalty_pool(ocp, nlp)

        self.min_bound = 0 if self.min_bound is None else self.min_bound
        self.max_bound = 0 if self.max_bound is None else self.max_bound

        for i in self.rows:
            min_bound = (
                self.min_bound[i]
                if hasattr(self.min_bound, "__getitem__") and self.min_bound.shape[0] > 1
                else self.min_bound
            )
            max_bound = (
                self.max_bound[i]
                if hasattr(self.max_bound, "__getitem__") and self.max_bound.shape[0] > 1
                else self.max_bound
            )
            self.bounds.concatenate(Bounds(min_bound, max_bound, interpolation=InterpolationType.CONSTANT))

    def get_penalty_pool(self, all_pn: PenaltyNodeList):
        """
        Add the objective function to the objective pool

        Parameters
        ----------
        all_pn: PenaltyNodeList
                The penalty node elements
        """

        if self.is_internal:
            return all_pn.nlp.g_internal if all_pn is not None and all_pn.nlp else all_pn.ocp.g_internal
        else:
            return all_pn.nlp.g if all_pn is not None and all_pn.nlp else all_pn.ocp.g

    def clear_penalty(self, ocp, nlp):
        """
        Resets a constraint. A negative penalty index creates a new empty constraint.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        if self.is_internal:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal
        else:
            g_to_add_to = nlp.g if nlp else ocp.g

        if self.list_index < 0:
            for i, j in enumerate(g_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                g_to_add_to.append([])
                self.list_index = len(g_to_add_to) - 1
        else:
            while self.list_index >= len(g_to_add_to):
                g_to_add_to.append([])
            g_to_add_to[self.list_index] = []


class ConstraintList(OptionList):
    """
    A list of Constraint if more than one is required

    Methods
    -------
    add(self, constraint: Union[Callable, "ConstraintFcn"], **extra_arguments)
        Add a new Constraint to the list
    print(self)
        Print the ConstraintList to the console
    """

    def add(self, constraint: Union[Callable, Constraint, Any], **extra_arguments: Any):
        """
        Add a new constraint to the list

        Parameters
        ----------
        constraint: Union[Callable, Constraint, ConstraintFcn]
            The chosen constraint
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if isinstance(constraint, Constraint):
            self.copy(constraint)

        else:
            super(ConstraintList, self)._add(option_type=Constraint, constraint=constraint, **extra_arguments)

    def print(self):
        """
        Print the ConstraintList to the console
        """
        # TODO: Print all elements in the console
        raise NotImplementedError("Printing of ConstraintList is not ready yet")


class ConstraintFunction(PenaltyFunctionAbstract):
    """
    Internal (re)implementation of the penalty functions

    Methods
    -------
    inter_phase_continuity(ocp: OptimalControlProgram, pt: "PhaseTransition")
        Add phase transition constraints between two phases.
    add_to_penalty(ocp: OptimalControlProgram, pn: PenaltyNodeList, val: Union[MX, SX], penalty: Constraint)
        Add the constraint to the constraint pool
    clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Constraint)
        Resets a penalty. A negative penalty index creates a new empty penalty.
    _parameter_modifier(constraint: Constraint)
        Apply some default parameters
    _span_checker(constraint, nlp)
        Check for any non sense in the requested times for the constraint. Raises an error if so
    penalty_nature() -> str
        Get the nature of the penalty
    """

    class Functions:
        """
        Implementation of all the constraint functions

        Methods
        -------
        time_constraint(constraint: Constraint, pn: PenaltyNodeList)
            The time constraint is taken care elsewhere, but must be declared here. This function therefore does nothing
        torque_max_from_actuators(constraint: Constraint, pn: PenaltyNodeList, min_torque=None)
            Non linear maximal values of joint torques computed from the torque-position-velocity relationship
        non_slipping(constraint: Constraint, pn: PenaltyNodeList,
                tangential_component_idx: int, normal_component_idx: int, static_friction_coefficient: float)
            Add a constraint of static friction at contact points allowing for small tangential forces. This constraint
            assumes that the normal forces is positive
        contact_force(constraint: Constraint, pn: PenaltyNodeList, contact_force_idx: int)
            Add a constraint of contact forces given by any forward dynamics with contact
        """

        @staticmethod
        def non_slipping(
            constraint: Constraint,
            all_pn: PenaltyNodeList,
            tangential_component_idx: int,
            normal_component_idx: int,
            static_friction_coefficient: float,
        ):
            """
            Add a constraint of static friction at contact points constraining for small tangential forces.
            This function make the assumption that normal_force is always positive
            That is mu*normal_force = tangential_force. To prevent from using a square root, the previous
            equation is squared

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            tangential_component_idx: int
                Index of the tangential component of the contact force.
                [0] = x_indices, [1] = y_indices / or [0] = component
            normal_component_idx: int
                Index of the normal component of the contact force
            static_friction_coefficient: float
                Static friction coefficient
            """

            if isinstance(tangential_component_idx, int):
                tangential_component_idx = [tangential_component_idx]
            elif not isinstance(tangential_component_idx, (tuple, list)):
                raise RuntimeError("tangential_component_idx must be a unique integer or a list of integer")

            if isinstance(normal_component_idx, int):
                normal_component_idx = [normal_component_idx]
            elif not isinstance(normal_component_idx, (tuple, list)):
                raise RuntimeError("normal_component_idx must be a unique integer or a list of integer")

            mu_squared = static_friction_coefficient ** 2
            constraint.min_bound = np.array([0, 0])
            constraint.max_bound = np.array([np.inf, np.inf])
            for i in range(len(all_pn.u)):
                contact = all_pn.nlp.contact_forces_func(all_pn.x[i], all_pn.u[i], all_pn.p)
                normal_contact_force_squared = sum1(contact[normal_component_idx, 0]) ** 2
                if len(tangential_component_idx) == 1:
                    tangential_contact_force_squared = sum1(contact[tangential_component_idx[0], 0]) ** 2
                elif len(tangential_component_idx) == 2:
                    tangential_contact_force_squared = (
                        sum1(contact[tangential_component_idx[0], 0]) ** 2
                        + sum1(contact[tangential_component_idx[1], 0]) ** 2
                    )
                else:
                    raise (ValueError("tangential_component_idx should either be x and y or only one component"))

                # Since it is non-slipping normal forces are supposed to be greater than zero
                ConstraintFunction.add_to_penalty(
                    all_pn.ocp,
                    all_pn,
                    vertcat(
                        mu_squared * normal_contact_force_squared - tangential_contact_force_squared,
                        mu_squared * normal_contact_force_squared + tangential_contact_force_squared,
                    ),
                    constraint,
                )

        @staticmethod
        def torque_max_from_actuators(
            constraint: Constraint,
            all_pn: PenaltyNodeList,
            min_torque=None,
        ):
            """
            Non linear maximal values of joint torques computed from the torque-position-velocity relationship

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            min_torque: float
                Minimum joint torques. This prevent from having too small torques, but introduces an if statement
            """

            # TODO: Add index to select the u (control_idx)
            nlp = all_pn.nlp
            q = [nlp.variable_mappings["q"].to_second.map(mx[nlp.states["q"].index, :]) for mx in all_pn.x]
            qdot = [nlp.variable_mappings["qdot"].to_second.map(mx[nlp.states["qdot"].index, :]) for mx in all_pn.x]

            if min_torque and min_torque < 0:
                raise ValueError("min_torque cannot be negative in tau_max_from_actuators")
            func = biorbd.to_casadi_func("torqueMax", nlp.model.torqueMax, nlp.states["q"].mx, nlp.states["qdot"].mx)
            constraint.min_bound = np.repeat([0, -np.inf], nlp.controls.shape)
            constraint.max_bound = np.repeat([np.inf, 0], nlp.controls.shape)
            for i in range(len(all_pn.u)):
                bound = func(q[i], qdot[i])
                if min_torque:
                    min_bound = nlp.variable_mappings["tau"].to_first.map(
                        if_else(lt(bound[:, 1], min_torque), min_torque, bound[:, 1])
                    )
                    max_bound = nlp.variable_mappings["tau"].to_first.map(
                        if_else(lt(bound[:, 0], min_torque), min_torque, bound[:, 0])
                    )
                else:
                    min_bound = nlp.variable_mappings["tau"].to_first.map(bound[:, 1])
                    max_bound = nlp.variable_mappings["tau"].to_first.map(bound[:, 0])

                ConstraintFunction.add_to_penalty(
                    all_pn.ocp, all_pn, vertcat(*[all_pn.u[i] + min_bound, all_pn.u[i] - max_bound]), constraint
                )

        @staticmethod
        def time_constraint(
            constraint: Constraint,
            all_pn: PenaltyNodeList,
            **unused_param,
        ):
            """
            The time constraint is taken care elsewhere, but must be declared here. This function therefore does nothing

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            pass

    @staticmethod
    def inner_phase_continuity(ocp):
        """
        Add continuity constraints between each nodes of a phase.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """
        # Dynamics must be sound within phases
        for i, nlp in enumerate(ocp.nlp):
            if ocp.n_threads > 1:
                raise NotImplementedError("n_threads is not implemented yet")
                # end_nodes = nlp.par_dynamics(horzcat(*nlp.X[:-1]), horzcat(*nlp.U), nlp.parameters.cx)[0]
                #     val = horzcat(*nlp.X[1:]) - end_nodes
                #     ConstraintFunction.add_to_penalty(ocp, None, val.reshape((nlp.states.shape * nlp.ns, 1)), penalty)
            else:
                for j in range(nlp.ns):
                    penalty = Constraint(ConstraintFcn.CONTINUITY, node=j, is_internal=True)
                    penalty.add_or_replace_to_penalty_pool(ocp, nlp)

    @staticmethod
    def inter_phase_continuity(ocp, pt):
        """
        Add phase transition constraints between two phases.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        pt: PhaseTransition
            The phase transition to add
        """

        # Dynamics must be respected between phases
        penalty = OptionGeneric()
        penalty.name = f"PHASE_TRANSITION {pt.phase_pre_idx}->{pt.phase_pre_idx + 1}"
        penalty.min_bound = 0
        penalty.max_bound = 0
        penalty.list_index = -1
        penalty.sliced_target = None
        pt.base.clear_penalty(ocp, None, penalty)
        val = pt.type.value[0](ocp, pt)
        casadi_name = f"PHASE_TRANSITION_{pt.phase_pre_idx}_{pt.phase_pre_idx + 1}"
        pre_nlp, post_nlp = ocp.nlp[pt.phase_pre_idx], ocp.nlp[(pt.phase_pre_idx + 1) % ocp.n_phases]
        pt.casadi_function = Function(
            casadi_name,
            [pre_nlp.X[-1], pre_nlp.U[-1], post_nlp.X[0], post_nlp.U[0], ocp.v.parameters_in_list.cx],
            [val],
        ).expand()
        pt.base.add_to_penalty(ocp, None, val, penalty)

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

        return "constraints"


class ConstraintFcn(Enum):
    """
    Selection of valid constraint functions

    Methods
    -------
    def get_type() -> Callable
        Returns the type of the penalty
    """

    CONTINUITY = (PenaltyFunctionAbstract.Functions.continuity,)
    TRACK_CONTROL = (PenaltyFunctionAbstract.Functions.minimize_controls,)
    TRACK_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
    TRACK_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
    TRACK_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
    SUPERIMPOSE_MARKERS = (PenaltyFunctionAbstract.Functions.superimpose_markers,)
    PROPORTIONAL_STATE = (PenaltyFunctionAbstract.Functions.proportional_states,)
    PROPORTIONAL_CONTROL = (PenaltyFunctionAbstract.Functions.proportional_controls,)
    TRACK_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_contact_forces,)
    TRACK_SEGMENT_WITH_CUSTOM_RT = (PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt,)
    TRACK_MARKER_WITH_SEGMENT_AXIS = (PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis,)
    TRACK_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
    TRACK_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
    CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)
    CONTACT_FORCE = (ConstraintFunction.Functions.contact_force,)
    NON_SLIPPING = (ConstraintFunction.Functions.non_slipping,)
    TORQUE_MAX_FROM_ACTUATORS = (ConstraintFunction.Functions.torque_max_from_actuators,)
    TIME_CONSTRAINT = (ConstraintFunction.Functions.time_constraint,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return ConstraintFunction


class ContinuityFunctions:
    """
    Interface between continuity and constraint
    """

    @staticmethod
    def continuity(ocp):
        """
        The declaration of inner- and inter-phase continuity constraints

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        ConstraintFunction.inner_phase_continuity(ocp)

        # Dynamics must be respected between phases
        for pt in ocp.phase_transitions:
            pt.base.inter_phase_continuity(ocp, pt)
