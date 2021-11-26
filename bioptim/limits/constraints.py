from typing import Callable, Union, Any
from enum import Enum

import numpy as np
from casadi import sum1, if_else, vertcat, lt, SX, MX

from .path_conditions import Bounds
from .penalty import PenaltyFunctionAbstract, PenaltyOption, PenaltyNodeList
from ..interfaces.biorbd_interface import BiorbdInterface
from ..misc.enums import Node, InterpolationType, ConstraintType
from ..misc.options import OptionList


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
        phase: int = -1,
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
        if not isinstance(constraint, (ConstraintFcn, ImplicitConstraintFcn)):
            custom_function = constraint
            constraint = ConstraintFcn.CUSTOM

        super(Constraint, self).__init__(
            penalty=constraint, phase=phase, quadratic=quadratic, custom_function=custom_function, **params
        )
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(interpolation=InterpolationType.CONSTANT)

    def set_penalty(self, penalty: Union[MX, SX], all_pn: PenaltyNodeList):
        super(Constraint, self).set_penalty(penalty, all_pn)
        self.min_bound = 0 if self.min_bound is None else self.min_bound
        self.max_bound = 0 if self.max_bound is None else self.max_bound

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        if self.type == ConstraintFcn.TIME_CONSTRAINT:
            self.node = Node.END

        super(Constraint, self).add_or_replace_to_penalty_pool(ocp, nlp)

        self.min_bound = np.array(self.min_bound) if isinstance(self.min_bound, (list, tuple)) else self.min_bound
        self.max_bound = np.array(self.max_bound) if isinstance(self.max_bound, (list, tuple)) else self.max_bound

        if self.bounds.shape[0] == 0:
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
        elif self.bounds.shape[0] != len(self.rows):
            raise RuntimeError(f"bounds rows is {self.bounds.shape[0]} but should be {self.rows} or empty")

    def _add_penalty_to_pool(self, all_pn: PenaltyNodeList):
        if self.constraint_type == ConstraintType.INTERNAL:
            pool = all_pn.nlp.g_internal if all_pn is not None and all_pn.nlp else all_pn.ocp.g_internal
        elif self.constraint_type == ConstraintType.IMPLICIT:
            pool = all_pn.nlp.g_implicit if all_pn is not None and all_pn.nlp else all_pn.ocp.g_implicit
        elif self.constraint_type == ConstraintType.USER:
            pool = all_pn.nlp.g if all_pn is not None and all_pn.nlp else all_pn.ocp.g
        else:
            raise ValueError(f"Invalid constraint type {self.contraint_type}.")
        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        if self.constraint_type == ConstraintType.INTERNAL:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal
        elif self.constraint_type == ConstraintType.IMPLICIT:
            g_to_add_to = nlp.g_implicit if nlp else ocp.g_implicit
        elif self.constraint_type == ConstraintType.USER:
            g_to_add_to = nlp.g if nlp else ocp.g
        else:
            raise ValueError(f"Invalid Type of Constraint {self.constraint_type}")

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
    inner_phase_continuity(ocp)
        Add continuity constraints between each nodes of a phase.
    inter_phase_continuity(ocp)
        Add phase transition constraints between two phases.
    clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Constraint)
        Resets a penalty. A negative penalty index creates a new empty penalty.
    penalty_nature() -> str
        Get the nature of the penalty
    """

    class Functions:
        """
        Implementation of all the constraint functions
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

            nlp = all_pn.nlp

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

            contact = all_pn.nlp.contact_forces_func(nlp.states.cx, nlp.controls.cx, nlp.parameters.cx)
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

            slipping = vertcat(
                mu_squared * normal_contact_force_squared - tangential_contact_force_squared,
                mu_squared * normal_contact_force_squared + tangential_contact_force_squared,
            )
            return slipping

        @staticmethod
        def torque_max_from_q_and_qdot(constraint: Constraint, all_pn: PenaltyNodeList, min_torque=None):
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

            nlp = all_pn.nlp
            if min_torque and min_torque < 0:
                raise ValueError("min_torque cannot be negative in tau_max_from_actuators")

            bound = nlp.model.torqueMax(nlp.states["q"].mx, nlp.states["qdot"].mx)
            min_bound = BiorbdInterface.mx_to_cx(
                "min_bound",
                nlp.controls["tau"].mapping.to_first.map(bound[1].to_mx()),
                nlp.states["q"],
                nlp.states["qdot"],
            )
            max_bound = BiorbdInterface.mx_to_cx(
                "max_bound",
                nlp.controls["tau"].mapping.to_first.map(bound[0].to_mx()),
                nlp.states["q"],
                nlp.states["qdot"],
            )
            if min_torque:
                min_bound = if_else(lt(min_bound, min_torque), min_torque, min_bound)
                max_bound = if_else(lt(max_bound, min_torque), min_torque, max_bound)

            value = vertcat(nlp.controls["tau"].cx + min_bound, nlp.controls["tau"].cx - max_bound)

            n_rows = constraint.rows if constraint.rows else int(value.shape[0] / 2)
            constraint.min_bound = [0] * n_rows + [-np.inf] * n_rows
            constraint.max_bound = [np.inf] * n_rows + [0] * n_rows
            return value

        @staticmethod
        def time_constraint(_: Constraint, all_pn: PenaltyNodeList, **unused_param):
            """
            The time constraint is taken care elsewhere, but must be declared here. This function therefore does nothing

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            return all_pn.nlp.tf

        @staticmethod
        def qddot_equals_forward_dynamics(_: Constraint, all_pn: PenaltyNodeList, **unused_param):
            """
            Compute the difference between symbolic joint accelerations and forward dynamic results
            It includes the inversion of mass matrix

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            nlp = all_pn.nlp
            q = nlp.states["q"].mx
            qdot = nlp.states["qdot"].mx
            tau = nlp.states["tau"].mx if "tau" in nlp.states.keys() else nlp.controls["tau"].mx

            qddot = nlp.model.ForwardDynamics(q, qdot, tau).to_mx()

            if "tau" in nlp.states.keys():
                res = BiorbdInterface.mx_to_cx(
                    "ForwardDynamics",
                    nlp.states["qddot"].mx - qddot,
                    nlp.states["q"],
                    nlp.states["qdot"],
                    nlp.states["tau"],
                    nlp.controls["taudot"],
                    nlp.states["qddot"],
                    nlp.controls["qdddot"],
                )

            else:
                res = BiorbdInterface.mx_to_cx(
                    "ForwardDynamics",
                    nlp.controls["qddot"].mx - qddot,
                    nlp.states["q"],
                    nlp.states["qdot"],
                    nlp.controls["tau"],
                    nlp.controls["qddot"],
                )
            return res

        @staticmethod
        def tau_equals_inverse_dynamics(_: Constraint, all_pn: PenaltyNodeList, **unused_param):
            """
            Compute the difference between symbolic joint torques and inverse dynamic results
            It does not include any inversion of mass matrix

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            nlp = all_pn.nlp
            q = nlp.states["q"].mx
            qdot = nlp.states["qdot"].mx
            tau = nlp.states["tau"].mx if "tau" in nlp.states.keys() else nlp.controls["tau"].mx
            qddot = nlp.states["qddot"].mx if "qddot" in nlp.states.keys() else nlp.controls["qddot"].mx

            if nlp.external_forces:
                raise NotImplementedError(
                    "This implicit constraint tau_equals_inverse_dynamics is not implemented yet with external forces"
                )
                # Todo: add fext tau_id = nlp.model.InverseDynamics(q, qdot, qddot, fext).to_mx()
                # fext need to be a mx

            tau_id = nlp.model.InverseDynamics(q, qdot, qddot).to_mx()

            if "tau" in nlp.states.keys():
                res = BiorbdInterface.mx_to_cx(
                    "InverseDynamics",
                    tau_id - tau,
                    nlp.states["q"],
                    nlp.states["qdot"],
                    nlp.states["tau"],
                    nlp.controls["taudot"],
                    nlp.states["qddot"],
                    nlp.controls["qdddot"],
                )

            else:
                res = BiorbdInterface.mx_to_cx(
                    "InverseDynamics",
                    tau_id - tau,
                    nlp.states["q"],
                    nlp.states["qdot"],
                    nlp.controls["tau"],
                    nlp.controls["qddot"],
                )
            return res

        @staticmethod
        def implicit_soft_contact_forces(_: Constraint, all_pn: PenaltyNodeList, **unused_param):
            """
            The time constraint is taken care elsewhere, but must be declared here. This function therefore does nothing

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            nlp = all_pn.nlp

            force_idx = []
            for i_sc in range(nlp.model.nbSoftContacts()):
                force_idx.append(3 + (6 * i_sc))
                force_idx.append(4 + (6 * i_sc))
                force_idx.append(5 + (6 * i_sc))

            soft_contact_all = nlp.soft_contact_forces_func(nlp.states.mx, nlp.controls.mx, nlp.parameters.mx)
            soft_contact_force = soft_contact_all[force_idx]

            var = [nlp.states["q"], nlp.states["qdot"]]
            dynamic_name = nlp.dynamics_type.type.name
            if nlp.dynamics_type.params["implicit_dynamics"]:
                if dynamic_name == "TORQUE_DERIVATIVE_DRIVEN":
                    var.extend(
                        [
                            nlp.states["tau"],
                            nlp.states["taudot"],
                            nlp.controls["fext"],
                            nlp.states["qddot"],
                            nlp.controls["qdddot"],
                        ]
                    )
                elif dynamic_name == "TORQUE_DRIVEN":
                    var.extend([nlp.controls["tau"], nlp.controls["fext"], nlp.controls["qddot"]])
                else:
                    raise NotImplementedError(
                        "Implicit Soft Contact with implicit dynamics has not been implemented yet"
                        f" for such as dynamic: {dynamic_name}. "
                        "It is only implemented for these dynamics:"
                        " TORQUE_DRIVEN and TORQUE_DERIVATIVE_DRIVEN."
                    )
            else:
                if dynamic_name == "TORQUE_DERIVATIVE_DRIVEN":
                    var.extend([
                        nlp.states["tau"],
                        nlp.controls["taudot"],
                        nlp.controls["fext"],
                    ])
                elif dynamic_name == "TORQUE_DRIVEN":
                    var.extend([nlp.controls["tau"], nlp.controls["fext"]])
                else:
                    raise NotImplementedError(
                        "Implicit Soft Contact has not been implemented yet"
                        f" for such as dynamic: {dynamic_name}. "
                        "It is only implemented for these dynamics:"
                        " TORQUE_DRIVEN and TORQUE_DERIVATIVE_DRIVEN."
                    )

            return BiorbdInterface.mx_to_cx("ForwardDynamics", nlp.controls["fext"].mx - soft_contact_force, *var)

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
            penalty = Constraint(
                ConstraintFcn.CONTINUITY, node=Node.ALL_SHOOTING, constraint_type=ConstraintType.INTERNAL
            )
            penalty.add_or_replace_to_penalty_pool(ocp, nlp)

    @staticmethod
    def inter_phase_continuity(ocp):
        """
        Add phase transition constraints between two phases.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """
        for i, pt in enumerate(ocp.phase_transitions):
            # Dynamics must be respected between phases
            pt.name = f"PHASE_TRANSITION {pt.phase_pre_idx}->{pt.phase_post_idx}"
            pt.list_index = -1
            pt.add_or_replace_to_penalty_pool(ocp, ocp.nlp[pt.phase_pre_idx])

    @staticmethod
    def get_dt(_):
        return 1

    @staticmethod
    def penalty_nature() -> str:
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
    NON_SLIPPING = (ConstraintFunction.Functions.non_slipping,)
    TORQUE_MAX_FROM_Q_AND_QDOT = (ConstraintFunction.Functions.torque_max_from_q_and_qdot,)
    TIME_CONSTRAINT = (ConstraintFunction.Functions.time_constraint,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return ConstraintFunction


class ImplicitConstraintFcn(Enum):
    """
    Selection of valid constraint functions

    Methods
    -------
    def get_type() -> Callable
        Returns the type of the penalty
    """

    QDDOT_EQUALS_FORWARD_DYNAMICS = (ConstraintFunction.Functions.qddot_equals_forward_dynamics,)
    TAU_EQUALS_INVERSE_DYNAMICS = (ConstraintFunction.Functions.tau_equals_inverse_dynamics,)
    SOFT_CONTACTS_EQUALS_SOFT_CONTACTS_DYNAMICS = (ConstraintFunction.Functions.implicit_soft_contact_forces,)

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
        ConstraintFunction.inter_phase_continuity(ocp)
