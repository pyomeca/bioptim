from typing import Callable, Any

import numpy as np
from casadi import (
    sum1,
    if_else,
    vertcat,
    lt,
    SX,
    MX,
    jacobian,
    Function,
    MX_eye,
    horzcat,
    ldl,
    diag,
    collocation_points,
)

from .path_conditions import Bounds
from .penalty import PenaltyFunctionAbstract, PenaltyOption, PenaltyController
from ..misc.enums import Node, InterpolationType, PenaltyType, ConstraintType, PhaseDynamics
from ..misc.fcn_enum import FcnEnum
from ..misc.options import OptionList
from ..models.protocols.stochastic_biomodel import StochasticBioModel


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
        min_bound: np.ndarray | float = None,
        max_bound: np.ndarray | float = None,
        quadratic: bool = False,
        phase: int = -1,
        is_stochastic: bool = False,
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
        is_stochastic: bool
            If the constraint is stochastic (if we should instead look at the rate of variation of the inequality
            constraint)
        params:
            Generic parameters for options
        """
        custom_function = None
        if not isinstance(constraint, (ConstraintFcn, ImplicitConstraintFcn)):
            custom_function = constraint
            constraint = ConstraintFcn.CUSTOM

        super(Constraint, self).__init__(
            penalty=constraint,
            phase=phase,
            quadratic=quadratic,
            custom_function=custom_function,
            is_stochastic=is_stochastic,
            **params,
        )

        if isinstance(constraint, ImplicitConstraintFcn):
            self.penalty_type = ConstraintType.IMPLICIT  # doing this puts the relevance of this enum in question

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds("constraints", interpolation=InterpolationType.CONSTANT)

    def set_penalty(self, penalty: MX | SX, controller: PenaltyController):
        super(Constraint, self).set_penalty(penalty, controller)
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
                self.bounds.concatenate(Bounds(None, min_bound, max_bound, interpolation=InterpolationType.CONSTANT))
        elif self.bounds.shape[0] != len(self.rows):
            raise RuntimeError(f"bounds rows is {self.bounds.shape[0]} but should be {self.rows} or empty")

    def _add_penalty_to_pool(self, controller: PenaltyController):
        if isinstance(controller, (list, tuple)):
            controller = controller[0]  # This is a special case of Node.TRANSITION

        if self.penalty_type == PenaltyType.INTERNAL:
            pool = (
                controller.get_nlp.g_internal
                if controller is not None and controller.get_nlp
                else controller.ocp.g_internal
            )
        elif self.penalty_type == ConstraintType.IMPLICIT:
            pool = (
                controller.get_nlp.g_implicit
                if controller is not None and controller.get_nlp
                else controller.ocp.g_implicit
            )
        elif self.penalty_type == PenaltyType.USER:
            pool = controller.get_nlp.g if controller is not None and controller.get_nlp else controller.ocp.g
        else:
            raise ValueError(f"Invalid constraint type {self.penalty_type}.")
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        if self.penalty_type == PenaltyType.INTERNAL:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal
        elif self.penalty_type == ConstraintType.IMPLICIT:
            g_to_add_to = nlp.g_implicit if nlp else ocp.g_implicit
        elif self.penalty_type == PenaltyType.USER:
            g_to_add_to = nlp.g if nlp else ocp.g
        else:
            raise ValueError(f"Invalid Type of Constraint {self.penalty_type}")

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
    add(self, constraint: Callable | "ConstraintFcn", **extra_arguments)
        Add a new Constraint to the list
    print(self)
        Print the ConstraintList to the console
    """

    def add(self, constraint: Callable | Constraint | Any, **extra_arguments: Any):
        """
        Add a new constraint to the list

        Parameters
        ----------
        constraint: Callable | Constraint | ConstraintFcn
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
    inter_node_continuity(ocp)
        Add phase multi node constraints between specified nodes and phases.
    ensure_penalty_sanity(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Constraint)
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
            controller: PenaltyController,
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
            controller: PenaltyController
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

            mu_squared = static_friction_coefficient**2
            constraint.min_bound = np.array([0, 0])
            constraint.max_bound = np.array([np.inf, np.inf])

            contact = controller.get_nlp.contact_forces_func(
                controller.time.cx,
                controller.states.cx_start,
                controller.controls.cx_start,
                controller.parameters.cx,
                controller.stochastic_variables.cx_start,
            )
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
        def torque_max_from_q_and_qdot(
            constraint: Constraint,
            controller: PenaltyController,
            min_torque=None,
        ):
            """
            Nonlinear maximal values of joint torques computed from the torque-position-velocity relationship

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            controller: PenaltyController
                The penalty node elements
            min_torque: float
                Minimum joint torques. This prevent from having too small torques, but introduces an if statement
            """

            if min_torque and min_torque < 0:
                raise ValueError("min_torque cannot be negative in tau_max_from_actuators")

            bound = controller.model.tau_max(controller.states["q"].mx, controller.states["qdot"].mx)
            min_bound = controller.mx_to_cx(
                "min_bound",
                controller.controls["tau"].mapping.to_first.map(bound[1]),
                controller.states["q"],
                controller.states["qdot"],
            )
            max_bound = controller.mx_to_cx(
                "max_bound",
                controller.controls["tau"].mapping.to_first.map(bound[0]),
                controller.states["q"],
                controller.states["qdot"],
            )
            if min_torque:
                min_bound = if_else(lt(min_bound, min_torque), min_torque, min_bound)
                max_bound = if_else(lt(max_bound, min_torque), min_torque, max_bound)

            value = vertcat(
                controller.controls["tau"].cx_start + min_bound, controller.controls["tau"].cx_start - max_bound
            )

            if constraint.rows is None:
                n_rows = value.shape[0] // 2
            else:
                if (
                    controller.get_nlp.phase_dynamics == PhaseDynamics.ONE_PER_NODE
                    and not isinstance(constraint.rows, int)
                    and len(constraint.rows) == value.shape[0]
                ):
                    # This is a very special case where phase_dynamics==ONE_PER_NODE declare rows by itself, but because
                    # this constraint is twice the real length (two constraints per value), it declares it too large
                    # on the subsequent pass. In reality, it means the user did not declare 'rows' by themselves.
                    # Therefore, we are acting as such
                    n_rows = value.shape[0] // 2
                else:
                    if isinstance(constraint.rows, int):
                        n_rows = 1
                    elif isinstance(constraint.rows, (tuple, list)):
                        n_rows = len(constraint.rows)
                    else:
                        raise ValueError("Wrong type for rows")
            constraint.min_bound = [0] * n_rows + [-np.inf] * n_rows
            constraint.max_bound = [np.inf] * n_rows + [0] * n_rows
            return value

        @staticmethod
        def time_constraint(_: Constraint, controller: PenaltyController, **unused_param):
            """
            The time constraint is taken care elsewhere, but must be declared here. This function therefore does nothing

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            controller: PenaltyController
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            return controller.tf

        @staticmethod
        def qddot_equals_forward_dynamics(
            _: Constraint,
            controller: PenaltyController,
            with_contact: bool,
            with_passive_torque: bool,
            with_ligament: bool,
            **unused_param,
        ):
            """
            Compute the difference between symbolic joint accelerations and forward dynamic results
            It includes the inversion of mass matrix

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            controller: PenaltyController
                The penalty node elements
            with_contact: bool
                True if the contact dynamics is handled
            with_passive_torque: bool
                True if the passive torque dynamics is handled
            with_ligament: bool
                True if the ligament dynamics is handled
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            q = controller.states["q"].mx
            qdot = controller.states["qdot"].mx
            passive_torque = controller.model.passive_joint_torque(q, qdot)
            tau = controller.states["tau"].mx if "tau" in controller.states else controller.controls["tau"].mx
            tau = tau + passive_torque if with_passive_torque else tau
            tau = tau + controller.model.ligament_joint_torque(q, qdot) if with_ligament else tau

            qddot = controller.controls["qddot"].mx if "qddot" in controller.controls else controller.states["qddot"].mx
            if with_contact:
                qddot_fd = controller.model.constrained_forward_dynamics(q, qdot, tau)
            else:
                qddot_fd = controller.model.forward_dynamics(q, qdot, tau)

            var = []
            var.extend([controller.states[key] for key in controller.states])
            var.extend([controller.controls[key] for key in controller.controls])
            var.extend([param for param in controller.parameters])

            return controller.mx_to_cx("forward_dynamics", qddot - qddot_fd, *var)

        @staticmethod
        def tau_equals_inverse_dynamics(
            _: Constraint,
            controller: PenaltyController,
            with_contact: bool,
            with_passive_torque: bool,
            with_ligament: bool,
            **unused_param,
        ):
            """
            Compute the difference between symbolic joint torques and inverse dynamic results
            It does not include any inversion of mass matrix

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            controller: PenaltyController
                The penalty node elements
            with_contact: bool
                True if the contact dynamics is handled
            with_passive_torque: bool
                True if the passive torque dynamics is handled
            with_ligament: bool
                True if the ligament dynamics is handled
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            q = controller.states["q"].mx
            qdot = controller.states["qdot"].mx
            tau = controller.states["tau"].mx if "tau" in controller.states else controller.controls["tau"].mx
            qddot = controller.states["qddot"].mx if "qddot" in controller.states else controller.controls["qddot"].mx
            passive_torque = controller.model.passive_joint_torque(q, qdot)
            tau = tau + passive_torque if with_passive_torque else tau
            tau = tau + controller.model.ligament_joint_torque(q, qdot) if with_ligament else tau

            if controller.get_nlp.external_forces:
                raise NotImplementedError(
                    "This implicit constraint tau_equals_inverse_dynamics is not implemented yet with external forces"
                )
                # Todo: add fext tau_id = nlp.model.inverse_dynamics(q, qdot, qddot, fext).to_mx()
            if with_contact:
                # todo: this should be done internally in BiorbdModel
                f_contact = (
                    controller.controls["fext"].mx if "fext" in controller.controls else controller.states["fext"].mx
                )
                f_contact_vec = controller.model.reshape_fext_to_fcontact(f_contact)

                tau_id = controller.model.inverse_dynamics(q, qdot, qddot, None, f_contact_vec)

            else:
                tau_id = controller.model.inverse_dynamics(q, qdot, qddot)

            var = []
            var.extend([controller.states[key] for key in controller.states])
            var.extend([controller.controls[key] for key in controller.controls])
            var.extend([param for param in controller.parameters])

            return controller.mx_to_cx("inverse_dynamics", tau_id - tau, *var)

        @staticmethod
        def implicit_marker_acceleration(
            _: Constraint, controller: PenaltyController, contact_index: int, contact_axis: int, **unused_param
        ):
            """
            Compute the acceleration of the contact node to set it at zero

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            controller: PenaltyController
                The penalty node elements
            contact_index: int
                The contact index
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            q = controller.states["q"].mx
            qdot = controller.states["qdot"].mx
            qddot = controller.states["qddot"].mx if "qddot" in controller.states else controller.controls["qddot"].mx

            # TODO get the index of the marker
            contact_acceleration = controller.model.rigid_contact_acceleration(
                q, qdot, qddot, contact_index, contact_axis
            )

            var = []
            var.extend([controller.states[key] for key in controller.states.keys()])
            var.extend([controller.controls[key] for key in controller.controls.keys()])
            var.extend([controller.parameters[key] for key in controller.parameters.keys()])

            return controller.mx_to_cx("contact_acceleration", contact_acceleration, *var)

        @staticmethod
        def tau_from_muscle_equal_inverse_dynamics(
            _: Constraint, controller: PenaltyController, with_passive_torque: bool, with_ligament: bool, **unused_param
        ):
            """
            Compute the difference between symbolic joint torques from muscle and inverse dynamic results
            It does not include any inversion of mass matrix

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            controller: PenaltyController
                The penalty node elements
            with_passive_torque: bool
                True if the passive torque dynamics is handled
            with_ligament: bool
                True if the ligament dynamics is handled
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            q = controller.states["q"].mx
            qdot = controller.states["qdot"].mx
            muscle_activations = controller.controls["muscles"].mx
            muscles_states = controller.model.state_set()
            passive_torque = controller.model.passive_joint_torque(q, qdot)
            for k in range(len(controller.controls["muscles"])):
                muscles_states[k].setActivation(muscle_activations[k])
            muscle_tau = controller.model.muscle_joint_torque(muscles_states, q, qdot)
            muscle_tau = muscle_tau + passive_torque if with_passive_torque else muscle_tau
            muscle_tau = muscle_tau + controller.model.ligament_joint_torque(q, qdot) if with_ligament else muscle_tau
            qddot = controller.states["qddot"].mx if "qddot" in controller.states else controller.controls["qddot"].mx

            if controller.get_nlp.external_forces:
                raise NotImplementedError(
                    "This implicit constraint tau_from_muscle_equal_inverse_dynamics is not implemented yet with external forces"
                )
                # Todo: add fext tau_id = nlp.model.inverse_dynamics(q, qdot, qddot, fext).to_mx()
                # fext need to be a mx

            tau_id = controller.model.inverse_dynamics(q, qdot, qddot)

            var = []
            var.extend([controller.states[key] for key in controller.states])
            var.extend([controller.controls[key] for key in controller.controls])
            var.extend([param for param in controller.parameters])

            return controller.mx_to_cx("inverse_dynamics", tau_id - muscle_tau, *var)

        @staticmethod
        def implicit_soft_contact_forces(_: Constraint, controller: PenaltyController, **unused_param):
            """
            Compute the difference between symbolic soft contact forces and actual force contact dynamic

            Parameters
            ----------
            _: Constraint
                The actual constraint to declare
            controller: PenaltyController
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            force_idx = []
            for i_sc in range(controller.model.nb_soft_contacts):
                force_idx.append(3 + (6 * i_sc))
                force_idx.append(4 + (6 * i_sc))
                force_idx.append(5 + (6 * i_sc))

            soft_contact_all = controller.get_nlp.soft_contact_forces_func(
                controller.states.mx, controller.controls.mx, controller.parameters.mx
            )
            soft_contact_force = soft_contact_all[force_idx]

            var = []
            var.extend([controller.states[key] for key in controller.states])
            var.extend([controller.controls[key] for key in controller.controls])
            var.extend([param for param in controller.parameters])

            return controller.mx_to_cx("forward_dynamics", controller.controls["fext"].mx - soft_contact_force, *var)

        @staticmethod
        def stochastic_covariance_matrix_continuity_implicit(
            penalty: PenaltyOption,
            controller: PenaltyController,
        ):
            """
            This functions constrain the covariance matrix to its actual value as in Gillis 2013.
            It is explained in more details here: https://doi.org/10.1109/CDC.2013.6761121
            P_k+1 = M_k @ (dg/dx @ P @ dg/dx + dg/dw @ sigma_w @ dg/dw) @ M_k
            """
            # TODO: Charbie -> This is only True for x=[q, qdot], u=[tau] (have to think on how to generalize it)

            if not controller.get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")

            if "cholesky_cov" in controller.stochastic_variables.keys():
                l_cov_matrix = StochasticBioModel.reshape_to_cholesky_matrix(
                    controller.stochastic_variables["cholesky_cov"].cx_start, controller.model.matrix_shape_cov_cholesky
                )
                cov_matrix = l_cov_matrix @ l_cov_matrix.T
            else:
                cov_matrix = StochasticBioModel.reshape_to_matrix(
                    controller.stochastic_variables["cov"].cx_start, controller.model.matrix_shape_cov
                )
            a_matrix = StochasticBioModel.reshape_to_matrix(
                controller.stochastic_variables["a"].cx_start, controller.model.matrix_shape_a
            )
            c_matrix = StochasticBioModel.reshape_to_matrix(
                controller.stochastic_variables["c"].cx_start, controller.model.matrix_shape_c
            )
            m_matrix = StochasticBioModel.reshape_to_matrix(
                controller.stochastic_variables["m"].cx_start, controller.model.matrix_shape_m
            )

            sigma_w = vertcat(
                controller.model.sensory_noise_magnitude, controller.model.motor_noise_magnitude
            ) * MX_eye(
                vertcat(controller.model.sensory_noise_magnitude, controller.model.motor_noise_magnitude).shape[0]
            )
            dt = controller.tf / controller.ns
            dg_dw = -dt * c_matrix
            dg_dx = -MX_eye(a_matrix.shape[0]) - dt / 2 * a_matrix

            cov_next = m_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ m_matrix.T
            cov_implicit_deffect = cov_next - cov_matrix

            penalty.expand = (
                controller.get_nlp.dynamics_type.expand_dynamics
            )  # TODO: Charbie -> should this be always true?
            penalty.explicit_derivative = True
            penalty.multi_thread = True

            out_vector = StochasticBioModel.reshape_to_vector(cov_implicit_deffect)
            return out_vector

        @staticmethod
        def stochastic_df_dx_implicit(
            penalty: Constraint,
            controller: PenaltyController,
        ):
            """
            This function constrains the stochastic matrix A to its actual value which is
            A = df/dx
            """
            if not controller.get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")

            dt = controller.tf / controller.ns

            nb_root = controller.model.nb_root
            # TODO: Charbie -> This is only True for x=[q, qdot], u=[tau] (have to think on how to generalize it)
            nu = controller.model.nb_q - controller.model.nb_root

            a_matrix = StochasticBioModel.reshape_to_matrix(
                controller.stochastic_variables["a"].cx_start, controller.model.matrix_shape_a
            )

            q_root = MX.sym("q_root", nb_root, 1)
            q_joints = MX.sym("q_joints", nu, 1)
            qdot_root = MX.sym("qdot_root", nb_root, 1)
            qdot_joints = MX.sym("qdot_joints", nu, 1)
            tau_joints = MX.sym("tau_joints", nu, 1)
            parameters_sym = MX.sym("parameters_sym", controller.parameters.shape, 1)
            stochastic_sym = MX.sym("stochastic_sym", controller.stochastic_variables.shape, 1)

            dx = controller.extra_dynamics(0)(
                controller.time.mx,
                vertcat(q_root, q_joints, qdot_root, qdot_joints),  # States
                tau_joints,
                parameters_sym,
                stochastic_sym,
            )

            non_root_index = list(range(nb_root, nb_root + nu)) + list(
                range(nb_root + nu + nb_root, nb_root + nu + nb_root + nu)
            )
            DF_DX_fun = Function(
                "DF_DX_fun",
                [
                    controller.time.mx,
                    q_root,
                    q_joints,
                    qdot_root,
                    qdot_joints,
                    tau_joints,
                    parameters_sym,
                    stochastic_sym,
                    controller.model.motor_noise_sym,
                    controller.model.sensory_noise_sym,
                ],
                [jacobian(dx[non_root_index], vertcat(q_joints, qdot_joints))],
            )

            DF_DX = DF_DX_fun(
                controller.time.cx,
                controller.states["q"].cx_start[:nb_root],
                controller.states["q"].cx_start[nb_root:],
                controller.states["qdot"].cx_start[:nb_root],
                controller.states["qdot"].cx_start[nb_root:],
                controller.controls.cx_start,
                controller.parameters.cx_start,
                controller.stochastic_variables.cx_start,
                controller.model.motor_noise_magnitude,
                controller.model.sensory_noise_magnitude,
            )

            out = a_matrix - (MX_eye(DF_DX.shape[0]) - DF_DX * dt / 2)

            out_vector = StochasticBioModel.reshape_to_vector(out)
            return out_vector

        @staticmethod
        def stochastic_helper_matrix_collocation(
            penalty: Constraint,
            controller: PenaltyController,
        ):
            """
            This function constrains the stochastic matrix M to its actual value which is
            dF/dz.T = dG/dz.T @ M.T
            where z = states at the collocation points, F = collocation continuity constraint (dxdt - x_k+1),
            and G = collocation slope constraints (defects).
            """

            if not controller.get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")

            polynomial_degree = controller.get_nlp.ode_solver.polynomial_degree
            Mc, _ = ConstraintFunction.Functions.collocation_jacobians(
                penalty,
                controller,
                polynomial_degree,
            )

            nb_root = controller.model.nb_root
            nu = controller.model.nb_q - nb_root
            z_joints = horzcat(*(controller.states.cx_intermediates_list))

            constraint = Mc(
                controller.time_cx,
                controller.states.cx_start[:nb_root],  # x_q_root
                controller.states.cx_start[nb_root : nb_root + nu],  # x_q_joints
                controller.states.cx_start[nb_root + nu : 2 * nb_root + nu],  # x_qdot_root
                controller.states.cx_start[2 * nb_root + nu : 2 * (nb_root + nu)],  # x_qdot_joints
                z_joints[:nb_root, :],  # z_q_root
                z_joints[nb_root : nb_root + nu, :],  # z_q_joints
                z_joints[nb_root + nu : 2 * nb_root + nu, :],  # z_qdot_root
                z_joints[2 * nb_root + nu : 2 * (nb_root + nu), :],  # z_qdot_joints
                controller.controls.cx_start,
                controller.parameters.cx_start,
                controller.stochastic_variables.cx_start,
                controller.model.motor_noise_magnitude,
                controller.model.sensory_noise_magnitude,
            )

            return StochasticBioModel.reshape_to_vector(constraint)

        @staticmethod
        def stochastic_covariance_matrix_continuity_collocation(
            penalty,
            controller: PenaltyController,
        ):
            """
            This functions allows to implicitly integrate the covariance matrix as in Gillis 2013.
            It is explained in more details here: https://doi.org/10.1109/CDC.2013.6761121
            P_k+1 = M_k @ (dg/dx @ P_k @ dg/dx + dg/dw @ sigma_w @ dg/dw) @ M_k
            """
            # TODO: Charbie -> This is only True for x=[q, qdot], u=[tau] (have to think on how to generalize it)

            if not controller.get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")

            polynomial_degree = controller.get_nlp.ode_solver.polynomial_degree
            _, Pf = ConstraintFunction.Functions.collocation_jacobians(
                penalty,
                controller,
                polynomial_degree,
            )

            cov_matrix_next = StochasticBioModel.reshape_to_matrix(
                controller.stochastic_variables["cov"].cx_end, controller.model.matrix_shape_cov
            )

            nb_root = controller.model.nb_root
            nu = controller.model.nb_q - nb_root
            z_joints = horzcat(*(controller.states.cx_intermediates_list))

            cov_next_computed = Pf(
                controller.time_cx,
                controller.states.cx_start[:nb_root],  # x_q_root
                controller.states.cx_start[nb_root : nb_root + nu],  # x_q_joints
                controller.states.cx_start[nb_root + nu : 2 * nb_root + nu],  # x_qdot_root
                controller.states.cx_start[2 * nb_root + nu : 2 * (nb_root + nu)],  # x_qdot_joints
                z_joints[:nb_root, :],  # z_q_root
                z_joints[nb_root : nb_root + nu, :],  # z_q_joints
                z_joints[nb_root + nu : 2 * nb_root + nu, :],  # z_qdot_root
                z_joints[2 * nb_root + nu : 2 * (nb_root + nu), :],  # z_qdot_joints
                controller.controls.cx_start,
                controller.parameters.cx_start,
                controller.stochastic_variables.cx_start,
                controller.model.motor_noise_magnitude,
                controller.model.sensory_noise_magnitude,
            )

            cov_implicit_defect = cov_matrix_next - cov_next_computed

            out_vector = StochasticBioModel.reshape_to_vector(cov_implicit_defect)

            penalty.explicit_derivative = True
            penalty.multi_thread = True

            return out_vector

        @staticmethod
        def stochastic_mean_sensory_input_equals_reference(
            penalty: Constraint,
            controller: PenaltyController,
        ):
            """
            Get the error between the hand position and the reference.
            """
            ref = controller.stochastic_variables["ref"].cx_start
            sensory_input = controller.model.sensory_reference(
                states=controller.states.cx_start,
                controls=controller.controls.cx_start,
                parameters=controller.parameters.cx_start,
                stochastic_variables=controller.stochastic_variables.cx_start,
                nlp=controller.get_nlp,
            )
            return sensory_input - ref

        @staticmethod
        def symmetric_matrix(
            penalty: Constraint,
            controller: PenaltyController,
            key: str,
        ):
            """
            This function constrains a matrix to be symmetric
            """
            variable = controller.stochastic_variables[key].cx_start
            if np.sqrt(variable.shape[0]) % 1 != 0:
                raise RuntimeError(f"The matrix {key} is not square")
            else:
                matrix_shape = int(np.sqrt(variable.shape[0]))
            matrix = StochasticBioModel.reshape_to_matrix(variable, (matrix_shape, matrix_shape))
            symmetry_constraint = matrix - matrix.T
            return StochasticBioModel.reshape_to_vector(symmetry_constraint)

        @staticmethod
        def semidefinite_positive_matrix(
            penalty: Constraint,
            controller: PenaltyController,
            key: str,
        ):
            """
            This function constrains a matrix to be semi-definite positive.
            """
            variable = controller.stochastic_variables[key].cx_start
            if np.sqrt(variable.shape[0]) % 1 != 0:
                raise RuntimeError(f"The matrix {key} is not square")
            else:
                matrix_shape = int(np.sqrt(variable.shape[0]))

            A = SX.sym("A", matrix_shape, matrix_shape)
            D = ldl(A)[0]  # Only guaranteed to work by casadi for positive definite matrix.
            func = Function("diagonal_terms", [A], [D])

            matrix = StochasticBioModel.reshape_to_matrix(variable, (matrix_shape, matrix_shape))
            diagonal_terms = func(matrix)

            return diagonal_terms

        @staticmethod
        def collocation_jacobians(penalty, controller, polynomial_degree):
            """
            This function computes the jacobians of the collocation equation and of the continuity equation with respect to the collocation points and the noise
            """
            sigma_ww = diag(vertcat(controller.model.motor_noise_sym, controller.model.sensory_noise_sym))

            cov_matrix = StochasticBioModel.reshape_to_matrix(
                controller.stochastic_variables["cov"].cx_start, controller.model.matrix_shape_cov
            )
            m_matrix = StochasticBioModel.reshape_to_matrix(
                controller.stochastic_variables["m"].cx_start, controller.model.matrix_shape_m
            )

            nb_root = controller.model.nb_root
            nu = controller.model.nb_q - nb_root
            joints_index = list(range(nb_root, nb_root + nu)) + list(range(2 * nb_root + nu, 2 * (nb_root + nu)))

            x_q_root = controller.cx.sym("x_q_root", nb_root, 1)
            x_q_joints = controller.cx.sym("x_q_joints", nu, 1)
            x_qdot_root = controller.cx.sym("x_qdot_root", nb_root, 1)
            x_qdot_joints = controller.cx.sym("x_qdot_joints", nu, 1)
            z_q_root = controller.cx.sym("z_q_root", nb_root, polynomial_degree + 1)
            z_q_joints = controller.cx.sym("z_q_joints", nu, polynomial_degree + 1)
            z_qdot_root = controller.cx.sym("z_qdot_root", nb_root, polynomial_degree + 1)
            z_qdot_joints = controller.cx.sym("z_qdot_joints", nu, polynomial_degree + 1)

            x_full = vertcat(x_q_root, x_q_joints, x_qdot_root, x_qdot_joints)
            z_full = vertcat(z_q_root, z_q_joints, z_qdot_root, z_qdot_joints)

            xf, xall, defects = controller.integrate_extra_dynamics(0).function(
                controller.time.cx,
                horzcat(x_full, z_full),
                controller.controls.cx_start,
                controller.parameters.cx_start,
                controller.stochastic_variables.cx_start,
            )
            G_joints = [x_full - z_full[:, 0]]
            nx = 2 * (nb_root + nu)
            for i in range(controller.ode_solver.polynomial_degree):
                idx = [j + i * nx for j in joints_index]
                G_joints.append(defects[idx])
            G_joints = vertcat(*G_joints)

            Gdx = jacobian(G_joints, horzcat(x_q_joints, x_qdot_joints))
            Gdz = jacobian(G_joints, horzcat(z_q_joints, z_qdot_joints))
            Gdw = jacobian(
                G_joints,
                vertcat(controller.model.motor_noise_sym, controller.model.sensory_noise_sym),
            )
            Fdz = jacobian(xf, horzcat(z_q_joints, z_qdot_joints))

            # Constraint Equality defining M
            Mc = Function(
                "M_cons",
                [
                    controller.time_cx,
                    x_q_root,
                    x_q_joints,
                    x_qdot_root,
                    x_qdot_joints,
                    z_q_root,
                    z_q_joints,
                    z_qdot_root,
                    z_qdot_joints,
                    controller.controls.cx_start,
                    controller.parameters.cx_start,
                    controller.stochastic_variables.cx_start,
                    controller.model.motor_noise_sym,
                    controller.model.sensory_noise_sym,
                ],
                [Fdz.T - Gdz.T @ m_matrix.T],
            )
            if penalty.expand:
                Mc = Mc.expand()

            # Covariance propagation rule
            Pf = Function(
                "P_next",
                [
                    controller.time_cx,
                    x_q_root,
                    x_q_joints,
                    x_qdot_root,
                    x_qdot_joints,
                    z_q_root,
                    z_q_joints,
                    z_qdot_root,
                    z_qdot_joints,
                    controller.controls.cx_start,
                    controller.parameters.cx_start,
                    controller.stochastic_variables.cx_start,
                    controller.model.motor_noise_sym,
                    controller.model.sensory_noise_sym,
                ],
                [m_matrix @ (Gdx @ cov_matrix @ Gdx.T + Gdw @ sigma_ww @ Gdw.T) @ m_matrix.T],
            )
            if penalty.expand:
                Pf = Pf.expand()

            return Mc, Pf

    @staticmethod
    def get_dt(_):
        return 1

    @staticmethod
    def penalty_nature() -> str:
        return "constraints"


class ConstraintFcn(FcnEnum):
    """
    Selection of valid constraint functions

    Methods
    -------
    def get_type() -> Callable
        Returns the type of the penalty
    """

    STATE_CONTINUITY = (PenaltyFunctionAbstract.Functions.state_continuity,)
    FIRST_COLLOCATION_HELPER_EQUALS_STATE = (PenaltyFunctionAbstract.Functions.first_collocation_point_equals_state,)
    CUSTOM = (PenaltyFunctionAbstract.Functions.custom,)
    NON_SLIPPING = (ConstraintFunction.Functions.non_slipping,)
    PROPORTIONAL_CONTROL = (PenaltyFunctionAbstract.Functions.proportional_controls,)
    PROPORTIONAL_STATE = (PenaltyFunctionAbstract.Functions.proportional_states,)
    TRACK_STOCHASTIC = (PenaltyFunctionAbstract.Functions.stochastic_minimize_variables,)
    STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_IMPLICIT = (
        ConstraintFunction.Functions.stochastic_covariance_matrix_continuity_implicit,
    )
    STOCHASTIC_DF_DX_IMPLICIT = (ConstraintFunction.Functions.stochastic_df_dx_implicit,)
    STOCHASTIC_HELPER_MATRIX_COLLOCATION = (ConstraintFunction.Functions.stochastic_helper_matrix_collocation,)
    STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION = (
        ConstraintFunction.Functions.stochastic_covariance_matrix_continuity_collocation,
    )

    STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE = (
        ConstraintFunction.Functions.stochastic_mean_sensory_input_equals_reference,
    )
    SYMMETRIC_MATRIX = (ConstraintFunction.Functions.symmetric_matrix,)
    SEMIDEFINITE_POSITIVE_MATRIX = (ConstraintFunction.Functions.semidefinite_positive_matrix,)
    SUPERIMPOSE_MARKERS = (PenaltyFunctionAbstract.Functions.superimpose_markers,)
    SUPERIMPOSE_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.superimpose_markers_velocity,)
    TIME_CONSTRAINT = (ConstraintFunction.Functions.time_constraint,)
    TORQUE_MAX_FROM_Q_AND_QDOT = (ConstraintFunction.Functions.torque_max_from_q_and_qdot,)
    TRACK_ANGULAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_angular_momentum,)
    TRACK_COM_POSITION = (PenaltyFunctionAbstract.Functions.minimize_com_position,)
    TRACK_COM_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_com_velocity,)
    TRACK_CONTACT_FORCES = (PenaltyFunctionAbstract.Functions.minimize_contact_forces,)
    TRACK_CONTROL = (PenaltyFunctionAbstract.Functions.minimize_controls,)
    TRACK_LINEAR_MOMENTUM = (PenaltyFunctionAbstract.Functions.minimize_linear_momentum,)
    TRACK_MARKER_WITH_SEGMENT_AXIS = (PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis,)
    TRACK_MARKERS = (PenaltyFunctionAbstract.Functions.minimize_markers,)
    TRACK_MARKERS_ACCELERATION = (PenaltyFunctionAbstract.Functions.minimize_markers_acceleration,)
    TRACK_MARKERS_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_markers_velocity,)
    TRACK_PARAMETER = (PenaltyFunctionAbstract.Functions.minimize_parameter,)
    TRACK_POWER = (PenaltyFunctionAbstract.Functions.minimize_power,)
    TRACK_QDDOT = (PenaltyFunctionAbstract.Functions.minimize_qddot,)
    TRACK_SEGMENT_ROTATION = (PenaltyFunctionAbstract.Functions.minimize_segment_rotation,)
    TRACK_SEGMENT_VELOCITY = (PenaltyFunctionAbstract.Functions.minimize_segment_velocity,)
    TRACK_SEGMENT_WITH_CUSTOM_RT = (PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt,)
    TRACK_STATE = (PenaltyFunctionAbstract.Functions.minimize_states,)
    TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS = (PenaltyFunctionAbstract.Functions.track_vector_orientations_from_markers,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return ConstraintFunction


class ImplicitConstraintFcn(FcnEnum):
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
    CONTACT_ACCELERATION_EQUALS_ZERO = (ConstraintFunction.Functions.implicit_marker_acceleration,)
    TAU_FROM_MUSCLE_EQUAL_INVERSE_DYNAMICS = (ConstraintFunction.Functions.tau_from_muscle_equal_inverse_dynamics,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return ConstraintFunction


class ParameterConstraint(PenaltyOption):
    """
    A placeholder for a parameter constraint

    Attributes
    ----------
    min_bound: np.ndarray
        The vector of minimum bound of the constraint. Default is 0
    max_bound: np.ndarray
        The vector of maximal bound of the constraint. Default is 0
    """

    def __init__(
        self,
        parameter_constraint: Any,
        min_bound: np.ndarray | float = None,
        max_bound: np.ndarray | float = None,
        quadratic: bool = False,
        **params: Any,
    ):
        """
        Parameters
        ----------
        parameter_constraint: ConstraintFcn
            The chosen parameter constraint
        min_bound: np.ndarray
            The vector of minimum bound of the constraint. Default is 0
        max_bound: np.ndarray
            The vector of maximal bound of the constraint. Default is 0
        quadratic: bool
            If the penalty is quadratic
        params:
            Generic parameters for options
        """
        custom_function = None
        if not isinstance(parameter_constraint, (ConstraintFcn, ImplicitConstraintFcn)):
            custom_function = parameter_constraint
            parameter_constraint = ConstraintFcn.CUSTOM

        super(ParameterConstraint, self).__init__(
            penalty=parameter_constraint, quadratic=quadratic, custom_function=custom_function, **params
        )

        if isinstance(parameter_constraint, ImplicitConstraintFcn):
            self.penalty_type = ConstraintType.IMPLICIT  # doing this puts the relevance of this enum in question

        self.min_bound = min_bound
        self.max_bound = max_bound
        # TODO Benjamin Check .name
        self.bounds = Bounds(parameter_constraint.name, interpolation=InterpolationType.CONSTANT)

    def set_penalty(self, penalty: MX | SX, controller: PenaltyController):
        super(ParameterConstraint, self).set_penalty(penalty, controller)
        self.min_bound = 0 if self.min_bound is None else self.min_bound
        self.max_bound = 0 if self.max_bound is None else self.max_bound

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        if self.type == ConstraintFcn.TIME_CONSTRAINT:
            self.node = Node.END

        super(ParameterConstraint, self).add_or_replace_to_penalty_pool(ocp, nlp)

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

    def _add_penalty_to_pool(self, controller: PenaltyController):
        if isinstance(controller, (list, tuple)):
            controller = controller[0]  # This is a special case of Node.TRANSITION

        if self.penalty_type == PenaltyType.INTERNAL:
            pool = (
                controller.get_nlp.g_internal
                if controller is not None and controller.get_nlp
                else controller.ocp.g_internal
            )
        elif self.penalty_type == ConstraintType.IMPLICIT:
            pool = (
                controller.get_nlp.g_implicit
                if controller is not None and controller.get_nlp
                else controller.ocp.g_implicit
            )
        elif self.penalty_type == PenaltyType.USER:
            pool = controller.get_nlp.g if controller is not None and controller.get_nlp else controller.ocp.g
        else:
            raise ValueError(f"Invalid constraint type {self.penalty_type}.")
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        if self.penalty_type == PenaltyType.INTERNAL:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal
        elif self.penalty_type == ConstraintType.IMPLICIT:
            g_to_add_to = nlp.g_implicit if nlp else ocp.g_implicit
        elif self.penalty_type == PenaltyType.USER:
            g_to_add_to = nlp.g if nlp else ocp.g
        else:
            raise ValueError(f"Invalid Type of Constraint {self.penalty_type}")

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


class ParameterConstraintList(OptionList):
    """
    A list of ParameterConstraint if more than one is required

    Methods
    -------
    add(self, parameter_constraint: Callable | "ConstraintFcn", **extra_arguments)
        Add a new Constraint to the list
    print(self)
        Print the ParameterConstraintList to the console
    """

    def add(self, parameter_constraint: Callable | ParameterConstraint | Any, **extra_arguments: Any):
        """
        Add a new constraint to the list

        Parameters
        ----------
        parameter_constraint: Callable | Constraint | ConstraintFcn
            The chosen parameter constraint
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if isinstance(parameter_constraint, Constraint):
            self.copy(parameter_constraint)

        else:
            super(ParameterConstraintList, self)._add(
                option_type=ParameterConstraint, parameter_constraint=parameter_constraint, **extra_arguments
            )

    def print(self):
        """
        Print the ParameterConstraintList to the console
        """
        # TODO: Print all elements in the console
        raise NotImplementedError("Printing of ParameterConstraintList is not ready yet")
