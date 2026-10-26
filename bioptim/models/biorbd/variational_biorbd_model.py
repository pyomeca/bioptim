"""
Biorbd model for holonomic constraints and variational integrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from casadi import SX, MX, DM, vertcat, jacobian, transpose

from .holonomic_biorbd_model import HolonomicBiorbdModel
from ...misc.enums import ControlType, QuadratureRule
from ...optimization.parameters import ParameterList
from ...misc.parameters_types import Str, Float

if TYPE_CHECKING:
    import biorbd_casadi as biorbd


class VariationalBiorbdModel(HolonomicBiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints and the methods for the variational
    integrator, very experimental and not tested.
    """

    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        parameters: ParameterList = None,
        **kwargs,
    ):
        """
        Initialize the VariationalBiorbdModel. The discretization (quadrature rules and control type) is chosen by
        OdeSolver.VARIATIONAL and the ocp, and given to the methods below.

        Parameters
        ----------
        bio_model : Str | biorbd.Model
            The path to the biorbd model file or a biorbd.Model instance.
        parameters : ParameterList, optional
            A list of parameters for the model (default is None).
        """
        super().__init__(bio_model=bio_model, parameters=parameters, **kwargs)

    def discrete_lagrangian(
        self,
        q1: MX | SX | DM,
        q2: MX | SX | DM,
        time_step: MX | SX | DM,
        discrete_approximation: QuadratureRule,
    ) -> MX | SX | DM:
        """
        Compute the discrete Lagrangian of a biorbd model.

        Parameters
        ----------
        q1: MX | SX
            The generalized coordinates at the first time step.
        q2: MX | SX
            The generalized coordinates at the second time step.
        time_step: float
            The time step.
        discrete_approximation: QuadratureRule
            The quadrature rule of the discrete Lagrangian.

        Returns
        -------
        The discrete Lagrangian.
        """
        if discrete_approximation == QuadratureRule.MIDPOINT:
            q_discrete = (q1 + q2) / 2
            qdot_discrete = (q2 - q1) / time_step
            return time_step * self.lagrangian()(q_discrete, qdot_discrete)
        elif discrete_approximation == QuadratureRule.RECTANGLE_LEFT:
            q_discrete = q1
            qdot_discrete = (q2 - q1) / time_step
            return time_step * self.lagrangian()(q_discrete, qdot_discrete)
        elif discrete_approximation == QuadratureRule.RECTANGLE_RIGHT:
            q_discrete = q2
            qdot_discrete = (q2 - q1) / time_step
            return time_step * self.lagrangian()(q_discrete, qdot_discrete)
        elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
            # from : M. West, “Variational integrators,” Ph.D. dissertation, California Inst.
            # Technol., Pasadena, CA, 2004. p 13
            qdot_discrete = (q2 - q1) / time_step
            return time_step / 2 * (self.lagrangian()(q1, qdot_discrete) + self.lagrangian()(q2, qdot_discrete))
        else:
            raise NotImplementedError(f"Discrete Lagrangian {discrete_approximation} is not implemented")

    def control_approximation(
        self,
        control_minus: MX | SX,
        control_plus: MX | SX,
        time_step: Float,
        control_type: ControlType,
        control_discrete_approximation: QuadratureRule,
    ):
        """
        Compute the term associated to the discrete forcing. The term associated to the controls in the Lagrangian
        equations is homogeneous to a force or a torque multiplied by a time.

        Parameters
        ----------
        control_minus: MX | SX
            Control at t_k (or t{k-1})
        control_plus: MX | SX
            Control at t_{k+1} (or tk)
        time_step: float
            The time step.
        control_type: ControlType
            The type of the controls. Only the controls of the first node are used by the constant ones.
        control_discrete_approximation: QuadratureRule
            The quadrature rule of the discrete forcing, only used by ControlType.LINEAR_CONTINUOUS.

        Returns
        ----------
        The term associated to the controls in the Lagrangian equations.

        Sources
        -------
        Johnson, E. R., & Murphey, T. D. (2009).
        Scalable Variational Integrators for Constrained Mechanical Systems in Generalized Coordinates.
        IEEE Transactions on Robotics, 25(6), 1249–1261. doi:10.1109/tro.2009.2032955
        """
        if control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
            return 1 / 2 * control_minus * time_step

        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if control_discrete_approximation == QuadratureRule.MIDPOINT:
                return 1 / 2 * (control_minus + control_plus) / 2 * time_step
            elif control_discrete_approximation == QuadratureRule.RECTANGLE_LEFT:
                return 1 / 2 * control_minus * time_step
            elif control_discrete_approximation == QuadratureRule.RECTANGLE_RIGHT:
                return 1 / 2 * control_plus * time_step
            else:
                raise NotImplementedError(
                    f"Discrete {control_discrete_approximation} is not implemented for {control_type} for "
                    f"VariationalBiorbdModel"
                )
        else:
            raise RuntimeError(f"The control type {control_type} is not implemented for the VariationalBiorbdModel. ")

    def discrete_holonomic_constraints_jacobian(self, time_step: MX | SX, q: MX | SX) -> MX | SX | None:
        """
        Compute the discrete Jacobian of the holonomic constraints. See Variational integrators for constrained
        dynamical systems (https://onlinelibrary.wiley.com/doi/epdf/10.1002/zamm.200700173) eq. (21) for more
        precisions.

        Parameters
        ----------
        time_step: MX | SX
            The time step.
        q:
            The coordinates.

        Returns
        -------
        holonomic_discrete_constraints_jacobian: MX | SX | None
            The discrete Jacobian of the holonomic constraints if there is constraints, None otherwise.
        """
        if self.holonomic_constraints_jacobian is None:
            return None
        return time_step * self.holonomic_constraints_jacobian(q)

    def discrete_euler_lagrange_equations(
        self,
        time_step: MX | SX,
        q_prev: MX | SX,
        q_cur: MX | SX,
        q_next: MX | SX,
        control_prev: MX | SX,
        control_cur: MX | SX,
        control_next: MX | SX,
        lambdas: MX | SX = None,
        *,
        discrete_approximation: QuadratureRule,
        control_type: ControlType,
        control_discrete_approximation: QuadratureRule,
    ) -> MX | SX:
        """
        Compute the discrete Euler-Lagrange equations of a biorbd model

        Parameters
        ----------
        time_step: MX | SX
            The time step.
        q_prev: MX | SX
            The generalized coordinates at the first node.
        q_cur: MX | SX
            The generalized coordinates at the second node.
        q_next: MX | SX
            The generalized coordinates at the third node.
        control_prev: MX | SX
            The generalized forces at the first node.
        control_cur: MX | SX
            The generalized forces at the second node.
        control_next: MX | SX
            The generalized forces at the third node.
        lambdas: MX | SX
            The Lagrange multipliers.
        discrete_approximation: QuadratureRule
            The quadrature rule of the discrete Lagrangian.
        control_type: ControlType
            The type of the controls.
        control_discrete_approximation: QuadratureRule
            The quadrature rule of the discrete forcing.

        Returns
        -------
        MX | SX
            The discrete Euler-Lagrange equations.

        Sources
        -------
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (10).
        """
        CX = MX if isinstance(q_prev, MX) else SX

        # Refers to D_2 L_d(q_{k-1}, q_k) (D_2 is the partial derivative with respect to the second argument, L_d is the
        # discrete Lagrangian)
        p_current = transpose(
            jacobian(self.discrete_lagrangian(q_prev, q_cur, time_step, discrete_approximation), q_cur)
        )
        # Refers to D_1 L_d(q_{k}, q_{k+1}) (D_2 is the partial derivative with respect to the second argument)
        d1_ld_qcur_qnext = transpose(
            jacobian(self.discrete_lagrangian(q_cur, q_next, time_step, discrete_approximation), q_cur)
        )
        if self.has_holonomic_constraints and lambdas is None:
            raise ValueError("As your model is constrained, you must specify the lambdas.")
        constraint_term = (
            transpose(self.discrete_holonomic_constraints_jacobian(time_step, q_cur)) @ lambdas
            if self.has_holonomic_constraints
            else CX.zeros(p_current.shape)
        )

        residual = (
            p_current
            + d1_ld_qcur_qnext
            - constraint_term
            + self.control_approximation(
                control_prev, control_cur, time_step, control_type, control_discrete_approximation
            )
            + self.control_approximation(
                control_cur, control_next, time_step, control_type, control_discrete_approximation
            )
        )

        if self.has_holonomic_constraints:
            return vertcat(residual, self.holonomic_constraints(q_next))
        else:
            return residual

    def compute_initial_states(
        self,
        time_step: MX | SX,
        q0: MX | SX,
        qdot0: MX | SX,
        q1: MX | SX,
        control0: MX | SX,
        control1: MX | SX,
        lambdas0: MX | SX = None,
        *,
        discrete_approximation: QuadratureRule,
        control_type: ControlType,
        control_discrete_approximation: QuadratureRule,
    ):
        """
        Parameters
        ----------
        time_step: MX | SX
            The time step.
        q0: MX | SX
            The generalized coordinates at the first node.
        qdot0: MX | SX
            The initial generalized velocities at the first node.
        q1: MX | SX
            The generalized coordinates at the second node.
        control0: MX | SX
            The generalized forces at the first node.
        control1: MX | SX
            The generalized forces at the second node.
        lambdas0: MX | SX
            The Lagrange multipliers at the first node.
        discrete_approximation: QuadratureRule
            The quadrature rule of the discrete Lagrangian.
        control_type: ControlType
            The type of the controls.
        control_discrete_approximation: QuadratureRule
            The quadrature rule of the discrete forcing.

        Returns
        -------
        MX | SX
            The discrete Euler-Lagrange equations adapted for the first node.

        Sources
        -------
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        indications given just before the equation (18) for p0 and pN.
        """
        CX = MX if isinstance(q0, MX) else SX

        # Refers to D_2 L(q_0, \dot{q_0}) (D_2 is the partial derivative with respect to the second argument)
        d2_l_q0_qdot0 = transpose(jacobian(self.lagrangian()(q0, qdot0), qdot0))
        # Refers to D_1 L_d(q_0, q_1) (D1 is the partial derivative with respect to the first argument, Ld is the
        # discrete Lagrangian)
        d1_ld_q0_q1 = transpose(jacobian(self.discrete_lagrangian(q0, q1, time_step, discrete_approximation), q0))
        f0_minus = self.control_approximation(
            control0, control1, time_step, control_type, control_discrete_approximation
        )
        if self.has_holonomic_constraints and lambdas0 is None:
            raise ValueError("As your model is constrained, you must specify the lambdas.")
        constraint_term = (
            1 / 2 * transpose(self.discrete_holonomic_constraints_jacobian(time_step, q0)) @ lambdas0
            if self.has_holonomic_constraints
            else CX.zeros(self.nb_q, 1)
        )
        residual = d2_l_q0_qdot0 + d1_ld_q0_q1 + f0_minus - constraint_term

        if self.has_holonomic_constraints:
            return vertcat(
                residual,
                self.holonomic_constraints(q1),  # consistent with the paper
                self.holonomic_constraints(q0),  # to make sure the initial position fulfills the constraints
                self.holonomic_constraints_jacobian(q0) @ qdot0,  # and the initial velocity fulfills its derivative
            )  # constraints(0) is never evaluated if not here
        else:
            return residual

    def compute_final_states(
        self,
        time_step: MX | SX,
        q_penultimate: MX | SX,
        q_ultimate: MX | SX,
        q_dot_ultimate: MX | SX,
        control_penultimate: MX | SX,
        control_ultimate: MX | SX,
        lambdas_ultimate: MX | SX = None,
        *,
        discrete_approximation: QuadratureRule,
        control_type: ControlType,
        control_discrete_approximation: QuadratureRule,
    ):
        """
        Compute the initial states of the system from the initial position and velocity.

        Parameters
        ----------
        time_step: MX | SX
            The time step.
        q_penultimate: MX | SX
            The generalized coordinates at the penultimate node.
        q_ultimate: MX | SX
            The generalized coordinates at the ultimate node.
        q_dot_ultimate: MX | SX
            The generalized velocities at the ultimate node.
        control_penultimate: MX | SX
            The generalized forces at the penultimate node.
        control_ultimate: MX | SX
            The generalized forces at the ultimate node.
        lambdas_ultimate: MX | SX
            The Lagrange multipliers at the ultimate node.
        discrete_approximation: QuadratureRule
            The quadrature rule of the discrete Lagrangian.
        control_type: ControlType
            The type of the controls.
        control_discrete_approximation: QuadratureRule
            The quadrature rule of the discrete forcing.

        Returns
        -------
        MX | SX
            The discrete Euler-Lagrange equations adapted for the ultimate node.

        Sources
        -------
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        indications given just before the equation (18) for p0 and pN.
        """
        CX = MX if isinstance(q_penultimate, MX) else SX

        # Refers to D_2 L(q_N, \dot{q_N}) (D_2 is the partial derivative with respect to the second argument)
        d2_l_q_ultimate_qdot_ultimate = transpose(
            jacobian(self.lagrangian()(q_ultimate, q_dot_ultimate), q_dot_ultimate)
        )
        # Refers to D_2 L_d(q_{n-1}, q_1) (Ld is the discrete Lagrangian)
        d2_ld_q_penultimate_q_ultimate = transpose(
            jacobian(self.discrete_lagrangian(q_penultimate, q_ultimate, time_step, discrete_approximation), q_ultimate)
        )
        fd_plus = self.control_approximation(
            control_penultimate, control_ultimate, time_step, control_type, control_discrete_approximation
        )
        if self.has_holonomic_constraints and lambdas_ultimate is None:
            raise ValueError("As your model is constrained, you must specify the lambdas.")
        constraint_term = (
            1 / 2 * transpose(self.discrete_holonomic_constraints_jacobian(time_step, q_ultimate)) @ lambdas_ultimate
            if self.has_holonomic_constraints
            else CX.zeros(self.nb_q, 1)
        )

        residual = -d2_l_q_ultimate_qdot_ultimate + d2_ld_q_penultimate_q_ultimate + fd_plus - constraint_term
        # constraints(q_ultimate) has already been evaluated in the last constraint calling
        # discrete_euler_lagrange_equations, thus it is not necessary to evaluate it again here.
        # however it is necessary to evaluate the derivative of the constraints with respect to q_ultimate
        if self.has_holonomic_constraints:
            return vertcat(residual, self.holonomic_constraints_jacobian(q_ultimate) @ q_dot_ultimate)
        else:
            return residual
