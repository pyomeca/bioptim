"""
Biorbd model for holonomic constraints and variational integrator.
"""
from typing import Callable

from bioptim import (
    BiorbdModel,
    ControlType,
)
import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, Function, solve, jacobian, transpose

from enums import QuadratureRule


class BiorbdModelCustomHolonomic(BiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints and the methods for the variational
    integrator, very experimental and not tested.
    """

    def __init__(self, bio_model: str | biorbd.Model):
        super().__init__(bio_model)
        self._holonomic_constraints = []
        self._holonomic_constraints_jacobian = []
        self._holonomic_constraints_derivatives = []
        self._holonomic_constraints_double_derivatives = []
        self.stabilization = False
        self.alpha = 0.01
        self.beta = 0.01

    def add_holonomic_constraint(
        self,
        constraint: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        """
        Add a holonomic constraint to the model

        Parameters
        ----------
        constraint: Function | Callable[[GeneralizedCoordinates], MX]
            The holonomic constraint
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX]
            The jacobian of the holonomic constraint
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX]
            The double derivative of the holonomic constraint
        """
        self._holonomic_constraints.append(constraint)
        self._holonomic_constraints_jacobian.append(constraint_jacobian)
        self._holonomic_constraints_double_derivatives.append(constraint_double_derivative)

    @property
    def nb_holonomic_constraints(self):
        """
        Get the number of holonomic constraints

        Returns
        -------
        int
            The number of holonomic constraints
        """
        return sum([c.nnz_out() for c in self._holonomic_constraints])

    def holonomic_constraints(self, q: MX):
        return vertcat(*[c(q) for c in self._holonomic_constraints])

    def holonomic_constraints_jacobian(self, q: MX):
        return vertcat(*[c(q) for c in self._holonomic_constraints_jacobian])

    def holonomic_constraints_derivative(self, q: MX, qdot: MX):
        return self.holonomic_constraints_jacobian(q) @ qdot

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX):
        return vertcat(*[c(q, qdot, qddot) for c in self._holonomic_constraints_double_derivatives])

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        """
        Compute the forward dynamics of the model

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        tau: MX
            The generalized torques
        external_forces: MX
            The external forces
        f_contacts: MX
            The contact forces

        Returns
        -------
        MX
            The generalized accelerations
        """
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)

        mass_matrix = self.model.massMatrix(q_biorbd).to_mx()
        constraint_jacobian = self.holonomic_constraints_jacobian(q)
        constraint_jacobian_transpose = constraint_jacobian.T

        # compute the matrix DAE
        mass_matrix_augmented = horzcat(mass_matrix, constraint_jacobian_transpose)
        mass_matrix_augmented = vertcat(
            mass_matrix_augmented,
            horzcat(
                constraint_jacobian,
                MX.zeros((constraint_jacobian_transpose.shape[1], constraint_jacobian_transpose.shape[1])),
            ),
        )

        # compute b vector
        tau_augmented = tau - self.model.NonLinearEffect(q_biorbd, qdot_biorbd, f_ext=None, f_contacts=None).to_mx()

        biais = -self.holonomic_constraints_jacobian(qdot) @ qdot
        if self.stabilization:
            biais -= self.alpha * self.holonomic_constraints(q) + self.beta * self.holonomic_constraints_derivative(
                q, qdot
            )

        tau_augmented = vertcat(tau_augmented, biais)

        # solve with casadi Ax = b

        x = solve(mass_matrix_augmented, tau_augmented, "symbolicqr")

        return x[: self.nb_qddot]

    def dae_inverse_dynamics(
        self, q, qdot, qddot, tau, lagrange_multipliers, external_forces=None, f_contacts=None
    ) -> MX:
        """
        Compute the inverse dynamics of the model
        Ax-b = 0
        """
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)

        mass_matrix = self.model.massMatrix(q_biorbd)
        constraint_jacobian = self.holonomic_constraints_jacobian(q)
        constraint_jacobian_transpose = constraint_jacobian.T

        # compute the matrix DAE
        mass_matrix_augmented = horzcat(mass_matrix, constraint_jacobian_transpose)
        mass_matrix_augmented = vertcat(
            mass_matrix_augmented, horzcat(constraint_jacobian, MX.zeros(constraint_jacobian_transpose.shape))
        )

        # compute b vector
        tau_augmented = tau - self.model.NonLinearEffect(q_biorbd, qdot_biorbd, f_ext=None, f_contacts=None)
        tau_augmented = vertcat(tau_augmented, self.holonomic_constraints_jacobian(qdot) @ qdot)

        # Ax-b = 0
        return mass_matrix_augmented @ vertcat(qddot_biorbd, lagrange_multipliers) - tau_augmented

    def lagrangian(self, q: MX | SX, qdot: MX | SX) -> MX | SX:
        """
        Compute the Lagrangian of a biorbd model.

        Parameters
        ----------
        q: MX | SX
            The generalized coordinates.
        qdot: MX | SX
            The generalized velocities.

        Returns
        -------
        The Lagrangian.
        """

        return self.model.KineticEnergy(q, qdot).to_mx() - self.model.PotentialEnergy(q).to_mx()

    def discrete_lagrangian(
        self,
        q1: MX | SX,
        q2: MX | SX,
        time_step: MX | SX,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
    ) -> MX | SX:
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
            The quadrature rule to use for the discrete Lagrangian.

        Returns
        -------
        The discrete Lagrangian.
        """
        if discrete_approximation == QuadratureRule.MIDPOINT:
            q_discrete = (q1 + q2) / 2
            qdot_discrete = (q2 - q1) / time_step
            return time_step * self.lagrangian(q_discrete, qdot_discrete)
        elif discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
            q_discrete = q1
            qdot_discrete = (q2 - q1) / time_step
            return time_step * self.lagrangian(q_discrete, qdot_discrete)
        elif discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
            q_discrete = q2
            qdot_discrete = (q2 - q1) / time_step
            return time_step * self.lagrangian(q_discrete, qdot_discrete)
        elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
            # from : M. West, “Variational integrators,” Ph.D. dissertation, California Inst.
            # Technol., Pasadena, CA, 2004. p 13
            qdot_discrete = (q2 - q1) / time_step
            return time_step / 2 * (self.lagrangian(q1, qdot_discrete) + self.lagrangian(q2, qdot_discrete))
        else:
            raise NotImplementedError(f"Discrete Lagrangian {discrete_approximation} is not implemented")

    @staticmethod
    def control_approximation(
        control_minus: MX | SX,
        control_plus: MX | SX,
        time_step: float,
        control_type: ControlType = ControlType.CONSTANT,
        discrete_approximation: QuadratureRule = QuadratureRule.MIDPOINT,
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
            The type of control.
        discrete_approximation: QuadratureRule
            The quadrature rule to use for the discrete Lagrangian.

        Returns
        ----------
        The term associated to the controls in the Lagrangian equations.
        Johnson, E. R., & Murphey, T. D. (2009).
        Scalable Variational Integrators for Constrained Mechanical Systems in Generalized Coordinates.
        IEEE Transactions on Robotics, 25(6), 1249–1261. doi:10.1109/tro.2009.2032955
        """
        if control_type == ControlType.CONSTANT:
            return 1 / 2 * control_minus * time_step

        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if discrete_approximation == QuadratureRule.MIDPOINT:
                return 1 / 2 * (control_minus + control_plus) / 2 * time_step
            elif discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
                return 1 / 2 * control_minus * time_step
            elif discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
                return 1 / 2 * control_plus * time_step
            elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
                raise NotImplementedError(f"Discrete {discrete_approximation} is not implemented for {control_type}")

    def discrete_euler_lagrange_equations(
        self,
        time_step: MX | SX,
        q_prev: MX | SX,
        q_cur: MX | SX,
        q_next: MX | SX,
        control_prev: MX | SX,
        control_cur: MX | SX,
        control_next: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas: MX | SX = None,
    ) -> MX | SX:
        """
        Compute the discrete Euler-Lagrange equations of a biorbd model

        Parameters
        ----------
        time_step: MX | SX
            The time step.
        q_prev: MX | SX
            The generalized coordinates at the first time step.
        q_cur: MX | SX
            The generalized coordinates at the second time step.
        q_next: MX | SX
            The generalized coordinates at the third time step.
        control_prev: MX | SX
            The generalized forces at the first time step.
        control_cur: MX | SX
            The generalized forces at the second time step.
        control_next: MX | SX
            The generalized forces at the third time step.
        constraints: Function
            The constraints.
        jac: Function
            The jacobian of the constraints.
        lambdas: MX | SX
            The Lagrange multipliers.
        """
        p_current = transpose(jacobian(self.discrete_lagrangian(q_prev, q_cur, time_step), q_cur))

        D1_Ld_qcur_qnext = transpose(jacobian(self.discrete_lagrangian(q_cur, q_next, time_step), q_cur))
        constraint_term = transpose(jac(q_cur)) @ lambdas if constraints is not None else MX.zeros(p_current.shape)

        residual = (
            p_current
            + D1_Ld_qcur_qnext
            - constraint_term
            + self.control_approximation(control_prev, control_cur, time_step)
            + self.control_approximation(control_cur, control_next, time_step)
        )

        if constraints is not None:
            return vertcat(residual, constraints(q_next))
        else:
            return residual

    def compute_initial_states(
        self,
        time_step: MX | SX,
        q0: MX | SX,
        q0_dot: MX | SX,
        q1: MX | SX,
        control0: MX | SX,
        control1: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas0: MX | SX = None,
    ):
        """
        Compute the initial states of the system from the initial position and velocity.
        """
        # The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        # constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        # indications given just before the equation (18) for p0 and pN.
        D2_L_q0_q0dot = transpose(jacobian(self.lagrangian(q0, q0_dot), q0_dot))
        D1_Ld_q0_q1 = transpose(jacobian(self.discrete_lagrangian(q0, q1, time_step), q0))
        f0_minus = self.control_approximation(control0, control1, time_step)
        constraint_term = 1 / 2 * transpose(jac(q0)) @ lambdas0 if constraints is not None else MX.zeros(self.nb_q, 1)
        residual = D2_L_q0_q0dot + D1_Ld_q0_q1 + f0_minus - constraint_term

        if constraints is not None:
            return vertcat(residual, constraints(q0), constraints(q1))  # constraints(0) is never evaluated if not here
        else:
            return residual

    def compute_final_states(
        self,
        time_step: MX | SX,
        qN_minus_1: MX | SX,
        qN: MX | SX,
        qN_dot: MX | SX,
        controlN_minus_1: MX | SX,
        controlN: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdasN: MX | SX = None,
    ):
        """
        Compute the initial states of the system from the initial position and velocity.
        """
        # The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        # constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        # indications given just before the equation (18) for p0 and pN.
        D2_L_qN_qN_dot = transpose(jacobian(self.lagrangian(qN, qN_dot), qN_dot))
        D2_Ld_qN_minus_1_qN = transpose(jacobian(self.discrete_lagrangian(qN_minus_1, qN, time_step), qN))
        fd_plus = self.control_approximation(controlN_minus_1, controlN, time_step)
        constraint_term = 1 / 2 * transpose(jac(qN)) @ lambdasN if constraints is not None else MX.zeros(self.nb_q, 1)

        residual = -D2_L_qN_qN_dot + D2_Ld_qN_minus_1_qN + fd_plus - constraint_term
        # constraints(qN) has already been evaluated in the last constraint calling
        # discrete_euler_lagrange_equations, thus it is not necessary to evaluate it again here.
        return residual

    def generate_constraint_and_jacobian_functions(
        self, marker_1: str, marker_2: str = None, index: slice = slice(0, 3)
    ) -> tuple[Function, Function]:
        """Generate a close loop constraint between two markers"""

        # symbolic variables to create the functions
        q_sym = MX.sym("q", self.nb_q, 1)

        # symbolic markers in global frame
        marker_1_sym = self.marker(q_sym, index=self.marker_index(marker_1))
        if marker_2 is not None:
            marker_2_sym = self.marker(q_sym, index=self.marker_index(marker_2))
            # the constraint is the distance between the two markers, set to zero
            constraint = (marker_1_sym - marker_2_sym)[index]
        else:
            # the constraint is the position of the marker, set to zero
            constraint = marker_1_sym[index]
        # the jacobian of the constraint
        constraint_jacobian = jacobian(constraint, q_sym)

        constraint_func = Function(
            "holonomic_constraint",
            [q_sym],
            [constraint],
            ["q"],
            ["holonomic_constraint"],
        ).expand()

        constraint_jacobian_func = Function(
            "holonomic_constraint_jacobian",
            [q_sym],
            [constraint_jacobian],
            ["q"],
            ["holonomic_constraint_jacobian"],
        ).expand()

        return constraint_func, constraint_jacobian_func
