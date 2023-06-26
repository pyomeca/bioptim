from typing import Protocol, Callable

from biorbd_casadi import GeneralizedCoordinates
from casadi import MX, SX, DM, Function
from ..misc.enums import ControlType, QuadratureRule
from ..misc.mapping import BiMapping, BiMappingList
from ..interfaces.biorbd_model import Bounds
from ..interfaces.biomodel_holonomic import BioModelHolonomic


class VariationalBioModel(BioModelHolonomic):
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
            The generalized coordinates at the first node.
        q2: MX | SX
            The generalized coordinates at the second node.
        time_step: float
            The time step.
        discrete_approximation: QuadratureRule
            The quadrature rule to use for the discrete Lagrangian.

        Returns
        -------
        The discrete Lagrangian.
        """

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

        Sources
        -------
        Johnson, E. R., & Murphey, T. D. (2009).
        Scalable Variational Integrators for Constrained Mechanical Systems in Generalized Coordinates.
        IEEE Transactions on Robotics, 25(6), 1249–1261. doi:10.1109/tro.2009.2032955
        """

    @staticmethod
    def compute_holonomic_discrete_constraints_jacobian(
        jac: Function, time_step: MX | SX, q: MX | SX
    ) -> MX | SX | None:
        """
        Compute the discrete Jacobian of the holonomic constraints. See Variational integrators for constrained
        dynamical systems (https://onlinelibrary.wiley.com/doi/epdf/10.1002/zamm.200700173) eq. (21) for more
        precisions.

        Parameters
        ----------
        jac: Function
            The Jacobian of the holonomic constraints.
        time_step: MX | SX
            The time step.
        q:
            The coordinates.

        Returns
        -------
        holonomic_discrete_constraints_jacobian: MX | SX | None
            The discrete Jacobian of the holonomic constraints if there is constraints, None otherwise.
        """

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
        constraints: Function
            The constraints.
        jac: Function
            The jacobian of the constraints.
        lambdas: MX | SX
            The Lagrange multipliers.

        Returns
        -------
        MX | SX
            The discrete Euler-Lagrange equations.

        Sources
        -------
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (10).
        """

    def compute_initial_states(
        self,
        time_step: MX | SX,
        q0: MX | SX,
        qdot0: MX | SX,
        q1: MX | SX,
        control0: MX | SX,
        control1: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas0: MX | SX = None,
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
        constraints: Function
            The constraints.
        jac: Function
            The jacobian of the constraints.
        lambdas0: MX | SX
            The Lagrange multipliers at the first node.

        Returns
        -------
        MX | SX
            The discrete Euler-Lagrange equations adapted for the first node.

        Sources
        -------
        Compute the initial states of the system from the initial position and velocity.
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        indications given just before the equation (18) for p0 and pN.
        """

    def compute_final_states(
        self,
        time_step: MX | SX,
        q_penultimate: MX | SX,
        q_ultimate: MX | SX,
        q_dot_ultimate: MX | SX,
        control_penultimate: MX | SX,
        control_ultimate: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas_ultimate: MX | SX = None,
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
        constraints: Function
            The constraints.
        jac: Function
            The jacobian of the constraints.
        lambdas_ultimate: MX | SX
            The Lagrange multipliers at the ultimate node.

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
