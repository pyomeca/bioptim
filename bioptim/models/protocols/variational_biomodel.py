from typing import Protocol
from casadi import MX, SX
from ..protocols.holonomic_biomodel import HolonomicBioModel


from ...misc.parameters_types import (
    Float,
    MXorSX,
    MXorSXOptional,
)


class VariationalBioModel(HolonomicBioModel, Protocol):
    def discrete_lagrangian(
        self,
        q1: MXorSX,
        q2: MXorSX,
        time_step: MXorSX,
    ) -> MXorSX:
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

        Returns
        -------
        The discrete Lagrangian.
        """

    def control_approximation(
        self,
        control_minus: MXorSX,
        control_plus: MXorSX,
        time_step: Float,
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

        Returns
        ----------
        The term associated to the controls in the Lagrangian equations.

        Sources
        -------
        Johnson, E. R., & Murphey, T. D. (2009).
        Scalable Variational Integrators for Constrained Mechanical Systems in Generalized Coordinates.
        IEEE Transactions on Robotics, 25(6), 1249–1261. doi:10.1109/tro.2009.2032955
        """

    def discrete_holonomic_constraints_jacobian(self, time_step: MXorSX, q: MXorSX) -> MXorSXOptional:
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

    def discrete_euler_lagrange_equations(
        self,
        time_step: MXorSX,
        q_prev: MXorSX,
        q_cur: MXorSX,
        q_next: MXorSX,
        control_prev: MXorSX,
        control_cur: MXorSX,
        control_next: MXorSX,
        lambdas: MXorSXOptional = None,
    ) -> MXorSX:
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
        time_step: MXorSX,
        q0: MXorSX,
        qdot0: MXorSX,
        q1: MXorSX,
        control0: MXorSX,
        control1: MXorSX,
        lambdas0: MXorSXOptional = None,
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

    def compute_final_states(
        self,
        time_step: MXorSX,
        q_penultimate: MXorSX,
        q_ultimate: MXorSX,
        q_dot_ultimate: MXorSX,
        control_penultimate: MXorSX,
        control_ultimate: MXorSX,
        lambdas_ultimate: MXorSXOptional = None,
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
