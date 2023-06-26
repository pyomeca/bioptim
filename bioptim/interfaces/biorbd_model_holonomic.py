from typing import Callable

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
)
from casadi import MX, DM, vertcat, horzcat, Function, solve, rootfinder, inv

from .biorbd_model import BiorbdModel


class BiorbdModelHolonomic(BiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints.
    """

    def __init__(self, bio_model: str | biorbd.Model):
        super().__init__(bio_model)
        self._holonomic_constraints = []
        self._holonomic_constraints_jacobians = []
        self._holonomic_constraints_derivatives = []
        self._holonomic_constraints_double_derivatives = []
        self.stabilization = False
        self.alpha = 0.01
        self.beta = 0.01
        self._dependent_joint_index = []
        self._independent_joint_index = [i for i in range(self.nb_q)]

    def set_dependencies(self, dependent_joint_index: list, independent_joint_index: list):
        if len(dependent_joint_index) + len(independent_joint_index) != self.nb_q:
            raise ValueError(
                "The sum of the number of dependent and independent joints should be equal to the number of DoF of the"
                " model"
            )

        for joint in dependent_joint_index:
            if joint >= self.nb_q:
                raise ValueError(f"Joint index {joint} is not a valid joint index since the model has {self.nb_q} DoF")
            if joint in independent_joint_index:
                raise ValueError(f"You registered {joint} as both dependent and independent")

        for joint in independent_joint_index:
            if joint >= self.nb_q:
                raise ValueError(f"Joint index {joint} is not a valid joint index since the model has {self.nb_q} DoF")

        self._dependent_joint_index = dependent_joint_index
        self._independent_joint_index = independent_joint_index

    @property
    def nb_independent_joints(self):
        return len(self._independent_joint_index)

    @property
    def nb_dependent_joints(self):
        return len(self._dependent_joint_index)

    def add_holonomic_constraint(
        self,
        constraint: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        self._holonomic_constraints.append(constraint)
        self._holonomic_constraints_jacobians.append(constraint_jacobian)
        self._holonomic_constraints_double_derivatives.append(constraint_double_derivative)

    @property
    def nb_holonomic_constraints(self):
        return sum([c.nnz_out() for c in self._holonomic_constraints])

    def holonomic_constraints(self, q: MX):
        return vertcat(*[c(q) for c in self._holonomic_constraints])

    def holonomic_constraints_jacobian(self, q: MX):
        return vertcat(*[c(q) for c in self._holonomic_constraints_jacobians])

    def holonomic_constraints_derivative(self, q: MX, qdot: MX):
        return self.holonomic_constraints_jacobian(q) @ qdot

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX):
        return vertcat(*[c(q, qdot, qddot) for c in self._holonomic_constraints_double_derivatives])

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
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

    def partitioned_mass_matrix(self, q):
        # u: independent
        # v: dependent
        mass_matrix = self.model.massMatrix(q).to_mx()
        mass_matrix_uu = mass_matrix[self._independent_joint_index, self._independent_joint_index]
        mass_matrix_uv = mass_matrix[self._independent_joint_index, self._dependent_joint_index]
        mass_matrix_vu = mass_matrix[self._dependent_joint_index, self._independent_joint_index]
        mass_matrix_vv = mass_matrix[self._dependent_joint_index, self._dependent_joint_index]

        first_line = horzcat(mass_matrix_uu, mass_matrix_uv)
        second_line = horzcat(mass_matrix_vu, mass_matrix_vv)

        return vertcat(first_line, second_line)

    def partitioned_non_linear_effect(self, q, qdot, f_ext=None, f_contacts=None):
        non_linear_effect = self.model.NonLinearEffect(q, qdot, f_ext=f_ext, f_contacts=f_contacts).to_mx()
        non_linear_effect_u = non_linear_effect[self._independent_joint_index]
        non_linear_effect_v = non_linear_effect[self._dependent_joint_index]

        return vertcat(non_linear_effect_u, non_linear_effect_v)

    def partitioned_q(self, q):
        q_u = q[self._independent_joint_index]
        q_v = q[self._dependent_joint_index]

        return vertcat(q_u, q_v)

    def partitioned_qdot(self, qdot):
        qdot_u = qdot[self._independent_joint_index]
        qdot_v = qdot[self._dependent_joint_index]

        return vertcat(qdot_u, qdot_v)

    def partitioned_tau(self, tau):
        tau_u = tau[self._independent_joint_index]
        tau_v = tau[self._dependent_joint_index]

        return vertcat(tau_u, tau_v)

    def partitioned_constrained_jacobian(self, q):
        constrained_jacobian = self.holonomic_constraints_jacobian(q)
        constrained_jacobian_u = constrained_jacobian[:, self._independent_joint_index]
        constrained_jacobian_v = constrained_jacobian[:, self._dependent_joint_index]

        return horzcat(constrained_jacobian_u, constrained_jacobian_v)

    def partitioned_forward_dynamics(self, u, udot, tau, external_forces=None, f_contacts=None, v_init=None) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        # compute v from u
        v = self.compute_v_from_u(u, v_init=v_init)
        q = self.q_from_u_and_v(u, v)

        Bvu = self.coupling_matrix(q)
        vdot = Bvu @ udot
        qdot = self.q_from_u_and_v(udot, vdot)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_uu = partitioned_mass_matrix[: self.nb_independent_joints, : self.nb_independent_joints]
        m_uv = partitioned_mass_matrix[: self.nb_independent_joints, self.nb_independent_joints :]
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        modified_mass_matrix = m_uu + m_uv @ Bvu + Bvu.T @ m_vu + Bvu.T @ m_vv @ Bvu
        second_term = m_uv + Bvu.T @ m_vv

        # compute the non linear effect
        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_u = non_linear_effect[: self.nb_independent_joints]
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        modified_non_linear_effect = non_linear_effect_u + Bvu.T @ non_linear_effect_v

        # compute the tau
        partitioned_tau = self.partitioned_tau(tau)
        tau_u = partitioned_tau[: self.nb_independent_joints]
        tau_v = partitioned_tau[self.nb_independent_joints :]

        modified_generalized_forces = tau_u + Bvu.T @ tau_v

        uddot = inv(modified_mass_matrix) @ (
            modified_generalized_forces - second_term @ self.biais_vector(q, qdot) - modified_non_linear_effect
        )

        return uddot

    def coupling_matrix(self, q: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        J = self.partitioned_constrained_jacobian(q)
        Jv = J[:, self.nb_independent_joints :]
        Jv_inv = inv(Jv)  # inv_minor otherwise ?

        Ju = J[:, : self.nb_independent_joints]

        return -Jv_inv @ Ju

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        The right term of the equation (15) in the paper.
        """
        J = self.partitioned_constrained_jacobian(q)
        Jv = J[:, self.nb_independent_joints :]
        Jv_inv = inv(Jv)  # inv_minor otherwise ?

        return Jv_inv @ self.holonomic_constraints_jacobian(qdot) @ qdot

    def q_from_u_and_v(self, u: MX, v: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        q = MX() if isinstance(u, MX) else DM()
        for i in range(self.nb_q):
            if i in self._independent_joint_index:
                q = vertcat(q, u[self._independent_joint_index.index(i)])
            else:
                q = vertcat(q, v[self._dependent_joint_index.index(i)])

        return q

    def compute_v_from_u(self, u: MX, v_init: MX = None) -> MX:
        if v_init is None:
            v_init = MX.zeros(self.nb_dependent_joints)
        decision_variables = MX.sym("decision_variables", self.nb_dependent_joints)
        q = self.q_from_u_and_v(u, decision_variables)
        mx_residuals = self.holonomic_constraints(q)

        residuals = Function(
            "final_states_residuals",
            [decision_variables, u],
            [mx_residuals],
        ).expand()

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": 1e-10}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)
        v_opt = ifcn(
            v_init,
            u,
        )

        return v_opt

    def compute_v_from_u_numeric(self, u: DM, v_init=None):
        if v_init is None:
            v_init = DM.zeros(self.nb_dependent_joints)
        decision_variables = MX.sym("decision_variables", self.nb_dependent_joints)
        q = self.q_from_u_and_v(u, decision_variables)
        mx_residuals = self.holonomic_constraints(q)

        residuals = Function(
            "final_states_residuals",
            [decision_variables],
            [mx_residuals],
        ).expand()

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": 1e-10}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)
        v_opt = ifcn(
            v_init,
        )

        return v_opt
