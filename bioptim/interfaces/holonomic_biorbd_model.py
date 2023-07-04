from typing import Callable

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
)
from casadi import MX, DM, vertcat, horzcat, Function, solve, rootfinder, inv

from .biorbd_model import BiorbdModel


class HolonomicBiorbdModel(BiorbdModel):
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
                raise ValueError(
                    f"Joint {joint} is both dependant and independent. You need to specify this index in "
                    f"only one of these arguments: dependent_joint_index: independent_joint_index."
                )

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

    @property
    def dependent_joint_index(self) -> list:
        return self._dependent_joint_index

    @property
    def independent_joint_index(self) -> list:
        return self._independent_joint_index

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

    @property
    def has_holonomic_constraints(self):
        return self.nb_holonomic_constraints > 0

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
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")

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
        tau_augmented = tau - self.model.NonLinearEffect(q_biorbd, qdot_biorbd, f_ext=external_forces, f_contacts=f_contacts).to_mx()

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
        # q_u: independent
        # q_v: dependent
        mass_matrix = self.model.massMatrix(q).to_mx()
        mass_matrix_uu = mass_matrix[self._independent_joint_index, self._independent_joint_index]
        mass_matrix_uv = mass_matrix[self._independent_joint_index, self._dependent_joint_index]
        mass_matrix_vu = mass_matrix[self._dependent_joint_index, self._independent_joint_index]
        mass_matrix_vv = mass_matrix[self._dependent_joint_index, self._dependent_joint_index]

        first_line = horzcat(mass_matrix_uu, mass_matrix_uv)
        second_line = horzcat(mass_matrix_vu, mass_matrix_vv)

        return vertcat(first_line, second_line)

    def partitioned_non_linear_effect(self, q, qdot, f_ext=None, f_contacts=None):
        if f_ext is not None:
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")
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

    def partitioned_constraints_jacobian(self, q):
        constrained_jacobian = self.holonomic_constraints_jacobian(q)
        constrained_jacobian_u = constrained_jacobian[:, self._independent_joint_index]
        constrained_jacobian_v = constrained_jacobian[:, self._dependent_joint_index]

        return horzcat(constrained_jacobian_u, constrained_jacobian_v)

    def partitioned_forward_dynamics(
        self, q_u, qdot_u, tau, external_forces=None, f_contacts=None, q_v_init=None
    ) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")

        # compute q and qdot
        q = self.compute_q(q_u, q_v_init=q_v_init)
        qdot = self.compute_qdot(q, qdot_u)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_uu = partitioned_mass_matrix[: self.nb_independent_joints, : self.nb_independent_joints]
        m_uv = partitioned_mass_matrix[: self.nb_independent_joints, self.nb_independent_joints :]
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        coupling_matrix_vu = self.coupling_matrix(q)
        modified_mass_matrix = (
            m_uu
            + m_uv @ coupling_matrix_vu
            + coupling_matrix_vu.T @ m_vu
            + coupling_matrix_vu.T @ m_vv @ coupling_matrix_vu
        )
        second_term = m_uv + coupling_matrix_vu.T @ m_vv

        # compute the non-linear effect
        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_u = non_linear_effect[: self.nb_independent_joints]
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        modified_non_linear_effect = non_linear_effect_u + coupling_matrix_vu.T @ non_linear_effect_v

        # compute the tau
        partitioned_tau = self.partitioned_tau(tau)
        tau_u = partitioned_tau[: self.nb_independent_joints]
        tau_v = partitioned_tau[self.nb_independent_joints :]

        modified_generalized_forces = tau_u + coupling_matrix_vu.T @ tau_v

        qddot_u = inv(modified_mass_matrix) @ (
            modified_generalized_forces - second_term @ self.biais_vector(q, qdot) - modified_non_linear_effect
        )

        return qddot_u

    def coupling_matrix(self, q: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        partitioned_constrained_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constrained_jacobian_v = partitioned_constrained_jacobian[:, self.nb_independent_joints :]
        partitioned_constrained_jacobian_v_inv = inv(partitioned_constrained_jacobian_v)  # inv_minor otherwise ?

        partitioned_constrained_jacobian_u = partitioned_constrained_jacobian[:, : self.nb_independent_joints]

        return -partitioned_constrained_jacobian_v_inv @ partitioned_constrained_jacobian_u

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        The right term of the equation (15) in the paper.
        """
        partitioned_constrained_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constrained_jacobian_v = partitioned_constrained_jacobian[:, self.nb_independent_joints :]
        partitioned_constrained_jacobian_v_inv = inv(partitioned_constrained_jacobian_v)

        return -partitioned_constrained_jacobian_v_inv @ self.holonomic_constraints_jacobian(qdot) @ qdot

    def state_from_partition(self, state_u: MX, state_v: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        q = MX() if isinstance(state_u, MX) else DM()
        for i in range(self.nb_q):
            if i in self._independent_joint_index:
                q = vertcat(q, state_u[self._independent_joint_index.index(i)])
            else:
                q = vertcat(q, state_v[self._dependent_joint_index.index(i)])

        return q

    def compute_q_v(self, q_u: MX, q_v_init: MX = None) -> MX:
        q_v_init = MX.zeros(self.nb_dependent_joints) if q_v_init is None else q_v_init
        decision_variables = MX.sym("decision_variables", self.nb_dependent_joints)
        q = self.state_from_partition(q_u, decision_variables)
        mx_residuals = self.holonomic_constraints(q)

        residuals = Function(
            "final_states_residuals",
            [decision_variables, q_u],
            [mx_residuals],
        ).expand()

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": 1e-10}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)
        v_opt = ifcn(
            q_v_init,
            q_u,
        )

        return v_opt

    def compute_q(self, q_u, q_v_init: MX = None) -> MX:
        q_v = self.compute_q_v(q_u, q_v_init)
        return self.state_from_partition(q_u, q_v)

    def compute_qdot_v(self, q: MX, qdot_u: MX) -> MX:
        coupling_matrix_vu = self.coupling_matrix(q)
        return coupling_matrix_vu @ qdot_u

    def compute_qdot(self, q: MX, qdot_u: MX) -> MX:
        qdot_v = self.compute_qdot_v(q, qdot_u)
        return self.state_from_partition(qdot_u, qdot_v)

    def compute_q_v_numeric(self, q_u: DM, q_v_init=None):
        q_v_init = DM.zeros(self.nb_dependent_joints) if q_v_init is None else q_v_init
        decision_variables = MX.sym("decision_variables", self.nb_dependent_joints)
        q = self.state_from_partition(q_u, decision_variables)
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
            q_v_init,
        )

        return v_opt

    def compute_the_Lagrangian_multiplier(
        self, q: MX, qdot: MX, qddot: MX, tau: MX, external_forces: MX = None, f_contacts: MX = None
    ) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")
        J = self.partitioned_constraints_jacobian(q)
        Jv = J[:, self.nb_independent_joints :]
        Jvt_inv = inv(Jv.T)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        qddot_u = qddot[self._independent_joint_index]
        qddot_v = qddot[self._dependent_joint_index]

        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        Q = self.partitioned_tau(tau)
        Qv = Q[self.nb_independent_joints :]

        return Jvt_inv @ (m_vu @ qddot_u - m_vv @ qddot_v + non_linear_effect_v - Qv)
