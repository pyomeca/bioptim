from typing import Callable

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
)
from casadi import MX, SX, DM, vertcat, horzcat, Function, solve, rootfinder, inv

from .biorbd_model import BiorbdModel
from ..holonomic_constraints import HolonomicConstraintsList
from ...optimization.parameters import ParameterList


class HolonomicBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints.
    """

    def __init__(
        self,
        bio_model: str | biorbd.Model,
        parameters: ParameterList = None,
    ):
        super().__init__(bio_model, parameters=parameters)
        self._newton_tol = 1e-10
        self._holonomic_constraints = []
        self._holonomic_constraints_jacobians = []
        self._holonomic_constraints_derivatives = []
        self._holonomic_constraints_double_derivatives = []
        self.stabilization = False
        self.alpha = 0.01
        self.beta = 0.01
        self._dependent_joint_index = []
        self._independent_joint_index = [i for i in range(self.nb_q)]

    def set_newton_tol(self, newton_tol: float):
        self._newton_tol = newton_tol

    def set_holonomic_configuration(
        self,
        constraints_list: HolonomicConstraintsList,
        dependent_joint_index: list[int] = None,
        independent_joint_index: list[int] = None,
    ):
        """
        The joint indexes are not mandatory because a HolonomicBiorbdModel can be used without the partitioned dynamics,
        for instance in VariationalOptimalControlProgram.
        """
        dependent_joint_index = dependent_joint_index or []
        independent_joint_index = independent_joint_index or [i for i in range(self.nb_q)]

        if (dependent_joint_index is None) != (independent_joint_index is None):
            raise ValueError(
                "You need to specify both dependent_joint_index and independent_joint_index or none of them."
            )

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

        if sorted(dependent_joint_index) != dependent_joint_index:
            raise ValueError("The dependent_joint_index should be sorted in ascending order.")
        if sorted(independent_joint_index) != independent_joint_index:
            raise ValueError("The independent_joint_index should be sorted in ascending order.")

        self._dependent_joint_index = dependent_joint_index
        self._independent_joint_index = independent_joint_index

        for constraints_name in constraints_list.keys():
            self._add_holonomic_constraint(
                constraints_list[constraints_name]["constraints"],
                constraints_list[constraints_name]["constraints_jacobian"],
                constraints_list[constraints_name]["constraints_double_derivative"],
            )

        if dependent_joint_index and independent_joint_index:
            self.check_dependant_jacobian()

    def check_dependant_jacobian(self):
        q_test = MX.sym("q_test", self.nb_q)
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(q_test)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        shape = partitioned_constraints_jacobian_v.shape
        if shape[0] != shape[1]:
            raise ValueError(
                f"The shape of the dependent joint Jacobian should be square. Got: {shape}."
                f"Please consider checking the dimension of the holonomic constraints Jacobian."
            )

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

    def _add_holonomic_constraint(
        self,
        constraints: Function | Callable[[GeneralizedCoordinates], MX],
        constraints_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraints_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        self._holonomic_constraints.append(constraints)
        self._holonomic_constraints_jacobians.append(constraints_jacobian)
        self._holonomic_constraints_double_derivatives.append(constraints_double_derivative)

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
        # @ipuch does this stays contrained_
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")
        external_forces_set = self.model.externalForceSet()

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)

        mass_matrix = self.model.massMatrix(q_biorbd).to_mx()
        constraints_jacobian = self.holonomic_constraints_jacobian(q)
        constraints_jacobian_transpose = constraints_jacobian.T

        # compute the matrix DAE
        mass_matrix_augmented = horzcat(mass_matrix, constraints_jacobian_transpose)
        mass_matrix_augmented = vertcat(
            mass_matrix_augmented,
            horzcat(
                constraints_jacobian,
                MX.zeros((constraints_jacobian_transpose.shape[1], constraints_jacobian_transpose.shape[1])),
            ),
        )

        # compute b vector
        tau_augmented = tau - self.model.NonLinearEffect(q_biorbd, qdot_biorbd, external_forces_set).to_mx()

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
        external_forces_set = self.model.externalForceSet()
        non_linear_effect = self.model.NonLinearEffect(q, qdot, external_forces_set).to_mx()
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
        Also denoted as Bvu in the literature.

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        partitioned_constraints_jacobian_v_inv = inv(partitioned_constraints_jacobian_v)  # inv_minor otherwise ?

        partitioned_constraints_jacobian_u = partitioned_constraints_jacobian[:, : self.nb_independent_joints]

        return -partitioned_constraints_jacobian_v_inv @ partitioned_constraints_jacobian_u

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        The right term of the equation (15) in the paper.
        """
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        partitioned_constraints_jacobian_v_inv = inv(partitioned_constraints_jacobian_v)

        return -partitioned_constraints_jacobian_v_inv @ self.holonomic_constraints_jacobian(qdot) @ qdot

    def state_from_partition(self, state_u: MX, state_v: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        self.check_state_u_size(state_u)
        self.check_state_v_size(state_v)

        q = MX() if isinstance(state_u, MX) else DM()
        for i in range(self.nb_q):
            if i in self._independent_joint_index:
                q = vertcat(q, state_u[self._independent_joint_index.index(i)])
            else:
                q = vertcat(q, state_v[self._dependent_joint_index.index(i)])

        return q

    def check_state_u_size(self, state_u):
        if state_u.shape[0] != self.nb_independent_joints:
            raise ValueError(f"Length of state u size should be: {self.nb_independent_joints}. Got: {state_u.shape[0]}")

    def check_state_v_size(self, state_v):
        if state_v.shape[0] != self.nb_dependent_joints:
            raise ValueError(f"Length of state v size should be: {self.nb_dependent_joints}. Got: {state_v.shape[0]}")

    def compute_q_v(self, q_u: MX | SX | DM, q_v_init: MX | SX | DM = None) -> MX | SX | DM:
        """
        Compute the dependent joint positions from the independent joint positions.
        This function might be misleading because it can be used for numerical purpose with DM
        or for symbolic purpose with MX. The return type is not enforced.
        """
        decision_variables = MX.sym("decision_variables", self.nb_dependent_joints)
        q = self.state_from_partition(q_u, decision_variables)
        mx_residuals = self.holonomic_constraints(q)

        if isinstance(q_u, MX | SX):
            q_v_init = MX.zeros(self.nb_dependent_joints) if q_v_init is None else q_v_init
            ifcn_input = (q_v_init, q_u)
            residuals = Function(
                "final_states_residuals",
                [decision_variables, q_u],
                [mx_residuals],
            ).expand()
        else:
            q_v_init = DM.zeros(self.nb_dependent_joints) if q_v_init is None else q_v_init
            ifcn_input = (q_v_init,)
            residuals = Function(
                "final_states_residuals",
                [decision_variables],
                [mx_residuals],
            ).expand()

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": self._newton_tol}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)
        v_opt = ifcn(*ifcn_input)

        return v_opt

    def compute_q(self, q_u: MX, q_v_init: MX = None) -> MX:
        q_v = self.compute_q_v(q_u, q_v_init)
        return self.state_from_partition(q_u, q_v)

    def compute_qdot_v(self, q: MX, qdot_u: MX) -> MX:
        coupling_matrix_vu = self.coupling_matrix(q)
        return coupling_matrix_vu @ qdot_u

    def _compute_qdot_v(self, q_u: MX, qdot_u: MX) -> MX:
        q = self.compute_q(q_u)
        return self.compute_qdot_v(q, qdot_u)

    def compute_qdot(self, q: MX, qdot_u: MX) -> MX:
        qdot_v = self.compute_qdot_v(q, qdot_u)
        return self.state_from_partition(qdot_u, qdot_v)

    def compute_qddot_v(self, q: MX, qdot: MX, qddot_u: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
        coupling_matrix_vu = self.coupling_matrix(q)
        return coupling_matrix_vu @ qddot_u + self.biais_vector(q, qdot)

    def compute_qddot(self, q: MX, qdot: MX, qddot_u: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
        qddot_v = self.compute_qddot_v(q, qdot, qddot_u)
        return self.state_from_partition(qddot_u, qddot_v)

    def compute_the_lagrangian_multipliers(
        self, q_u: MX, qdot_u: MX, tau: MX, external_forces: MX = None, f_contacts: MX = None
    ) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        q = self.compute_q(q_u)
        qdot = self.compute_qdot(q, qdot_u)
        qddot_u = self.partitioned_forward_dynamics(q_u, qdot_u, tau, external_forces, f_contacts)
        qddot = self.compute_qddot(q, qdot, qddot_u)

        return self._compute_the_lagrangian_multipliers(q, qdot, qddot, tau, external_forces, f_contacts)

    def _compute_the_lagrangian_multipliers(
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
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        partitioned_constraints_jacobian_v_t_inv = inv(partitioned_constraints_jacobian_v.T)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        qddot_u = qddot[self._independent_joint_index]
        qddot_v = qddot[self._dependent_joint_index]

        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        partitioned_tau = self.partitioned_tau(tau)
        partitioned_tau_v = partitioned_tau[self.nb_independent_joints :]

        return partitioned_constraints_jacobian_v_t_inv @ (
            m_vu @ qddot_u + m_vv @ qddot_v + non_linear_effect_v - partitioned_tau_v
        )
