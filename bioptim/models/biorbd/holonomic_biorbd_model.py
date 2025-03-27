from typing import Callable
from functools import wraps

import biorbd_casadi as biorbd
import numpy as np
import scipy.linalg as la
from biorbd_casadi import (
    GeneralizedCoordinates,
)
from casadi import MX, DM, vertcat, horzcat, Function, solve, rootfinder, inv, nlpsol

from .biorbd_model import BiorbdModel
from ..holonomic_constraints import HolonomicConstraintsList
from ...optimization.parameters import ParameterList

from ...misc.parameters_types import (
    Str,
    Float,
    NpArray,
    IntListOptional,
    AnyList,
    BiorbdModel,
)


class HolonomicBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints.
    """

    def __init__(
        self,
        bio_model: Str | BiorbdModel,
        friction_coefficients: NpArray = None,
        parameters: ParameterList = None,
    ):
        super().__init__(bio_model, friction_coefficients, parameters)
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

        if parameters is not None:
            raise NotImplementedError("HolonomicBiorbdModel does not support parameters yet")

        self._cached_functions = {}

    def _holonomic_symbolic_variables(self):
        # Declaration of MX variables of the right shape for the creation of CasADi Functions
        self.q_u = MX.sym("q_u_mx", self.nb_independent_joints, 1)
        self.qdot_u = MX.sym("qdot_u_mx", self.nb_independent_joints, 1)
        self.qddot_u = MX.sym("qddot_u_mx", self.nb_independent_joints, 1)
        self.q_v = MX.sym("q_v_mx", self.nb_dependent_joints, 1)
        self.qdot_v = MX.sym("qdot_v_mx", self.nb_dependent_joints, 1)
        self.qddot_v = MX.sym("qddot_v_mx", self.nb_dependent_joints, 1)
        self.q_v_init = MX.sym("q_v_init_mx", self.nb_dependent_joints, 1)

    def cache_function(method):
        """Decorator to cache CasADi functions automatically"""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Create a unique key based on the method name and arguments
            key = (method.__name__, args, frozenset(kwargs.items()))
            if key in self._cached_functions:
                return self._cached_functions[key]

            # Call the original function to create the CasADi function
            casadi_fun = method(self, *args, **kwargs)

            # Store in the cache
            self._cached_functions[key] = casadi_fun
            return casadi_fun

        return wrapper

    def set_newton_tol(self, newton_tol: Float):
        self._newton_tol = newton_tol

    def set_holonomic_configuration(
        self,
        constraints_list: HolonomicConstraintsList,
        dependent_joint_index: IntListOptional = None,
        independent_joint_index: IntListOptional = None,
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

        self._holonomic_symbolic_variables()

    def check_dependant_jacobian(self):
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(self.q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        shape = partitioned_constraints_jacobian_v.shape
        if shape[0] != shape[1]:
            output = self.partition_coordinates()
            raise ValueError(
                f"The shape of the dependent joint Jacobian should be square. Got: {shape}."
                f"Please consider checking the dimension of the holonomic constraints Jacobian.\n"
                f"Here is a recommended partitioning: "
                f"      - independent_joint_index: {output[1]},"
                f"      - dependent_joint_index: {output[0]}."
            )

    def partition_coordinates(self):
        q = MX.sym("q", self.nb_q, 1)
        s = nlpsol("sol", "ipopt", {"x": q, "g": self.holonomic_constraints(q)})
        q_star = np.array(
            s(
                x0=np.zeros(self.nb_q),
                lbg=np.zeros(self.nb_holonomic_constraints),
                ubg=np.zeros(self.nb_holonomic_constraints),
            )["x"]
        )[:, 0]
        return self.jacobian_coordinate_partitioning(self.holonomic_constraints_jacobian(q_star).toarray())

    @staticmethod
    def jacobian_coordinate_partitioning(J, tol=None):
        """
        Determine a coordinate partitioning q = {q_u, q_v} from a Jacobian J(q) of size (m x n),
        where m is the number of constraints and
        n is the total number of generalized coordinates.

        We want to find an invertible submatrix J_v of size (m x m) by reordering
        the columns of J according to the largest pivots. Those columns in J_v
        correspond to the 'dependent' coordinates q_v, while the remaining columns
        correspond to the 'independent' coordinates q_u.

        Parameters
        ----------
        J : array_like, shape (m, n)
            The constraint Jacobian evaluated at the current configuration q.
        tol : float, optional
            Tolerance for rank detection. If None, a default based on the machine
            precision and the size of J is used.

        Returns
        -------
        qv_indices : ndarray of shape (r,)
            The indices of the columns in J chosen as dependent coordinates.
            Typically, we expect r = m if J has full row rank (i.e. no redundant constraints).
        qu_indices : ndarray of shape (n - r,)
            The indices of the columns chosen as independent coordinates.
        rankJ : int
            The detected rank of J. If rankJ < m, it means some constraints are redundant.

        Notes
        -----
        - If rankJ < m, then there are redundant or degenerate constraints in J.
          The 'extra' constraints can be ignored in subsequent computations.
        - If rankJ = m, then J has full row rank and the submatrix J_v is invertible.
        - After obtaining qv_indices and qu_indices, one typically reorders q
          so that q = [q_u, q_v], and likewise reorders the columns of J so that
          J = [J_u, J_v].
        """

        # J is (m, n): number of constraints = m, number of coords = n.
        J = np.asarray(J, dtype=float)
        m, n = J.shape

        # Perform a pivoted QR factorization: J = Q * R[:, pivot]
        # pivot is a permutation of [0, 1, ..., n-1],
        # reordering the columns from "largest pivot" to "smallest pivot" in R.
        Q, R, pivot = la.qr(J, pivoting=True)

        # If no tolerance is specified, pick a default related to the matrix norms and eps
        if tol is None:
            # A common heuristic: tol ~ max(m, n) * machine_eps * largest_abs_entry_in_R
            # The largest absolute entry is often approximated by abs(R[0, 0]) if the matrix
            # is well-ordered by pivot. However, you can also do np.linalg.norm(R, ord=np.inf).
            tol = max(m, n) * np.finfo(J.dtype).eps * abs(R[0, 0])

        # Rank detection from the diagonal of R
        diagR = np.abs(np.diag(R))
        rankJ = np.sum(diagR > tol)

        # The 'best' columns (by largest pivots) are pivot[:rankJ].
        # If J is full row rank and not degenerate, we expect rankJ == m.
        qv_indices = pivot[:rankJ]  # Dependent variables
        qu_indices = pivot[rankJ:]  # Independent variables

        return qv_indices, qu_indices, rankJ

    @property
    def nb_independent_joints(self):
        return len(self._independent_joint_index)

    @property
    def nb_dependent_joints(self):
        return len(self._dependent_joint_index)

    @property
    def dependent_joint_index(self) -> AnyList:
        return self._dependent_joint_index

    @property
    def independent_joint_index(self) -> AnyList:
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

    def holonomic_constraints(self, q: MX) -> MX:
        return vertcat(*[c(q) for c in self._holonomic_constraints])

    def holonomic_constraints_jacobian(self, q: MX) -> MX:
        return vertcat(*[c(q) for c in self._holonomic_constraints_jacobians])

    def holonomic_constraints_derivative(self, q: MX, qdot: MX) -> MX:
        return self.holonomic_constraints_jacobian(q) @ qdot

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX) -> MX:
        return vertcat(*[c(q, qdot, qddot) for c in self._holonomic_constraints_double_derivatives])

    @cache_function
    def constrained_forward_dynamics(self) -> Function:

        mass_matrix = self.mass_matrix()(self.q, self.parameters)
        constraints_jacobian = self.holonomic_constraints_jacobian(self.q)
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
        tau_augmented = self.tau - self.non_linear_effects()(self.q, self.qdot, self.parameters)

        biais = -self.holonomic_constraints_jacobian(self.qdot) @ self.qdot
        if self.stabilization:
            biais -= self.alpha * self.holonomic_constraints(
                self.q
            ) + self.beta * self.holonomic_constraints_derivative(self.q, self.qdot)

        tau_augmented = vertcat(tau_augmented, biais)

        # solve with casadi Ax = b
        x = solve(mass_matrix_augmented, tau_augmented, "symbolicqr")

        biorbd_return = x[: self.nb_qddot]

        casadi_fun = Function(
            "constrained_forward_dynamics",
            [self.q, self.qdot, self.tau, self.parameters],
            [biorbd_return],
            ["q", "qdot", "tau", "parameters"],
            ["qddot"],
        )
        return casadi_fun

    def partitioned_mass_matrix(self, q: MX) -> MX:
        # q_u: independent
        # q_v: dependent
        mass_matrix = self.mass_matrix()(q, [])
        mass_matrix_uu = mass_matrix[self._independent_joint_index, self._independent_joint_index]
        mass_matrix_uv = mass_matrix[self._independent_joint_index, self._dependent_joint_index]
        mass_matrix_vu = mass_matrix[self._dependent_joint_index, self._independent_joint_index]
        mass_matrix_vv = mass_matrix[self._dependent_joint_index, self._dependent_joint_index]

        first_line = horzcat(mass_matrix_uu, mass_matrix_uv)
        second_line = horzcat(mass_matrix_vu, mass_matrix_vv)

        return vertcat(first_line, second_line)

    def partitioned_non_linear_effect(self, q: MX, qdot: MX) -> MX:
        non_linear_effect = self.non_linear_effects()(q, qdot, [])
        non_linear_effect_u = non_linear_effect[self._independent_joint_index]
        non_linear_effect_v = non_linear_effect[self._dependent_joint_index]

        return vertcat(non_linear_effect_u, non_linear_effect_v)

    def partitioned_q(self, q: MX) -> MX:
        q_u = q[self._independent_joint_index]
        q_v = q[self._dependent_joint_index]

        return vertcat(q_u, q_v)

    def partitioned_qdot(self, qdot: MX) -> MX:
        qdot_u = qdot[self._independent_joint_index]
        qdot_v = qdot[self._dependent_joint_index]

        return vertcat(qdot_u, qdot_v)

    def partitioned_tau(self, tau: MX) -> MX:
        tau_u = tau[self._independent_joint_index]
        tau_v = tau[self._dependent_joint_index]

        return vertcat(tau_u, tau_v)

    def partitioned_constraints_jacobian(self, q: MX) -> MX:
        constrained_jacobian = self.holonomic_constraints_jacobian(q)
        constrained_jacobian_u = constrained_jacobian[:, self._independent_joint_index]
        constrained_jacobian_v = constrained_jacobian[:, self._dependent_joint_index]

        return horzcat(constrained_jacobian_u, constrained_jacobian_v)

    @cache_function
    def partitioned_forward_dynamics(self) -> Function:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        q = self.compute_q()(self.q_u, self.q_v_init)
        qddot_u = self.partitioned_forward_dynamics_full()(q, self.qdot_u, self.tau)

        casadi_fun = Function(
            "partitioned_forward_dynamics",
            [self.q_u, self.qdot_u, self.q_v_init, self.tau],
            [qddot_u],
            ["q_u", "qdot_u", "q_v_init", "tau"],
            ["qddot_u"],
        )
        return casadi_fun

    @cache_function
    def partitioned_forward_dynamics_with_qv(self) -> Function:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        q = self.state_from_partition(self.q_u, self.q_v)
        qddot_u = self.partitioned_forward_dynamics_full()(q, self.qdot_u, self.tau)

        casadi_fun = Function(
            "partitioned_forward_dynamics",
            [self.q_u, self.q_v, self.qdot_u, self.tau],
            [qddot_u],
            ["q_u", "q_v", "qdot_u", "tau"],
            ["qddot_u"],
        )

        return casadi_fun

    @cache_function
    def partitioned_forward_dynamics_full(self) -> Function:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """

        # compute q and qdot
        q = self.q
        qdot = self.compute_qdot()(q, self.qdot_u)
        tau = self.tau

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
        non_linear_effect = self.partitioned_non_linear_effect(q, qdot)
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

        casadi_fun = Function(
            "partitioned_forward_dynamics",
            [self.q, self.qdot_u, self.tau],
            [qddot_u],
            ["q", "qdot_u", "tau"],
            ["qddot_u"],
        )

        return casadi_fun

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
                slicing = slice(self._independent_joint_index.index(i), self._independent_joint_index.index(i) + 1)
                q = vertcat(q, state_u[slicing, :])
            else:
                slicing = slice(self._dependent_joint_index.index(i), self._dependent_joint_index.index(i) + 1)
                q = vertcat(q, state_v[slicing, :])

        return q

    def check_state_u_size(self, state_u):
        if state_u.shape[0] != self.nb_independent_joints:
            raise ValueError(f"Length of state u size should be: {self.nb_independent_joints}. Got: {state_u.shape[0]}")

    def check_state_v_size(self, state_v):
        if state_v.shape[0] != self.nb_dependent_joints:
            raise ValueError(f"Length of state v size should be: {self.nb_dependent_joints}. Got: {state_v.shape[0]}")

    @cache_function
    def compute_q_v(self) -> Function:
        """
        Compute the dependent joint positions (q_v) from the independent joint positions (q_u).
        """
        q_v_sym = MX.sym("q_v_sym", self.nb_dependent_joints)
        q_u_sym = MX.sym("q_u_sym", self.q_u.shape[0], self.q_u.shape[1])
        q = self.state_from_partition(q_u_sym, q_v_sym)
        mx_residuals = self.holonomic_constraints(q)

        ifcn_input = (self.q_v_init, self.q_u)
        residuals = Function(
            "final_states_residuals",
            [q_v_sym, q_u_sym],
            [mx_residuals],
        ).expand()

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": self._newton_tol}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)
        v_opt = ifcn(*ifcn_input)

        casadi_fun = Function("compute_q_v", [self.q_u, self.q_v_init], [v_opt], ["q_u", "q_v_init"], ["q_v"])
        return casadi_fun

    @cache_function
    def compute_q(self) -> Function:
        """
        If you don't know what to put as a q_v_init, use zeros.
        """
        q_v = self.compute_q_v()(self.q_u, self.q_v_init)
        biorbd_return = self.state_from_partition(self.q_u, q_v)
        casadi_fun = Function("compute_q", [self.q_u, self.q_v_init], [biorbd_return], ["q_u", "q_v_init"], ["q"])
        return casadi_fun

    @cache_function
    def compute_qdot_v(self) -> Function:
        coupling_matrix_vu = self.coupling_matrix(self.q)
        biorbd_return = coupling_matrix_vu @ self.qdot_u
        casadi_fun = Function("compute_qdot_v", [self.q, self.qdot_u], [biorbd_return], ["q", "qdot_u"], ["qdot_v"])
        return casadi_fun

    @cache_function
    def _compute_qdot_v(self) -> Function:
        q = self.compute_q()(self.q_u, self.q_v_init)
        biorbd_return = self.compute_qdot_v()(q, self.qdot_u)
        casadi_fun = Function(
            "compute_qdot_v",
            [self.q_u, self.qdot_u, self.q_v_init],
            [biorbd_return],
            ["q_u", "qdot_u", "q_v_init"],
            ["qdot_v"],
        )
        return casadi_fun

    @cache_function
    def compute_qdot(self) -> Function:
        qdot_v = self.compute_qdot_v()(self.q, self.qdot_u)
        biorbd_return = self.state_from_partition(self.qdot_u, qdot_v)
        casadi_fun = Function("compute_qdot", [self.q, self.qdot_u], [biorbd_return], ["q", "qdot_u"], ["qdot"])
        return casadi_fun

    @cache_function
    def compute_qddot_v(self) -> Function:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
        coupling_matrix_vu = self.coupling_matrix(self.q)
        biorbd_return = coupling_matrix_vu @ self.qddot_u + self.biais_vector(self.q, self.qdot)
        casadi_fun = Function(
            "compute_qddot_v", [self.q, self.qdot, self.qddot_u], [biorbd_return], ["q", "qdot", "qddot_u"], ["qddot_v"]
        )
        return casadi_fun

    @cache_function
    def compute_qddot(self) -> Function:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
        qddot_v = self.compute_qddot_v()(self.q, self.qdot, self.qddot_u)
        biorbd_return = self.state_from_partition(self.qddot_u, qddot_v)
        casadi_fun = Function(
            "compute_qddot", [self.q, self.qdot, self.qddot_u], [biorbd_return], ["q", "qdot", "qddot_u"], ["qddot"]
        )
        return casadi_fun

    @cache_function
    def compute_the_lagrangian_multipliers(self) -> Function:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        q = self.compute_q()(self.q_u, self.q_v_init)
        qdot = self.compute_qdot()(q, self.qdot_u)
        qddot_u = self.partitioned_forward_dynamics()(self.q_u, self.qdot_u, self.q_v_init, self.tau)
        qddot = self.compute_qddot()(q, qdot, qddot_u)

        biorbd_return = self._compute_the_lagrangian_multipliers()(q, qdot, qddot, self.tau)
        casadi_fun = Function(
            "compute_the_lagrangian_multipliers",
            [self.q_u, self.qdot_u, self.q_v_init, self.tau],
            [biorbd_return],
            ["q_u", "qdot_u", "q_v_init", "tau"],
            ["lambda"],
        )
        return casadi_fun

    @cache_function
    def _compute_the_lagrangian_multipliers(self) -> Function:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """

        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(self.q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        partitioned_constraints_jacobian_v_t_inv = inv(partitioned_constraints_jacobian_v.T)

        partitioned_mass_matrix = self.partitioned_mass_matrix(self.q)
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        qddot_u = self.qddot[self._independent_joint_index]
        qddot_v = self.qddot[self._dependent_joint_index]

        non_linear_effect = self.partitioned_non_linear_effect(self.q, self.qdot)
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        partitioned_tau = self.partitioned_tau(self.tau)
        partitioned_tau_v = partitioned_tau[self.nb_independent_joints :]

        biorbd_return = partitioned_constraints_jacobian_v_t_inv @ (
            m_vu @ qddot_u + m_vv @ qddot_v + non_linear_effect_v - partitioned_tau_v
        )
        casadi_fun = Function(
            "compute_the_lagrangian_multipliers",
            [self.q, self.qdot, self.qddot, self.tau],
            [biorbd_return],
            ["q", "qdot", "qddot", "tau"],
            ["lambda"],
        )
        return casadi_fun
