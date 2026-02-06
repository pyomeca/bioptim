from typing import Callable

import biorbd_casadi as biorbd
import numpy as np
import scipy.linalg as la
from biorbd_casadi import (
    GeneralizedCoordinates,
)
from casadi import MX, DM, vertcat, horzcat, Function, solve, rootfinder, inv, nlpsol, jacobian

from .biorbd_model import BiorbdModel
from ...models.protocols.holonomic_constraints import HolonomicConstraintsList
from ...optimization.parameters import ParameterList

from ...misc.parameters_types import (
    Str,
    Float,
    NpArray,
    IntListOptional,
    AnyList,
)
from ..utils import cache_function


class HolonomicBiorbdModel(BiorbdModel):
    """
    A biomechanical model with holonomic constraints for constrained multibody dynamics.

    This class extends BiorbdModel to support custom holonomic constraints, enabling the
    simulation and optimization of mechanical systems with geometric restrictions (e.g.,
    closed kinematic chains, contact constraints, or enforced alignments).

    The class provides two formulations for constrained dynamics:

    1. **Full-coordinate formulation** (via Lagrange multipliers):
       Solves: M(q)q̈ + h(q,q̇) = τ + Jᵀλ subject to Φ(q) = 0

    2. **Reduced-coordinate formulation** (partitioned dynamics):
       Eliminates dependent coordinates by partitioning q = [q_u; q_v] where q_v
       is determined by q_u through the constraints, resulting in a smaller system.

    Attributes
    ----------
    _newton_tol : float, default=1e-10
        Convergence tolerance for Newton's method when solving constraint equations.
        Used in compute_q_v() to find dependent coordinates from independent ones.

    _holonomic_constraints : list[Function]
        List of CasADi Functions representing the constraint equations Φ(q) = 0.
        Each function maps q → constraint_residual.

    _holonomic_constraints_jacobians : list[Function]
        List of CasADi Functions for constraint Jacobians J(q) = ∂Φ/∂q.
        Each function maps q → Jacobian_matrix.

    _holonomic_constraints_bias : list[Function]
        List of CasADi Functions for constraint bias terms J̇q̇.
        Each function maps (q, q̇) → bias_vector, representing velocity-dependent
        accelerations computed via the Hessian method.

    stabilization : bool, default=False
        Whether to enable Baumgarte constraint stabilization in forward dynamics.
        When True, adds feedback terms to reduce constraint drift:
            Φ̈ = Jq̈ + J̇q̇ + αΦ + βJ̇ = 0

    alpha : float, default=0.01
        Baumgarte stabilization gain for position-level constraint errors.
        Higher values increase stiffness but may cause numerical issues.
        Only active when stabilization=True.

    beta : float, default=0.01
        Baumgarte stabilization gain for velocity-level constraint errors.
        Acts as damping on constraint violations.
        Only active when stabilization=True.

    _dependent_joint_index : list[int]
        Indices of dependent (constrained) generalized coordinates q_v.
        These coordinates are determined from independent coordinates via constraints.
        Must be sorted in ascending order.

    _independent_joint_index : list[int]
        Indices of independent (unconstrained) generalized coordinates q_u.
        These are the minimal coordinates needed to describe the system state.
        Must be sorted in ascending order.
        Default: all coordinates [0, 1, ..., nb_q-1] if not partitioned.

    _cached_functions : dict
        Internal cache for compiled CasADi Functions to avoid recomputation.
        Populated by the @cache_function decorator.

    Parameters
    ----------
    bio_model : str | biorbd.Model
        Path to the bioMod file or a biorbd.Model instance.
    friction_coefficients : np.ndarray, optional
        Friction coefficients for contact dynamics (inherited from BiorbdModel).
    parameters : ParameterList, optional
        Model parameters (currently not supported with holonomic constraints).
    **kwargs
        Additional arguments passed to BiorbdModel.__init__().

    Notes
    -----
    - Constraints must be set via set_holonomic_configuration() before use
    - The partitioning q = [q_u; q_v] is optional but required for reduced dynamics
    - For reduced dynamics, J_v (dependent Jacobian block) must be invertible
    - The class automatically verifies invertibility via check_dependant_jacobian()

    Mathematical Background
    -----------------------
    Holonomic constraints are geometric restrictions of the form Φ(q) = 0.
    Taking time derivatives:
        - Velocity level: Φ̇ = J(q)q̇ = 0
        - Acceleration level: Φ̈ = J(q)q̈ + J̇(q)q̇ = 0

    The bias term J̇q̇ is computed using the Hessian method:
        (J̇q̇)ᵢ = Σⱼ Σₖ (∂Jᵢⱼ/∂qₖ) q̇ₖ q̇ⱼ = q̇ᵀ Hᵢ q̇

    For partitioned dynamics, the coupling relationships are:
        q̇_v = B_vu q̇_u  where  B_vu = -J_v⁻¹ J_u
        q̈_v = B_vu q̈_u + b_v  where  b_v = -J_v⁻¹(J̇q̇)

    Examples
    --------
    Basic setup with marker superimposition constraint:

    >>> from bioptim import HolonomicBiorbdModel, HolonomicConstraintsList, HolonomicConstraintsFcn
    >>>
    >>> model = HolonomicBiorbdModel("my_model.bioMod")
    >>>
    >>> # Define constraints
    >>> constraints = HolonomicConstraintsList()
    >>> constraints.add(
    ...     "marker_constraint",
    ...     constraints_fcn=HolonomicConstraintsFcn.superimpose_markers,
    ...     marker_1="hand",
    ...     marker_2="target"
    ... )
    >>>
    >>> # Configure with partitioning
    >>> model.set_holonomic_configuration(
    ...     constraints_list=constraints,
    ...     independent_joint_index=[0, 1, 2],
    ...     dependent_joint_index=[3, 4, 5]
    ... )

    Using reduced-coordinate forward dynamics:

    >>> # Only need independent coordinates
    >>> q_u = np.array([0.1, 0.2, 0.3])
    >>> qdot_u = np.array([0.01, 0.02, 0.03])
    >>> tau = np.zeros(model.nb_q)
    >>>
    >>> # Compute full state from independent coordinates
    >>> q = model.compute_q()(q_u, np.zeros(3))  # q_v_init = zeros
    >>>
    >>> # Compute accelerations (reduced system)
    >>> qddot_u = model.partitioned_forward_dynamics()(q_u, qdot_u, np.zeros(3), tau)

    See Also
    --------
    BiorbdModel : Parent class for unconstrained biomechanical models
    HolonomicConstraintsList : Container for defining multiple constraints
    HolonomicConstraintsFcn : Library of predefined constraint types

    References
    ----------
    .. [1] Docquier, N., Poncelet, A., and Fisette, P. (2013).
           ROBOTRAN: a powerful symbolic generator of multibody models.
           Mech. Sci., 4, 199–219. https://doi.org/10.5194/ms-4-199-2013
    .. [2] Baumgarte, J. (1972). Stabilization of constraints and integrals of motion
           in dynamical systems. Computer Methods in Applied Mechanics and Engineering.
    """

    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        friction_coefficients: NpArray = None,
        parameters: ParameterList = None,
        **kwargs,
    ):
        super().__init__(
            bio_model=bio_model, friction_coefficients=friction_coefficients, parameters=parameters, **kwargs
        )
        self._newton_tol = 1e-10
        self._holonomic_constraints = []
        self._holonomic_constraints_jacobians = []
        self._holonomic_constraints_bias = []
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
            constraint_fcn = constraints_list[constraints_name]["constraints_fcn"]
            extra_arguments = constraints_list[constraints_name]["extra_arguments"]
            constraint, constraint_jacobian, constraint_bias = constraint_fcn(model=self, **extra_arguments)
            self._add_holonomic_constraint(
                constraint,
                constraint_jacobian,
                constraint_bias,
            )

        if dependent_joint_index and independent_joint_index:
            self.check_dependant_jacobian()

        self._holonomic_symbolic_variables()

    def check_dependant_jacobian(self):
        """
        Check the declared partitioning.
        """
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
        constraints_bias: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        self._holonomic_constraints.append(constraints)
        self._holonomic_constraints_jacobians.append(constraints_jacobian)
        self._holonomic_constraints_bias.append(constraints_bias)

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
        """
        Compute the time derivative of the holonomic constraints at the velocity level.

        For holonomic constraints Φ(q) = 0, this computes:
            Φ̇ = J(q)q̇

        where J(q) = ∂Φ/∂q is the constraint Jacobian.

        Parameters
        ----------
        q : MX
            Generalized coordinates, shape (n × 1).
        qdot : MX
            Generalized velocities, shape (n × 1).

        Returns
        -------
        MX
            Time derivative of constraints Φ̇, shape (m × 1) where m is the number of constraints.
        """
        return self.holonomic_constraints_jacobian(q) @ qdot

    def holonomic_constraints_bias(self, q: MX, qdot: MX, parameters: MX = None) -> MX:
        """
        Compute the bias vector J̇q̇ for the holonomic constraint acceleration equation.

        This method evaluates the velocity-dependent acceleration term that appears in the
        second time derivative of the holonomic constraints:
            Φ̈ = J(q)q̈ + J̇(q)q̇ = 0

        The bias term J̇q̇ is computed using the Hessian method for each constraint.

        Mathematical Background
        -----------------------
        For constraints Φ(q) = 0, the acceleration-level equation is:
            d²Φ/dt² = (∂Φ/∂q)q̈ + d/dt(∂Φ/∂q)q̇ = Jq̈ + J̇q̇ = 0

        The bias vector represents the quadratic velocity terms:
            (J̇q̇)ᵢ = Σⱼ Σₖ (∂Jᵢⱼ/∂qₖ) q̇ₖ q̇ⱼ = q̇ᵀ Hᵢ q̇

        where Hᵢ is the Hessian of the i-th constraint.

        Parameters
        ----------
        q : MX
            Generalized coordinates, shape (n × 1).
        qdot : MX
            Generalized velocities, shape (n × 1).

        Returns
        -------
        MX
            Bias vector J̇q̇, shape (m × 1) where m is the total number of holonomic constraints.

        See Also
        --------
        holonomic_constraints_double_derivative : Full acceleration-level constraint equation
        constrained_forward_dynamics : Uses this bias term in the constrained dynamics
        bias_vector : Partitioned version used in reduced-coordinate formulation
        """
        if parameters:
            return vertcat(*[b(q, qdot, parameters) for b in self._holonomic_constraints_bias])
        else:
            return vertcat(*[b(q, qdot) for b in self._holonomic_constraints_bias])

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX, parameters: MX = None) -> MX:
        """
        Compute the second time derivative of the holonomic constraints (acceleration level).

        For holonomic constraints Φ(q) = 0, this computes:
            Φ̈ = J(q)q̈ + J̇(q)q̇

        where:
            - J(q) = ∂Φ/∂q is the constraint Jacobian
            - J̇q̇ is the bias vector (velocity-dependent acceleration term)

        This equation must equal zero for the constraints to be satisfied at the acceleration level,
        which is enforced in constrained dynamics formulations.

        Parameters
        ----------
        q : MX
            Generalized coordinates, shape (n × 1).
        qdot : MX
            Generalized velocities, shape (n × 1).
        qddot : MX
            Generalized accelerations, shape (n × 1).

        Returns
        -------
        MX
            Second time derivative Φ̈ = Jq̈ + J̇q̇, shape (m × 1) where m is the number of constraints.

        See Also
        --------
        holonomic_constraints_bias : Computes the J̇q̇ term
        constrained_forward_dynamics : Enforces Φ̈ = 0 to solve for q̈ and constraint forces
        """
        if parameters:
            return vertcat(
                *[
                    J(q, parameters) @ qddot + b(q, qdot, parameters)
                    for J, b in zip(self._holonomic_constraints_jacobians, self._holonomic_constraints_bias)
                ]
            )
        else:
            return vertcat(
                *[
                    J(q) @ qddot + Jdot_qdot(q, qdot)
                    for J, Jdot_qdot in zip(self._holonomic_constraints_jacobians, self._holonomic_constraints_bias)
                ]
            )

    @cache_function
    def constrained_forward_dynamics(self) -> Function:
        """
        Compute forward dynamics for a system with holonomic constraints using Lagrange multipliers.

        This method solves the constrained equations of motion:
            M(q)q̈ + h(q,q̇) = τ + Jᵀλ
            J(q)q̈ + J̇(q)q̇ = 0

        where:
            - M(q) is the mass/inertia matrix
            - h(q,q̇) are the nonlinear effects (Coriolis, centrifugal, gravity)
            - τ are the applied generalized forces
            - J(q) is the constraint Jacobian
            - λ are the Lagrange multipliers (constraint forces)

        Mathematical Formulation
        ------------------------
        The augmented system is solved as a linear system:
            [M   Jᵀ] [q̈]   [τ - h      ]
            [J   0 ] [λ] = [-J̇q̇       ]

        Optional Baumgarte stabilization can be enabled to reduce constraint drift:
            [-J̇q̇ - αΦ - βJ̇]

        where α and β are stabilization gains.

        Returns
        -------
        Function
            CasADi Function with signature:
                Inputs: q, qdot, tau, parameters
                Output: qddot (generalized accelerations satisfying constraints)

        Notes
        -----
        - The method uses symbolic QR decomposition for numerical stability
        - Lagrange multipliers are not returned but can be computed separately
        - For reduced-coordinate formulations, use partitioned_forward_dynamics instead

        See Also
        --------
        partitioned_forward_dynamics : Reduced-coordinate formulation (more efficient)
        holonomic_constraints_bias : Computes the J̇q̇ bias term

        References
        ----------
        .. [1] Baumgarte, J. (1972). Stabilization of constraints and integrals of motion
               in dynamical systems. Computer Methods in Applied Mechanics and Engineering.
        """

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

        bias = -self.holonomic_constraints_bias(self.q, self.qdot)
        if self.stabilization:
            bias -= self.alpha * self.holonomic_constraints(
                self.q
            ) + self.beta * self.holonomic_constraints_derivative(self.q, self.qdot)

        tau_augmented = vertcat(tau_augmented, bias)

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
        ROBOTRAN: a powerful symbolic generator of multibody models, Mech. Sci., 4, 199–219,
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
        Compute forward dynamics in reduced (partitioned) coordinates with explicit dependency handling.

        This method eliminates the dependent coordinates q_v by expressing the dynamics entirely
        in terms of the independent coordinates q_u. The result is a reduced-order system that
        automatically satisfies the holonomic constraints.

        Mathematical Formulation
        ------------------------
        Starting from the full constrained dynamics:
            M(q)q̈ + h(q,q̇) = τ + Jᵀλ
            Φ̈ = J(q)q̈ + J̇q̇ = 0

        Partition coordinates as q = [q_u; q_v] and the mass matrix:
            M = [M_uu  M_uv]
                [M_vu  M_vv]

        Using the constraint relationships:
            q̇_v = B_vu q̇_u
            q̈_v = B_vu q̈_u + b_v

        The reduced dynamics becomes:
            M̄ q̈_u = τ̄ - h̄

        where the modified (reduced) mass matrix is:
            M̄ = M_uu + M_uv B_vu + B_vu^T M_vu + B_vu^T M_vv B_vu

        and the modified forces account for the dependent coordinates:
            τ̄ = τ_u + B_vu^T τ_v
            h̄ = h_u + B_vu^T h_v + (M_uv + B_vu^T M_vv) b_v

        Advantages
        ----------
        - Smaller system to solve (n_u DOF instead of n)
        - No constraint drift (constraints satisfied by construction)
        - More efficient than full augmented formulation
        - O(n_u³) instead of O(n³) complexity

        Returns
        -------
        Function
            CasADi Function with signature:
                Inputs: q, qdot_u, tau
                Output: qddot_u (independent coordinate accelerations)

        Notes
        -----
        - Requires q (full coordinates) and qdot_u (independent velocities only)
        - Dependent velocities are computed internally using coupling_matrix
        - The method assumes J_v is invertible (verified during setup)

        See Also
        --------
        partitioned_forward_dynamics : Wrapper that also computes q from q_u
        coupling_matrix : Computes B_vu = -J_v⁻¹ J_u
        bias_vector : Computes b_v = -J_v⁻¹(J̇q̇)
        constrained_forward_dynamics : Full-coordinate formulation with Lagrange multipliers

        References
        ----------
        .. [1] Docquier, N., Poncelet, A., and Fisette, P. (2013).
               ROBOTRAN: a powerful symbolic generator of multibody models.
               Mech. Sci., 4, 199–219. https://doi.org/10.5194/ms-4-199-2013
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
            modified_generalized_forces - second_term @ self.bias_vector(q, qdot) - modified_non_linear_effect
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
        Compute the coupling matrix B_vu relating independent and dependent velocity coordinates.

        The coupling matrix (also denoted as B_vu in the literature) relates the velocities of
        dependent coordinates q̇_v to independent coordinates q̇_u through the constraint equation:
            q̇_v = B_vu · q̇_u

        Mathematical Derivation
        -----------------------
        From the velocity-level constraint equation:
            Φ̇ = J(q)q̇ = 0

        Partitioning the Jacobian J = [J_u | J_v] and coordinates q̇ = [q̇_u; q̇_v]:
            J_u q̇_u + J_v q̇_v = 0

        Solving for q̇_v (assuming J_v is invertible):
            q̇_v = -J_v⁻¹ J_u q̇_u = B_vu q̇_u

        where B_vu = -J_v⁻¹ J_u.

        Parameters
        ----------
        q : MX
            Generalized coordinates, shape (n × 1).

        Returns
        -------
        MX
            Coupling matrix B_vu, shape (n_v × n_u) where:
                - n_v is the number of dependent coordinates
                - n_u is the number of independent coordinates

        Notes
        -----
        - The matrix J_v must be invertible for the coupling matrix to be well-defined
        - This is verified during setup by check_dependant_jacobian()
        - The coupling matrix is used extensively in the partitioned formulation

        See Also
        --------
        compute_qdot_v : Uses this matrix to compute dependent velocities
        bias_vector : Acceleration-level equivalent
        partitioned_forward_dynamics_full : Uses coupling matrix in reduced dynamics

        References
        ----------
        .. [1] Docquier, N., Poncelet, A., and Fisette, P. (2013).
               ROBOTRAN: a powerful symbolic generator of multibody models.
               Mech. Sci., 4, 199–219. https://doi.org/10.5194/ms-4-199-2013
        """
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        partitioned_constraints_jacobian_v_inv = inv(partitioned_constraints_jacobian_v)  # inv_minor otherwise ?

        partitioned_constraints_jacobian_u = partitioned_constraints_jacobian[:, : self.nb_independent_joints]

        return -partitioned_constraints_jacobian_v_inv @ partitioned_constraints_jacobian_u

    def bias_vector(self, q: MX, qdot: MX) -> MX:
        """
        Compute the partitioned bias vector for dependent coordinate accelerations.

        This method computes the velocity-dependent acceleration term that appears in the
        relationship between dependent and independent accelerations:
            q̈_v = B_vu · q̈_u + b_v

        where b_v is the bias vector computed by this method.

        Mathematical Derivation
        -----------------------
        Taking the time derivative of the velocity constraint:
            d/dt(J_u q̇_u + J_v q̇_v) = 0
            J_u q̈_u + J̇_u q̇_u + J_v q̈_v + J̇_v q̇_v = 0

        Rearranging:
            q̈_v = -J_v⁻¹(J_u q̈_u + J̇_u q̇_u + J̇_v q̇_v)
                = B_vu q̈_u - J_v⁻¹(J̇_u q̇_u + J̇_v q̇_v)

        The bias vector is:
            b_v = -J_v⁻¹ · (J̇q̇)

        where J̇q̇ is computed using the Hessian method.

        Parameters
        ----------
        q : MX
            Generalized coordinates, shape (n × 1).
        qdot : MX
            Generalized velocities, shape (n × 1).

        Returns
        -------
        MX
            Bias vector b_v, shape (n_v × 1) where n_v is the number of dependent coordinates.

        Notes
        -----
        This corresponds to equation (15) in the ROBOTRAN paper. The bias vector represents
        the quadratic velocity terms in the acceleration-level constraint equation.

        See Also
        --------
        coupling_matrix : Velocity-level coupling matrix B_vu
        compute_qddot_v : Uses this bias vector to compute dependent accelerations
        holonomic_constraints_bias : Full-coordinate version of the bias term

        References
        ----------
        .. [1] Docquier, N., Poncelet, A., and Fisette, P. (2013).
               ROBOTRAN: a powerful symbolic generator of multibody models.
               Mech. Sci., 4, 199–219. https://doi.org/10.5194/ms-4-199-2013
        """
        partitioned_constraints_jacobian = self.partitioned_constraints_jacobian(q)
        partitioned_constraints_jacobian_v = partitioned_constraints_jacobian[:, self.nb_independent_joints :]
        partitioned_constraints_jacobian_v_inv = inv(partitioned_constraints_jacobian_v)

        return -partitioned_constraints_jacobian_v_inv @ self.holonomic_constraints_bias(q, qdot)

    def state_from_partition(self, state_u: MX, state_v: MX) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic generator of multibody models, Mech. Sci., 4, 199–219,
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
        Compute dependent coordinate positions from independent coordinate positions.

        This function solves the nonlinear holonomic constraint equation:
            Φ(q_u, q_v) = 0

        for the dependent coordinates q_v, given the independent coordinates q_u.

        Solution Method
        ---------------
        Uses Newton's method (via CasADi's rootfinder) to solve the implicit equation.
        The algorithm iteratively refines q_v starting from an initial guess q_v_init until
        the constraint residual ||Φ|| < tolerance.

        Parameters (via CasADi Function)
        ---------------------------------
        q_u : MX
            Independent coordinate positions, shape (n_u × 1).
        q_v_init : MX
            Initial guess for dependent coordinates, shape (n_v × 1).
            Use zeros if no better guess is available.

        Returns (via CasADi Function)
        ------------------------------
        q_v : MX
            Dependent coordinate positions satisfying Φ(q_u, q_v) = 0, shape (n_v × 1).

        Notes
        -----
        - Convergence depends on the quality of q_v_init
        - For time-stepping simulations, use the previous time step's q_v as initial guess
        - Tolerance is controlled by set_newton_tol() (default: 1e-10)
        - This is a nonlinear solve, unlike the linear relationships for velocities/accelerations

        See Also
        --------
        compute_q : Wrapper that reconstructs full coordinates q from q_u
        compute_qdot_v : Velocity-level equivalent (linear)
        set_newton_tol : Adjust convergence tolerance
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
        Reconstruct full generalized coordinates from independent coordinates.

        This function computes the complete set of generalized coordinates q by:
        1. Solving for dependent coordinates: q_v = solve(Φ(q_u, q_v) = 0)
        2. Reassembling: q = [q_u; q_v] in the original coordinate ordering

        Parameters (via CasADi Function)
        ---------------------------------
        q_u : MX
            Independent coordinate positions, shape (n_u × 1).
        q_v_init : MX
            Initial guess for dependent coordinates, shape (n_v × 1).
            Use zeros if no better initial guess is available.

        Returns (via CasADi Function)
        ------------------------------
        q : MX
            Full generalized coordinates satisfying Φ(q) = 0, shape (n × 1).

        Notes
        -----
        - The output q respects the original joint ordering in the model
        - Internally uses compute_q_v() to solve for dependent coordinates
        - For better convergence in simulations, warm-start q_v_init with previous values

        See Also
        --------
        compute_q_v : Solves for dependent coordinates only
        compute_qdot : Velocity-level equivalent
        state_from_partition : Low-level function for reassembling coordinates
        """
        q_v = self.compute_q_v()(self.q_u, self.q_v_init)
        biorbd_return = self.state_from_partition(self.q_u, q_v)
        casadi_fun = Function("compute_q", [self.q_u, self.q_v_init], [biorbd_return], ["q_u", "q_v_init"], ["q"])
        return casadi_fun

    def compute_q_from_u_iterative(self, q_u_array: np.ndarray, q_v_init: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct full coordinate trajectories from independent coordinate trajectories.

        This method is useful for post-processing optimal control solutions that only contain
        independent coordinates (q_u), reconstructing the complete state trajectory including
        dependent coordinates (q_v).

        The function iterates through each time point, solving for q_v at each step and using
        the previous solution as a warm start for the next, which improves convergence and
        computational efficiency.

        Parameters
        ----------
        q_u_array : np.ndarray
            Independent coordinate trajectory, shape (n_u × n_nodes) where:
                - n_u is the number of independent coordinates
                - n_nodes is the number of time points
        q_v_init : np.ndarray, optional
            Initial guess for dependent coordinates at the first node, shape (n_v,) or (n_v × 1).
            If None, uses zeros. For subsequent nodes, the solution from the previous node
            is used as the initial guess.

        Returns
        -------
        np.ndarray
            Full coordinate trajectory, shape (n × n_nodes) where n = n_u + n_v.
            Coordinates are arranged in the original model ordering.

        Examples
        --------
        Reconstruct full trajectory from optimal control solution:

        >>> from bioptim import SolutionMerge
        >>> # Assuming 'sol' is the solution from ocp.solve()
        >>> states = sol.decision_states(to_merge=SolutionMerge.NODES)
        >>> q_u_traj = states["q_u"]  # shape (n_u, n_nodes)
        >>>
        >>> # Reconstruct full coordinates
        >>> q_full = model.compute_q_from_u_iterative(q_u_traj)
        >>> # q_full has shape (model.nb_q, n_nodes)

        Notes
        -----
        - The method uses warm-starting: q_v from node i initializes the solve at node i+1
        - This significantly improves convergence compared to using zeros at each node
        - For the first node, either provide q_v_init or zeros will be used
        - Convergence tolerance is controlled by the model's Newton tolerance (set_newton_tol)

        See Also
        --------
        compute_q : Single-point version (CasADi Function)
        compute_q_v : Computes only dependent coordinates
        set_newton_tol : Adjust convergence tolerance for constraint solving

        Raises
        ------
        ValueError
            If q_u_array has incorrect first dimension (not matching nb_independent_joints)
        """
        # Validate input shape
        if q_u_array.shape[0] != self.nb_independent_joints:
            raise ValueError(
                f"First dimension of q_u_array must match number of independent joints. "
                f"Expected {self.nb_independent_joints}, got {q_u_array.shape[0]}"
            )

        n_nodes = q_u_array.shape[1] if q_u_array.ndim > 1 else 1

        # Handle 1D input
        if q_u_array.ndim == 1:
            q_u_array = q_u_array[:, np.newaxis]

        # Initialize output and warm-start vector
        q_full = np.zeros((self.nb_q, n_nodes))

        if q_v_init is None:
            q_v_init = DM.zeros(self.nb_dependent_joints)
        else:
            q_v_init = DM(q_v_init.flatten())

        # Iterate through time nodes
        for i in range(n_nodes):
            q_u_i = q_u_array[:, i]

            # Solve for dependent coordinates using previous solution as warm start
            q_v_i = self.compute_q_v()(q_u_i, q_v_init).toarray().flatten()

            # Reconstruct full coordinate vector
            q_full[:, i] = self.state_from_partition(q_u_i[:, np.newaxis], q_v_i[:, np.newaxis]).toarray().flatten()

            # Warm-start next iteration with current solution
            q_v_init = DM(q_v_i)

        return q_full

    def compute_all_states_from_u_iterative(
        self, q_u_array: np.ndarray, qdot_u_array: np.ndarray, tau_array: np.ndarray, q_v_init: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct all state trajectories from independent coordinates and controls.

        This method computes the complete state trajectory (positions, velocities, accelerations,
        and Lagrange multipliers) from the independent coordinate trajectories and control inputs.
        It is designed for post-processing optimal control solutions.

        The method iterates through each time point, solving for dependent coordinates and computing
        all derived quantities using the partitioned dynamics formulation. Warm-starting is used to
        improve convergence.

        Parameters
        ----------
        q_u_array : np.ndarray
            Independent coordinate trajectory, shape (n_u × n_nodes).
        qdot_u_array : np.ndarray
            Independent velocity trajectory, shape (n_u × n_nodes).
        tau_array : np.ndarray
            Control torque trajectory for all joints, shape (nb_tau × n_controls).
            Typically n_controls = n_nodes - 1 for piecewise constant controls.
        q_v_init : np.ndarray, optional
            Initial guess for dependent coordinates at the first node, shape (n_v,).
            If None, uses zeros. Subsequent nodes use warm-starting from previous solutions.

        Returns
        -------
        q : np.ndarray
            Full coordinate trajectory, shape (nb_q × n_nodes).
        qdot : np.ndarray
            Full velocity trajectory, shape (nb_q × n_nodes).
        qddot : np.ndarray
            Full acceleration trajectory, shape (nb_q × n_nodes).
        lambdas : np.ndarray
            Lagrange multiplier trajectory, shape (n_v × n_nodes).
            Represents constraint forces in the dependent directions.

        Examples
        --------
        Compute all states from optimal control solution:

        >>> from bioptim import SolutionMerge
        >>> states = sol.decision_states(to_merge=SolutionMerge.NODES)
        >>> controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        >>>
        >>> # Prepare tau array (pad with zeros for final node if needed)
        >>> n_nodes = states["q_u"].shape[1]
        >>> tau = np.zeros((model.nb_tau, n_nodes))
        >>> tau[:, :-1] = controls["tau"]
        >>>
        >>> # Compute all states
        >>> q, qdot, qddot, lambdas = model.compute_all_states_from_u_iterative(
        ...     states["q_u"],
        ...     states["qdot_u"],
        ...     tau
        ... )

        Notes
        -----
        - Uses warm-starting: each node initializes from the previous solution
        - Lagrange multipliers represent constraint forces maintaining the holonomic constraints
        - The tau_array should contain torques for all joints (both independent and dependent)
        - If tau has fewer columns than state nodes, the last column is assumed to be zero

        See Also
        --------
        compute_q_from_u_iterative : Compute only positions
        compute_the_lagrangian_multipliers : Single-point Lagrange multiplier computation
        partitioned_forward_dynamics : Forward dynamics in reduced coordinates

        Raises
        ------
        ValueError
            If array shapes are incompatible with the model dimensions.
        """
        # Validate input shapes
        if q_u_array.shape[0] != self.nb_independent_joints:
            raise ValueError(
                f"First dimension of q_u_array must match number of independent joints. "
                f"Expected {self.nb_independent_joints}, got {q_u_array.shape[0]}"
            )
        if qdot_u_array.shape[0] != self.nb_independent_joints:
            raise ValueError(
                f"First dimension of qdot_u_array must match number of independent joints. "
                f"Expected {self.nb_independent_joints}, got {qdot_u_array.shape[0]}"
            )
        if tau_array.shape[0] != self.nb_tau:
            raise ValueError(
                f"First dimension of tau_array must match number of torques. "
                f"Expected {self.nb_tau}, got {tau_array.shape[0]}"
            )

        n_nodes = q_u_array.shape[1] if q_u_array.ndim > 1 else 1
        n_controls = tau_array.shape[1] if tau_array.ndim > 1 else 1

        # Handle 1D inputs
        if q_u_array.ndim == 1:
            q_u_array = q_u_array[:, np.newaxis]
        if qdot_u_array.ndim == 1:
            qdot_u_array = qdot_u_array[:, np.newaxis]
        if tau_array.ndim == 1:
            tau_array = tau_array[:, np.newaxis]

        # Pad tau with zeros if needed (for final node in piecewise constant control)
        if n_controls < n_nodes:
            tau_padded = np.zeros((self.nb_tau, n_nodes))
            tau_padded[:, :n_controls] = tau_array
            tau_array = tau_padded

        # Initialize outputs
        q = np.zeros((self.nb_q, n_nodes))
        qdot = np.zeros((self.nb_q, n_nodes))
        qddot = np.zeros((self.nb_q, n_nodes))
        lambdas = np.zeros((self.nb_dependent_joints, n_nodes))

        # Initialize warm-start
        if q_v_init is None:
            q_v_init = DM.zeros(self.nb_dependent_joints)
        else:
            q_v_init = DM(q_v_init.flatten())

        # Iterate through time nodes
        for i in range(n_nodes):
            q_u_i = q_u_array[:, i]
            qdot_u_i = qdot_u_array[:, i]
            tau_i = tau_array[:, i]

            # Solve for dependent coordinates
            q_v_i = self.compute_q_v()(q_u_i, q_v_init).toarray().flatten()

            # Reconstruct full state
            q[:, i] = self.state_from_partition(q_u_i[:, np.newaxis], q_v_i[:, np.newaxis]).toarray().flatten()

            # Compute full velocity
            qdot[:, i] = self.compute_qdot()(q[:, i], qdot_u_i).toarray().flatten()

            # Compute independent accelerations from forward dynamics
            qddot_u_i = self.partitioned_forward_dynamics()(q_u_i, qdot_u_i, q_v_init, tau_i).toarray().flatten()

            # Compute full acceleration
            qddot[:, i] = self.compute_qddot()(q[:, i], qdot[:, i], qddot_u_i).toarray().flatten()

            # Compute Lagrange multipliers (constraint forces)
            lambdas[:, i] = (
                self.compute_the_lagrangian_multipliers()(q_u_i[:, np.newaxis], qdot_u_i, q_v_init, tau_i)
                .toarray()
                .flatten()
            )

            # Warm-start next iteration
            q_v_init = DM(q_v_i)

        return q, qdot, qddot, lambdas

    @cache_function
    def compute_qdot_v(self) -> Function:
        """
        Compute dependent coordinate velocities from independent coordinate velocities.

        This function computes the velocities of dependent coordinates q̇_v given the
        velocities of independent coordinates q̇_u, using the constraint relationship:
            q̇_v = B_vu · q̇_u

        where B_vu = -J_v⁻¹ J_u is the coupling matrix.

        This relationship is derived from the velocity-level constraint:
            Φ̇ = J_u q̇_u + J_v q̇_v = 0

        Parameters (via CasADi Function)
        ---------------------------------
        q : MX
            Full generalized coordinates, shape (n × 1).
        qdot_u : MX
            Independent coordinate velocities, shape (n_u × 1).

        Returns (via CasADi Function)
        ------------------------------
        qdot_v : MX
            Dependent coordinate velocities, shape (n_v × 1).

        Notes
        -----
        This is the fundamental velocity relationship in partitioned coordinate formulations.
        It ensures that the velocity-level constraint Φ̇ = 0 is satisfied.

        See Also
        --------
        coupling_matrix : Computes the B_vu matrix used in this relationship
        compute_qddot_v : Acceleration-level equivalent
        compute_qdot : Reconstructs full velocity vector from q̇_u and q̇_v
        """
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
        Compute dependent coordinate accelerations from independent coordinate accelerations.

        This function computes the accelerations of dependent coordinates q̈_v given the
        accelerations of independent coordinates q̈_u, using the constraint relationship:
            q̈_v = B_vu · q̈_u + b_v

        where:
            - B_vu = -J_v⁻¹ J_u is the coupling matrix
            - b_v = -J_v⁻¹(J̇q̇) is the bias vector

        This relationship ensures the acceleration-level constraint Φ̈ = 0 is satisfied.

        Parameters (via CasADi Function)
        ---------------------------------
        q : MX
            Full generalized coordinates, shape (n × 1).
        qdot : MX
            Full generalized velocities, shape (n × 1).
        qddot_u : MX
            Independent coordinate accelerations, shape (n_u × 1).

        Returns (via CasADi Function)
        ------------------------------
        qddot_v : MX
            Dependent coordinate accelerations, shape (n_v × 1).

        Notes
        -----
        Corresponds to equation (17) in the ROBOTRAN paper. This is the acceleration-level
        equivalent of the velocity relationship q̇_v = B_vu q̇_u.

        See Also
        --------
        coupling_matrix : Computes the B_vu matrix
        bias_vector : Computes the b_v bias term
        compute_qdot_v : Velocity-level equivalent
        compute_qddot : Reconstructs full acceleration vector from q̈_u and q̈_v

        References
        ----------
        .. [1] Docquier, N., Poncelet, A., and Fisette, P. (2013).
               ROBOTRAN: a powerful symbolic generator of multibody models.
               Mech. Sci., 4, 199–219. https://doi.org/10.5194/ms-4-199-2013
        """
        coupling_matrix_vu = self.coupling_matrix(self.q)
        biorbd_return = coupling_matrix_vu @ self.qddot_u + self.bias_vector(self.q, self.qdot)
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
