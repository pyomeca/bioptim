from typing import Protocol, Callable

from biorbd_casadi import GeneralizedCoordinates
from casadi import MX, SX, DM, Function
from ..misc.enums import ControlType, QuadratureRule
from ..misc.mapping import BiMapping, BiMappingList
from ..interfaces.biorbd_model import Bounds


class BioModel(Protocol):
    def copy(self):
        """copy the model by reloading one"""

    def serialize(self) -> tuple[Callable, dict]:
        """transform the class into a save and load format"""

    @property
    def gravity(self) -> MX:
        """Get the current gravity applied to the model"""
        return MX()

    def set_gravity(self, new_gravity):
        """Set the gravity vector"""

    @property
    def nb_tau(self) -> int:
        """Get the number of generalized forces"""
        return -1

    @property
    def nb_segments(self) -> int:
        """Get the number of segment"""
        return -1

    def segment_index(self, segment_name) -> int:
        """Get the segment index from its name"""

    @property
    def nb_quaternions(self) -> int:
        """Get the number of quaternion"""
        return -1

    @property
    def nb_dof(self) -> int:
        """Get the number of dof"""
        return -1

    @property
    def nb_q(self) -> int:
        """Get the number of Generalized coordinates"""
        return -1

    @property
    def nb_qdot(self) -> int:
        """Get the number of Generalized velocities"""
        return -1

    @property
    def nb_qddot(self) -> int:
        """Get the number of Generalized accelerations"""
        return -1

    @property
    def nb_root(self) -> int:
        """Get the number of root Dof"""
        return -1

    @property
    def segments(self) -> tuple:
        """Get all segments"""
        return ()

    def homogeneous_matrices_in_global(self, q, reference_idx, inverse=False) -> tuple:
        """
        Get the homogeneous matrices of all segments in the world frame,
        such as: P_R0 = T_R0_R1 * P_R1
        with P_R0 the position of any point P in the world frame,
        T_R0_R1 the homogeneous matrix that transform any point in R1 frame to R0.
        P_R1 the position of any point P in the segment R1 frame.
        """

    def homogeneous_matrices_in_child(self, *args) -> tuple:
        """
        Get the homogeneous matrices of all segments in their parent frame,
        such as: P_R1 = T_R1_R2 * P_R2
        with P_R1 the position of any point P in the segment R1 frame,
        with P_R2 the position of any point P in the segment R2 frame,
        T_R1_R2 the homogeneous matrix that transform any point in R2 frame to R1 frame.
        """

    @property
    def mass(self) -> MX:
        """Get the mass of the model"""
        return MX()

    def center_of_mass(self, q) -> MX:
        """Get the center of mass of the model"""

    def center_of_mass_velocity(self, q, qdot) -> MX:
        """Get the center of mass velocity of the model"""

    def center_of_mass_acceleration(self, q, qdot, qddot) -> MX:
        """Get the center of mass acceleration of the model"""

    def angular_momentum(self, q, qdot) -> MX:
        """Get the angular momentum of the model"""

    def reshape_qdot(self, q, qdot):
        """
        In case, qdot need to be reshaped, such as if one want to get velocities from quaternions.
        Since we don't know if this is the case, this function is always called
        """

    @property
    def name_dof(self) -> tuple[str, ...]:
        """Get the name of the degrees of freedom"""
        return ()

    @property
    def contact_names(self) -> tuple[str, ...]:
        """Get the name of the contacts"""
        return ()

    @property
    def nb_soft_contacts(self) -> int:
        """Get the number of soft contacts"""
        return -1

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        """Get the soft contact names"""
        return ()

    @property
    def muscle_names(self) -> tuple[str, ...]:
        """Get the muscle names"""
        return ()

    def torque(self, activation, q, qdot) -> MX:
        """Get the muscle torque"""

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        """compute the free floating base forward dynamics"""

    def reorder_qddot_root_joints(self, qddot_root, qddot_joints) -> MX:
        """reorder the qddot, from the root dof and the joints dof"""

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None) -> MX:
        """compute the forward dynamics"""

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        """compute the forward dynamics with constraints"""

    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None) -> MX:
        """compute the inverse dynamics"""

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, f_ext=None) -> MX:
        """compute the contact forces"""

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        """compute the constraint impulses"""

    def muscle_activation_dot(self, muscle_excitations) -> MX:
        """Get the activation derivative of the muscles states"""

    def muscle_joint_torque(self, muscle_states, q, qdot) -> MX:
        """Get the muscular joint torque"""

    def marker(self, q, marker_index: int, reference_frame_idx: int = None) -> MX:
        """Get the position of a marker"""

    def markers(self, q) -> MX:
        """Get the markers of the model"""

    @property
    def nb_markers(self) -> int:
        """Get the number of markers of the model"""
        return -1

    def marker_index(self, name) -> int:
        """Get the index of a marker"""

    @property
    def nb_rigid_contacts(self) -> int:
        """Get the number of rigid contacts"""
        return -1

    def marker_velocities(self, q, qdot, reference_index=None) -> MX:
        """Get the marker velocities of the model"""

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        """Get the maximum torque"""

    def rigid_contact_acceleration(self, q, qdot, qddot, contact_index, contact_axis) -> MX:
        """Get the rigid contact acceleration"""

    @property
    def marker_names(self) -> tuple[str, ...]:
        """Get the marker names"""
        return ()

    def soft_contact_forces(self, q, qdot) -> MX:
        """Get the soft contact forces in the global frame"""

    def normalize_state_quaternions(self, x: MX | SX) -> MX | SX:
        """
        Normalize the quaternions of the state

        Parameters
        ----------
        x: MX | SX
            The state to normalize

        Returns
        -------
        The normalized states
        """

    def contact_forces(self, q, qdot, tau, external_forces: list = None) -> MX:
        """
        Easy accessor for the contact forces in contact dynamics

        Parameters
        ----------
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        tau: MX | SX
            The value of tau from "get"
        external_forces: list[np.ndarray]
            The value of external_forces, one for each frame

        Returns
        -------
        The contact forces MX of size [nb_rigid_contacts, 1],
        or [nb_rigid_contacts, n_frames] if external_forces is not None
        """

    def passive_joint_torque(self, q, qdot) -> MX:
        """Get the passive joint torque"""

    def ligament_joint_torque(self, q, qdot) -> MX:
        """Get the ligament joint torque"""

    def bounds_from_ranges(self, variables: str | list[str, ...], mapping: BiMapping | BiMappingList = None) -> Bounds:
        """
        Create bounds from ranges of the model depending on the variable chosen, such as q, qdot, qddot

        Parameters
        ----------
        variables: [str, ...]
           Input or list of input such as ["q"] for bounds on q ranges, ["q", "qdot"] for bounds on q and qdot ranges
            or even ["q", "qdot", qddot"] for bounds on q, qdot and qddot ranges
        mapping: Union[BiMapping, BiMappingList]
            The mapping of q and qdot (if only q, then qdot = q)
        Returns
        -------
        Create the desired bounds
        """

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

    def set_dependencies(self, dependent_joint_index: list, independent_joint_index: list):
        """
        Set the dependencies between the joints of the model

        Parameters
        ----------
        dependent_joint_index: list
            The list of the index of the dependent joints
        independent_joint_index: list
            The list of the index of the independent joints
        """

    @property
    def nb_independent_joints(self) -> int:
        """
        Get the number of independent joints

        Returns
        -------
        int
            The number of independent joints
        """
        return -1

    @property
    def nb_dependent_joints(self) -> int:
        """
        Get the number of dependent joints

        Returns
        -------
        int
            The number of dependent joints
        """
        return -1

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

    @property
    def nb_holonomic_constraints(self) -> int:
        """
        Get the number of holonomic constraints

        Returns
        -------
        int
            The number of holonomic constraints
        """
        return -1

    def holonomic_constraints(self, q: MX) -> MX:
        """
        Get the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The holonomic constraints
        """

    def holonomic_constraints_jacobian(self, q: MX) -> MX:
        """
        Get the jacobian of the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The holonomic constraints jacobian
        """

    def holonomic_constraints_derivative(self, q: MX, qdot: MX) -> MX:
        """
        Get the derivative of the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities

        Returns
        -------
        MX
            The holonomic constraints derivative
        """

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX) -> MX:
        """
        Get the double derivative of the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        qddot: MX
            The generalized accelerations

        Returns
        -------
        MX
            The holonomic constraints double derivative
        """

    def partitioned_mass_matrix(self, q: MX) -> MX:
        """
        This function returns the partitioned mass matrix, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned mass matrix, reordered in function independent and dependent joints
        """

    def partitioned_non_linear_effect(self, q: MX, qdot: MX, f_ext=None, f_contacts=None) -> MX:
        """
        This function returns the partitioned non-linear effect, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        f_ext: MX
            The external forces
        f_contacts: MX
            The contact forces
        """

    def partitioned_q(self, q: MX) -> MX:
        """
        This function returns the partitioned q, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned q, reorder in function independent and dependent joints
        """

    def partitioned_qdot(self, qdot: MX) -> MX:
        """
        This function returns the partitioned qdot, reordered in function independent and dependent joints

        Parameters
        ----------
        qdot: MX
            The generalized velocities

        Returns
        -------
        MX
            The partitioned qdot, reordered in function independent and dependent joints
        """

    def partitioned_tau(self, tau: MX) -> MX:
        """
        This function returns the partitioned tau, reordered in function independent and dependent joints

        Parameters
        ----------
        tau: MX
            The generalized torques

        Returns
        -------
        MX
            The partitioned tau, reordered in function independent and dependent joints
        """

    def partitioned_constrained_jacobian(self, q: MX) -> MX:
        """
        This function returns the partitioned constrained jacobian, reordered in function independent and dependent
        joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned constrained jacobian, reordered in function independent and dependent joints
        """

    def forward_dynamics_constrained_independent(
        self, u: MX, udot: MX, tau: MX, external_forces=None, f_contacts=None
    ) -> MX:
        """
        This is the forward dynamics of the model, but only for the independent joints

        Parameters
        ----------
        u: MX
            The independent generalized coordinates
        udot: MX
            The independent generalized velocities
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

    def coupling_matrix(self, q: MX) -> MX:
        """
        Compute the coupling matrix, denoted Bvu in the paper :

        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        """

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        """
        Compute the biais vector, denoted b in the paper :

        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        The right term of the equation (15) in the paper.

        """

    def q_from_u_and_v(self, u: MX, v: MX) -> MX:
        """
        Compute the generalized coordinates from the independent and dependent joint coordinates

        Parameters
        ----------
        u: MX
            The independent joint coordinates
        v: MX
            The dependent joint coordinates

        Returns
        -------
        MX
            The generalized coordinates
        """

    def compute_v_from_u(self, u: MX) -> MX:
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! Symbolic version of the function

        Parameters
        ----------
        u: MX
            The generalized coordinates

        Returns
        -------
        MX
            The dependent joint
        """

    def compute_v_from_u_numeric(self, u: DM, v_init=None) -> DM:
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! Numeric version of the function

        Parameters
        ----------
        u: DM
            The generalized coordinates
        v_init: DM
            The initial guess for the dependent joint

        Returns
        -------
        DM
            The numerical values of the dependent joint for a given independent joint u
        """

    def partitioned_forward_dynamics(self, u, udot, tau, external_forces=None, f_contacts=None) -> MX:
        """not used"""

    def dae_inverse_dynamics(
        self, q, qdot, qddot, tau, lagrange_multipliers, external_forces=None, f_contacts=None
    ) -> MX:
        """
        Compute the inverse dynamics of the model
        Ax-b = 0
        """

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
        lambdasN: MX | SX = None,
    ):
        """
        Compute the initial states of the system from the initial position and velocity.
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        indications given just before the equation (18) for p0 and pN.
        """
