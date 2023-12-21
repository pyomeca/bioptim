import numpy as np
from typing import Protocol, Callable, Any

from casadi import MX, SX
from ...misc.mapping import BiMapping, BiMappingList
from ...limits.path_conditions import Bounds


class BioModel(Protocol):
    """
    This protocol defines the minimal set of attributes and methods a model should possess to access every feature of
    bioptim.
    As a reminder for developers: only necessary attributes and methods should appear here.
    """

    @property
    def name(self) -> str:
        """Get the name of the model"""
        return ""

    def copy(self):
        """copy the model by reloading one"""

    def serialize(self) -> tuple[Callable, dict]:
        """transform the class into a save and load format"""

    @property
    def friction_coefficients(self) -> MX:
        """Get the coefficient of friction to apply to specified elements in the dynamics"""
        return MX()

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

    def homogeneous_matrices_in_global(self, q, segment_id, inverse=False) -> tuple:
        """
        Get the homogeneous matrices of all segments in the world frame,
        such as: P_R0 = T_R0_R1 * P_R1
        with P_R0 the position of any point P in the world frame,
        T_R0_R1 the homogeneous matrix that transform any point in R1 frame to R0.
        P_R1 the position of any point P in the segment R1 frame.
        """

    def homogeneous_matrices_in_child(self, segment_id) -> MX:
        """
        Get the homogeneous matrices of one segment in its parent frame,
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

    def reshape_qdot(self, q, qdot) -> MX:
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

    def forward_dynamics(self, q, qdot, tau, external_forces=None, translational_forces=None) -> MX:
        """compute the forward dynamics"""

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None, translational_forces=None) -> MX:
        """compute the forward dynamics with constraints"""

    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, external_forces=None, translational_forces=None) -> MX:
        """compute the inverse dynamics"""

    def contact_forces_from_constrained_forward_dynamics(
        self, q, qdot, tau, external_forces=None, translational_forces=None
    ) -> MX:
        """compute the contact forces"""

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        """compute the constraint impulses"""

    def muscle_activation_dot(self, muscle_excitations) -> MX:
        """Get the activation derivative of the muscles states"""

    def muscle_joint_torque(self, muscle_states, q, qdot) -> MX:
        """Get the muscular joint torque"""

    def muscle_length_jacobian(self, q) -> MX:
        """Get the muscle velocity"""

    def muscle_velocity(self, q, qdot) -> MX:
        """Get the muscle velocity"""

    def marker(self, q, marker_index: int, reference_frame_idx: int = None) -> MX:
        """Get the position of a marker"""

    def markers(self, q) -> list[MX]:
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

    def marker_velocities(self, q, qdot, reference_index=None) -> list[MX]:
        """Get the marker velocities of the model"""

    def marker_accelerations(self, q, qdot, qddot, reference_index=None) -> list[MX]:
        """Get the marker accelerations of the model"""

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

    def partitioned_forward_dynamics(
        self, q_u, qdot_u, tau, external_forces=None, translational_forces=None, q_v_init=None
    ) -> MX:
        """
        This is the forward dynamics of the model, but only for the independent joints

        Parameters
        ----------
        q_u: MX
            The independent generalized coordinates
        qdot_u: MX
            The independent generalized velocities
        tau: MX
            The generalized torques
        external_forces: MX
            The external forces
        translational_forces: MX
            The translational forces

        Returns
        -------
        MX
            The generalized accelerations

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """

    @staticmethod
    def animate(
        ocp,
        solution: "SolutionData",
        show_now: bool = True,
        tracked_markers: list[np.ndarray, ...] = None,
        **kwargs: Any
    ) -> None | list:
        """
        Animate a solution

        Parameters
        ----------
        solution: SolutionData
            The solution to animate
        show_now: bool
            If the animation should be shown immediately or not
        tracked_markers: list[np.ndarray, ...]
            The tracked markers (3, n_markers, n_frames)
        kwargs: dict
            The options to pass to the animator

        Returns
        -------
        The animator object or None if show_now
        """
