import numpy as np
from typing import Protocol, Callable, Any

from casadi import MX, SX, Function
from ...misc.mapping import BiMapping, BiMappingList
from ...limits.path_conditions import Bounds
from ..utils import cache_function

from ...misc.parameters_types import (
    AnyDict,
    AnyTuple,
    Str,
    Int,
    MXList,
    StrTuple,
    StrOrIterable,
    Bool,
    NpArrayListOptional,
    AnyListOptional,
)


class BioModel(Protocol):
    """
    This protocol defines the minimal set of attributes and methods a model should possess to access every feature of
    bioptim.
    As a reminder for developers: only necessary attributes and methods should appear here.
    """

    @property
    def name(self) -> Str:
        """Get the name of the model"""
        return ""

    def copy(self):
        """copy the model by reloading one"""

    def serialize(self) -> tuple[Callable, AnyDict]:
        """transform the class into a save and load format"""

    @property
    def friction_coefficients(self) -> Function:
        """Get the coefficient of friction to apply to specified elements in the dynamics"""
        return MX()

    @cache_function
    def gravity(self) -> Function:
        """Get the current gravity applied to the model"""

    def set_gravity(self, new_gravity):
        """Set the gravity vector"""

    @property
    def nb_tau(self) -> Int:
        """Get the number of generalized forces"""
        return -1

    @property
    def nb_segments(self) -> Int:
        """Get the number of segment"""
        return -1

    def segment_index(self, segment_name) -> Int:
        """Get the segment index from its name"""

    @property
    def nb_quaternions(self) -> Int:
        """Get the number of quaternion"""
        return -1

    @property
    def nb_dof(self) -> Int:
        """Get the number of dof"""
        return -1

    @property
    def nb_q(self) -> Int:
        """Get the number of Generalized coordinates"""
        return -1

    @property
    def nb_qdot(self) -> Int:
        """Get the number of Generalized velocities"""
        return -1

    @property
    def nb_qddot(self) -> Int:
        """Get the number of Generalized accelerations"""
        return -1

    @property
    def nb_root(self) -> Int:
        """Get the number of root Dof"""
        return -1

    @property
    def nb_ligaments(self) -> int:
        return -1

    @property
    def nb_passive_joint_torques(self) -> int:
        return -1

    @property
    def segments(self) -> AnyTuple:
        """Get all segments"""
        return ()

    @cache_function
    def rotation_matrix_to_euler_angles(self, sequence: Str) -> AnyTuple:
        """
        Get the Euler angles from a rotation matrix, in the sequence specified
        args: rotation matrix
        """

    @cache_function
    def mass(self) -> Function:
        """Get the mass of the model"""

    @cache_function
    def rt(self, rt_index) -> Function:
        """
        Get the rototrans matrix of an object (e.g., an IMU) that is placed on the model
        args: q
        """

    @cache_function
    def center_of_mass(self) -> Function:
        """
        Get the center of mass of the model
        args: q
        """

    @cache_function
    def center_of_mass_velocity(self) -> Function:
        """
        Get the center of mass velocity of the model
        args: q, qdot
        """

    @cache_function
    def center_of_mass_acceleration(self) -> Function:
        """
        Get the center of mass acceleration of the model
        args: q, qdot, qddot
        """

    @cache_function
    def angular_momentum(self) -> Function:
        """
        Get the angular momentum of the model
        args: q, qdot
        """

    @cache_function
    def reshape_qdot(self) -> Function:
        """
        In case, qdot need to be reshaped, such as if one want to get velocities from quaternions.
        Since we don't know if this is the case, this function is always called
        args: q, qdot
        """

    @property
    def name_dofs(self) -> StrTuple:
        """Get the name of the degrees of freedom"""
        return ()

    @property
    def rigid_contact_names(self) -> StrTuple:
        """Get the name of the contacts"""
        return ()

    @property
    def nb_soft_contacts(self) -> Int:
        """Get the number of soft contacts"""
        return -1

    @property
    def soft_contact_names(self) -> StrTuple:
        """Get the soft contact names"""
        return ()

    @property
    def muscle_names(self) -> StrTuple:
        """Get the muscle names"""
        return ()

    @cache_function
    def torque(self) -> Function:
        """
        Get the muscle torque
        args: activation, q, qdot
        """

    @cache_function
    def forward_dynamics_free_floating_base(self) -> Function:
        """
        compute the free floating base forward dynamics
        args: q, qdot, qddot_joints
        """

    def reorder_qddot_root_joints(self) -> Function:
        """
        reorder the qddot, from the root dof and the joints dof
        args: qddot_root, qddot_joints
        """

    @cache_function
    def forward_dynamics(self, with_contact=False) -> Function:
        """
        compute the forward dynamics
        args: q, qdot, tau, external_forces
        """

    @cache_function
    def inverse_dynamics(self) -> Function:
        """
        compute the inverse dynamics
        args: q, qdot, qddot, external_forces
        """

    @cache_function
    def contact_forces_from_constrained_forward_dynamics(self) -> Function:
        """
        compute the contact forces
        args: q, qdot, tau, external_forces
        """

    @cache_function
    def qdot_from_impact(self) -> Function:
        """
        compute the constraint impulses
        args: q, qdot_pre_impact
        """

    @cache_function
    def muscle_activation_dot(self) -> Function:
        """
        Get the activation derivative of the muscles states
        args: muscle_excitations, muscle_activations
        """

    @cache_function
    def muscle_joint_torque(self) -> Function:
        """
        Get the muscular joint torque
        args: muscle_states, q, qdot
        """

    @cache_function
    def muscle_length_jacobian(self) -> Function:
        """
        Get the muscle velocity
        args: q
        """

    @cache_function
    def muscle_velocity(self) -> Function:
        """
        Get the muscle velocity
        args: q, qdot
        """

    @cache_function
    def marker(self, marker_index: Int, reference_frame_idx: Int = None) -> Function:
        """
        Get the position of a marker
        args: q
        """

    @cache_function
    def markers(self) -> MXList:
        """
        Get the markers of the model
        args: q
        """

    @property
    def nb_markers(self) -> Int:
        """Get the number of markers of the model"""
        return -1

    def marker_index(self, name) -> Int:
        """Get the index of a marker"""

    @property
    def nb_rigid_contacts(self) -> Int:
        """Get the number of rigid contacts"""
        return -1

    @cache_function
    def markers_velocities(self, reference_index=None) -> Function:
        """
        Get the marker velocities of the model, in the reference frame number reference_index
        args: q, qdot
        """

    @cache_function
    def marker_velocity(self, marker_index=None) -> Function:
        """
        Get the velocity of one marker from the model
        args: q, qdot
        """

    @cache_function
    def markers_accelerations(self, reference_index=None) -> Function:
        """
        Get the marker accelerations of the model, in the reference frame number reference_index
        args: q, qdot, qddot
        """

    @cache_function
    def marker_acceleration(self, marker_index=None) -> Function:
        """
        Get the acceleration of one marker from the model
        args: q, qdot, qddot
        """

    @cache_function
    def tau_max(self) -> Function:
        """
        Get the maximum torque
        args: q, qdot
        """

    @cache_function
    def rigid_contact_acceleration(self, contact_index, contact_axis) -> Function:
        """
        Get the rigid contact acceleration
        args: q, qdot, qddot
        """

    @property
    def marker_names(self) -> StrTuple:
        """Get the marker names"""
        return ()

    @cache_function
    def soft_contact_forces(self) -> Function:
        """
        Get the soft contact forces in the global frame
        args: q, qdot
        """

    @cache_function
    def normalize_state_quaternions(self) -> Function:
        """
        Normalize the quaternions of the state
        args: q (The joint generalized coordinates to normalize)
        """

    @cache_function
    def rigid_contact_forces(self) -> Function:
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

    @cache_function
    def passive_joint_torque(self) -> Function:
        """
        Get the passive joint torque
        args: q, qdot
        """

    @cache_function
    def ligament_joint_torque(self) -> Function:
        """
        Get the ligament joint torque
        args: q, qdot
        """

    def bounds_from_ranges(self, variables: StrOrIterable, mapping: BiMapping | BiMappingList = None) -> Bounds:
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

    @cache_function
    def lagrangian(self) -> Function:
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

    @cache_function
    def partitioned_forward_dynamics(self, q_u, qdot_u, q_v_init, tau) -> Function:
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
        ocp, solution: "SolutionData", show_now: Bool = True, tracked_markers: NpArrayListOptional = None, **kwargs: Any
    ) -> AnyListOptional:
        """
        Animate a solution

        Parameters
        ----------
        solution: SolutionData
            The solution to animate
        show_now: bool
            If the animation should be shown immediately or not
        tracked_markers: list[np.ndarray]
            The tracked markers (3, n_markers, n_frames)
        kwargs: dict
            The options to pass to the animator

        Returns
        -------
        The animator object or None if show_now
        """

    def to_pyorerun_model(self) -> Any:
        """
        Create a pyorerun-compatible model for visualization.

        Each BioModel implementation should return the appropriate pyorerun model type
        (e.g., pyorerun.BiorbdModel for biorbd, pyorerun.PinocchioModel for pinocchio).

        Returns
        -------
        Any
            A pyorerun model suitable for animation
        """

    @property
    def pyorerun_marker_names(self) -> list[str]:
        """
        Get marker names formatted for pyorerun visualization.

        Returns
        -------
        list[str]
            List of marker names as strings
        """
        return []
