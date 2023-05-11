from typing import Protocol, Callable, ClassVar

from casadi import MX, SX
from ..misc.mapping import BiMapping, BiMappingList
from ..interfaces.biorbd_model import Bounds


class BioModel(Protocol):
    def copy(self):
        """copy the model by reloading one"""

    def serialize(self) -> tuple[Callable, dict]:
        """transform the class into a save and load format"""

    gravity: MX | ClassVar

    def set_gravity(self, new_gravity):
        """Set the gravity vector"""

    nb_tau: int | ClassVar
    """Get the number of generalized forces"""

    nb_segments: int | ClassVar
    """Get the number of segment"""

    def segment_index(self, segment_name) -> int:
        """Get the segment index from its name"""

    nb_quaternions: int | ClassVar
    """Get the number of quaternion"""

    nb_dof: int | ClassVar
    """Get the number of dof"""

    nb_q: int | ClassVar
    """Get the number of Q"""

    nb_qdot: int | ClassVar
    """Get the number of Qdot"""

    nb_qddot: int | ClassVar
    """Get the number of Qddot"""

    nb_root: int | ClassVar
    """Get the number of root Dof"""

    segments: tuple | ClassVar
    """Get all segments"""

    def homogeneous_matrices_in_global(self, q, reference_idx, inverse=False) -> tuple:
        """
        Get the homogeneous matrices of all segments in the world frame,
        such as: P_R0 = T_R0_R1 * P_R1
        with P_R0 the position of any point P in the world frame,
        T_R0_R1 the homogeneous matrix that transform any point in R1 frame to R0.
        P_R1 the position of any point P in the segment R1 frame.
        """

    def homogeneous_matrices_in_child(self) -> tuple:
        """
        Get the homogeneous matrices of all segments in their parent frame,
        such as: P_R1 = T_R1_R2 * P_R2
        with P_R1 the position of any point P in the segment R1 frame,
        with P_R2 the position of any point P in the segment R2 frame,
        T_R1_R2 the homogeneous matrix that transform any point in R2 frame to R1 frame.
        """

    mass: MX | ClassVar
    """Get the mass of the model"""

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

    name_dof: tuple[str, ...] | ClassVar
    """Get the name of the degrees of freedom"""

    contact_names: tuple[str, ...] | ClassVar
    """Get the name of the contacts"""

    nb_soft_contacts: int | ClassVar
    """Get the number of soft contacts"""

    soft_contact_names: tuple[str, ...] | ClassVar
    """Get the soft contact names"""

    muscle_names: tuple[str, ...] | ClassVar
    """Get the muscle names"""

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

    nb_markers: int | ClassVar
    """Get the number of markers of the model"""

    def marker_index(self, name) -> int:
        """Get the index of a marker"""

    nb_rigid_contacts: int | ClassVar
    """Get the number of rigid contacts"""

    def marker_velocities(self, q, qdot, reference_index=None) -> MX:
        """Get the marker velocities of the model"""

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        """Get the maximum torque"""

    def rigid_contact_acceleration(self, q, qdot, qddot, contact_index, contact_axis) -> MX:
        """Get the rigid contact acceleration"""

    marker_names: tuple[str, ...] | ClassVar
    """Get the marker names"""

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
