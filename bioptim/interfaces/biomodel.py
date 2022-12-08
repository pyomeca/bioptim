from casadi import MX, SX
import biorbd_casadi as biorbd
from typing import Protocol
from pathlib import Path


class BioModel(Protocol):

    gravity: MX
    """Get the gravity vector"""

    def set_gravity(self, newGravity):
        """Set the gravity vector"""

    nb_tau: int
    """Get the number of generalized forces"""

    nb_segments: int
    """Get the number of segment"""

    def segment_index(self, segmentName) -> int:
        """Get the segment index"""

    nb_quaternions: int
    """Get the number of quaternion"""

    nb_dof: int
    """Get the number of dof"""

    nb_q: int
    """Get the number of Q"""

    nb_qdot: int
    """Get the number of Qdot"""

    nb_qddot: int
    """Get the number of Qddot"""

    nb_root: int
    """Get the number of root Dof"""

    segments: tuple
    """Get all segments"""

    global_homogeneous_matrices: tuple
    """Get the homogeneous matrices of all segments in the world frame"""

    child_homogeneous_matrices: tuple
    """Get the homogeneous matrices of all segments in their parent frame"""

    mass: MX
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

    name_dof: tuple[str]
    """Get the name of the degrees of freedom"""

    contact_names: tuple[str]
    """Get the name of the contacts"""

    nb_soft_contacts: int
    """Get the number of soft contacts"""

    soft_contact_names: tuple[str]
    """Get the soft contact names"""

    def soft_contacts(self, *args):
        # todo: forces from soft contact ?
        """Get the soft contact"""

    muscle_names: tuple[str]
    """Get the muscle names"""

    def torque(self, q, qdot, activation) -> MX:
        """Get the muscle torque"""

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        """compute the free floating base forward dynamics"""

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None) -> MX:
        """compute the forward dynamics"""

    def constrained_forward_dynamics(self, *args) -> MX:
        """compute the forward dynamics with constraints"""

    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None) -> MX:
        """compute the inverse dynamics"""

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, f_ext=None) -> MX:
        """compute the contact forces"""

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        """compute the constraint impulses"""

    def state_set(self):
        """Get the state set of the model #todo: to be deleted but get muscle excitations"""

    def muscle_activation_dot(self, muscle_states) -> MX:
        """Get the activation derivative of the muscles states"""

    def muscle_joint_torque(self, muscle_states, q, qdot) -> MX:
        """Get the muscular joint torque"""

    def markers(self, q) -> MX:
        """Get the markers of the model"""

    nb_markers: int
    """Get the number of markers of the model"""

    def marker_index(self, name) -> int:
        """Get the index of a marker"""

    nb_rigid_contacts: int
    """Get the number of rigid contacts"""

    path: Path
    """Get the path of the model"""

    def marker_velocities(self, q, qdot) -> MX:
        """Get the marker velocities of the model"""

    def reshape_fext_to_fcontact(self, fext: MX) -> biorbd.VecBiorbdVector:
        """Reshape the external forces to contact forces"""

    def tau_max(self) -> tuple[MX, MX]:
        """Get the maximum torque"""

    def rigid_contact_acceleration(self, q, qdot, qddot) -> MX:
        """Get the rigid contact acceleration"""

    def object_homogeneous_matrices(self, q) -> tuple:
        """Get homogeneous matrices of all objects stored in the model"""

    marker_names: tuple[str]
    """Get the marker names"""

    def soft_contact_forces(self, q, qdot) -> MX:
        """Get the soft contact forces in the global frame"""

    def normalize_state_quaternions(self, x: MX | SX) -> MX | SX:
        """
        Normalize the quaternions of the state

        Parameters
        ----------
        x: Union[MX, SX]
            The state to normalize

        Returns
        -------
        The normalized states
        """