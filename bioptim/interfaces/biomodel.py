from casadi import MX
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

    def get_constraints(self):
        """Get the constraints of the model #todo: return constraint forces instead"""

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

    def rigid_contact_axis_idx(self, idx) -> int:
        """Get the rigid contact axis index, todo: to remove"""

    def tau_max(self) -> MX:
        """Get the maximum torque"""

    def rigid_contact_acceleration(self, q, qdot, qddot) -> MX:
        """Get the rigid contact acceleration"""

    def object_homogeneous_matrices(self, q) -> tuple:
        """Get homogeneous matrices of all objects stored in the model"""

    marker_names: tuple[str]
    """Get the marker names"""


class CustomModel(BioModel):
    """
    This is a custom model that inherits from bioptim.BioModel
    This class is made for the user to help him create his own model
    """

    # ---- absolutely needed to be implemented ---- #
    def nb_quaternions(self):
        """Number of quaternion in the model"""
        return 0

    # ---- The rest can raise NotImplementedError ---- #
    def nb_tau(self):
        raise NotImplementedError("nb_tau is not implemented")

    def nb_q(self):
        raise NotImplementedError("nb_q is not implemented")

    def nb_qdot(self):
        raise NotImplementedError("nb_qdot is not implemented")

    def nb_qddot(self):
        raise NotImplementedError("nb_qddot is not implemented")

    def mass(self):
        raise NotImplementedError("mass is not implemented")

    def name_dof(self):
        raise NotImplementedError("name_dof is not implemented")

    def path(self):
        return NotImplementedError("path is not implemented")

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        return NotImplementedError("forward_dynamics is not implemented")

    def gravity(self):
        raise NotImplementedError("gravity is not implemented")

    def nb_segments(self):
        raise NotImplementedError("nb_segments is not implemented")

    def nb_root(self):
        raise NotImplementedError("nb_root is not implemented")

    def nb_rigid_contacts(self):
        raise NotImplementedError("nb_rigid_contacts is not implemented")

    def mass_matrix(self, Q, updateKin=True):
        raise NotImplementedError("mass_matrix is not implemented")

    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        raise NotImplementedError("inverse_dynamics is not implemented")

    def reshape_qdot(self, Q, QDot, k_stab=1):
        raise NotImplementedError("reshape_qdot is not implemented")

    def deep_copy(self, *args):
        raise NotImplementedError("deep_copy is not implemented")

    def add_segment(self, *args):
        raise NotImplementedError("add_segment is not implemented")

    def set_gravity(self, newGravity):
        raise NotImplementedError("set_gravity is not implemented")

    def get_body_biorbd_id(self, segmentName):
        raise NotImplementedError("get_body_biorbd_id is not implemented")

    def get_body_rbdl_id(self, segmentName):
        raise NotImplementedError("get_body_rbdl_id is not implemented")

    def get_body_rbdl_id_to_biorbd_id(self, idx):
        raise NotImplementedError("get_body_rbdl_id_to_biorbd_id is not implemented")

    def get_body_biorbd_id_to_rbdl_id(self, idx):
        raise NotImplementedError("get_body_biorbd_id_to_rbdl_id is not implemented")

    def get_dof_subtrees(self):
        raise NotImplementedError("get_dof_subtrees is not implemented")

    def get_dof_index(self, SegmentName, dofName):
        raise NotImplementedError("get_dof_index is not implemented")

    def update_segment_characteristics(self, idx, characteristics):
        raise NotImplementedError("update_segment_characteristics is not implemented")

    def segment(self, *args):
        raise NotImplementedError("segment is not implemented")

    def segments(self, i):
        raise NotImplementedError("segments is not implemented")

    def dispatched_force(self, *args):
        raise NotImplementedError("dispatched_force is not implemented")

    def update_kinematics_custom(self, Q=None, Qdot=None, Qddot=None):
        raise NotImplementedError("update_kinematics_custom is not implemented")

    def all_global_jcs(self, *args):
        raise NotImplementedError("all_global_jcs is not implemented")

    def global_homogeneous_matrices(self, *args):
        raise NotImplementedError("global_homogeneous_matrices is not implemented")

    def child_homogeneous_matrices(self, *args):
        raise NotImplementedError("child_homogeneous_matrices is not implemented")

    def project_point(self, *args):
        raise NotImplementedError("project_point is not implemented")

    def project_point_jacobian(self, *args):
        raise NotImplementedError("project_point_jacobian is not implemented")

    def center_of_mass(self, Q, updateKin=True):
        raise NotImplementedError("center_of_mass is not implemented")

    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        raise NotImplementedError("CoMbySegmentInMatrix is not implemented")

    def CoMbySegment(self, *args):
        raise NotImplementedError("CoMbySegment is not implemented")

    def center_of_mass_velocity(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("center_of_mass_velocity is not implemented")

    def center_of_mass_acceleration(self, Q, Qdot, Qddot, updateKin=True):
        raise NotImplementedError("comddot is not implemented")

    def comdot_by_segment(self, *args):
        raise NotImplementedError("comdot_by_segment is not implemented")

    def comddot_by_segment(self, *args):
        raise NotImplementedError("comddot_by_segment is not implemented")

    def com_jacobian(self, Q, updateKin=True):
        raise NotImplementedError("com_jacobian is not implemented")

    def mesh_points(self, *args):
        raise NotImplementedError("mesh_points is not implemented")

    def mesh_points_in_matrix(self, Q, updateKin=True):
        raise NotImplementedError("mesh_points_in_matrix is not implemented")

    def mesh_faces(self, *args):
        raise NotImplementedError("mesh_faces is not implemented")

    def mesh(self, *args):
        raise NotImplementedError("mesh is not implemented")

    def angular_momentum(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("angular_momentum is not implemented")

    def mass_matrix_inverse(self, Q, updateKin=True):
        raise NotImplementedError("mass_matrix_inverse is not implemented")

    def calc_angular_momentum(self, *args):
        raise NotImplementedError("calc_angular_momentum is not implemented")

    def calc_segments_angular_momentum(self, *args):
        raise NotImplementedError("calc_segments_angular_momentum is not implemented")

    def body_angular_velocity(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("body_angular_velocity is not implemented")

    def calc_mat_rot_jacobian(self, Q, segmentIdx, rotation, G, updateKin):
        raise NotImplementedError("calc_mat_rot_jacobian is not implemented")

    def jacobian_segment_rot_mat(self, Q, segmentIdx, updateKin):
        raise NotImplementedError("jacobian_segment_rot_mat is not implemented")

    def segment_angular_velocity(self, Q, Qdot, idx, updateKin=True):
        raise NotImplementedError("segment_angular_velocity is not implemented")

    def kinetic_energy(self, Q, QDot, updateKin=True):
        raise NotImplementedError("calc_kinetic_energy is not implemented")

    def potential_energy(self, Q, updateKin=True):
        raise NotImplementedError("calc_potential_energy is not implemented")

    def contact_names(self):
        raise NotImplementedError("contact_names is not implemented")

    def nb_soft_contacts(self):
        return NotImplementedError("nb_soft_contacts is not implemented")

    def soft_contact_names(self):
        raise NotImplementedError("soft_contact_names is not implemented")

    def muscle_names(self):
        raise NotImplementedError("muscle_names is not implemented")

    def torque(self, tau_activations, q, qdot):
        raise NotImplementedError("torque is not implemented")

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints):
        raise NotImplementedError("forward_dynamics_free_floating_base is not implemented")

    def constrained_forward_dynamics(self, *args):
        raise NotImplementedError("constrained_forward_dynamics is not implemented")

    def non_linear_effect(self, Q, QDot, f_ext=None, f_contacts=None):
        raise NotImplementedError("non_linear_effect is not implemented")

    def contact_forces_from_constrained_forward_dynamics(self, Q, QDot, Tau, f_ext=None):
        raise NotImplementedError("contact_forces_from_forward_dynamics_constraints_direct is not implemented")

    def body_inertia(self, Q, updateKin=True):
        raise NotImplementedError("body_inertia is not implemented")

    def qdot_from_impact(self, Q, QDotPre):
        raise NotImplementedError("compute_constraint_impulses_direct is not implemented")

    def check_generalized_dimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        raise NotImplementedError("check_generalized_dimensions is not implemented")

    def state_set(self):
        raise NotImplementedError("state_set is not implemented")

    def muscle_activation_dot(self, muscle_states):
        raise NotImplementedError("activation_dot is not implemented")

    def muscle_joint_torque(self, muscle_states, q, qdot):
        raise NotImplementedError("muscular_joint_torque is not implemented")

    def get_constraints(self):
        raise NotImplementedError("markers is not implemented")

    def markers(self, Q, updateKin=True):
        raise NotImplementedError("markers is not implemented")

    def marker(self, i=None):
        raise NotImplementedError("marker is not implemented")

    def marker_index(self, name):
        raise NotImplementedError("marker_index is not implemented")

    def nb_markers(self):
        raise NotImplementedError("nb_markers is not implemented")

    def segment_index(self, name):
        raise NotImplementedError("segment_index is not implemented")

    def marker_velocities(self, Q, Qdot):
        raise NotImplementedError("marker_velocity is not implemented")

    def tau_max(self):
        raise NotImplementedError("torque_max is not implemented")

    def rigid_contact_acceleration(self, Q, Qdot, Qddot, updateKin=True):
        raise NotImplementedError("rigid_contact_acceleration is not implemented")

    def soft_contact(self, *args):
        raise NotImplementedError("soft_contact is not implemented")

    def rigid_contact_axis_idx(self, *args):
        raise NotImplementedError("rigid_contact_axis_idx is not implemented")

    def object_homogeneous_matrix(self, *args):
        raise NotImplementedError("rt is not implemented")

    def nb_dof(self):
        raise NotImplementedError("nb_dof is not implemented")

    def marker_names(self):
        raise NotImplementedError("marker_names is not implemented")

    def soft_contact_name(self, i):
        raise NotImplementedError("soft_contact_name is not implemented")
