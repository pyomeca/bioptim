"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""
import numpy as np
from casadi import sin, MX

from bioptim import (
    Model,
)


class MyModel(Model):
    """
    This is a custom model that inherits from bioptim.Model
    As Model is an abstract class, we need to implement all the following methods,
    otherwise it will raise an error
    """

    def __init__(self):
        # custom values for the example
        self.com = MX(np.array([-0.0005, 0.0688, -0.9542]))
        self.inertia = MX(0.0391)

    # ---- absolutely needed to be implemented ---- #
    def nb_quat(self):
        """Number of quaternion in the model"""
        return 0

    # ---- Needed for the example ---- #
    def nb_generalized_torque(self):
        return 1

    def nb_q(self):
        return 1

    def nb_qdot(self):
        return 1

    def nb_qddot(self):
        return 1

    def mass(self):
        return 1

    def name_dof(self):
        return ["rotx"]

    def path(self):
        # note: can we do something with this?
        return None

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        # This is where you can implement your own forward dynamics with casadi it your are dealing with mechanical systems
        d = 0  # damping
        L = self.com[2]
        I = self.inertia
        m = self.mass()
        g = 9.81
        return 1 / (I + m * L**2) * (-qdot[0] * d - g * m * L * sin(q[0]) + tau[0])

    # def system_dynamics(self, *args):
    # This is where you can implement your system dynamics with casadi if you are dealing with other systems

    # ---- The rest can raise NotImplementedError ---- #
    def get_gravity(self):
        raise NotImplementedError("get_gravity is not implemented")

    def nb_segment(self):
        raise NotImplementedError("nb_segment is not implemented")

    def nb_root(self):
        raise NotImplementedError("nb_root is not implemented")

    def nb_rigid_contacts(self):
        raise NotImplementedError("nb_rigid_contacts is not implemented")

    def mass_matrix(self, Q, updateKin=True):
        raise NotImplementedError("mass_matrix is not implemented")

    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        raise NotImplementedError("inverse_dynamics is not implemented")

    def compute_qdot(self, Q, QDot, k_stab=1):
        raise NotImplementedError("compute_qdot is not implemented")

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

    def global_jcs(self, *args):
        raise NotImplementedError("global_jcs is not implemented")

    def local_jcs(self, *args):
        raise NotImplementedError("local_jcs is not implemented")

    def project_point(self, *args):
        raise NotImplementedError("project_point is not implemented")

    def project_point_jacobian(self, *args):
        raise NotImplementedError("project_point_jacobian is not implemented")

    def com(self, Q, updateKin=True):
        raise NotImplementedError("com is not implemented")

    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        raise NotImplementedError("CoMbySegmentInMatrix is not implemented")

    def CoMbySegment(self, *args):
        raise NotImplementedError("CoMbySegment is not implemented")

    def comdot(self, Q, Qdot, updateKin=True):
        raise NotImplementedError("comdot is not implemented")

    def comddot(self, Q, Qdot, Qddot, updateKin=True):
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

    def calc_kinetic_energy(self, Q, QDot, updateKin=True):
        raise NotImplementedError("calc_kinetic_energy is not implemented")

    def calc_potential_energy(self, Q, updateKin=True):
        raise NotImplementedError("calc_potential_energy is not implemented")

    def contact_names(self):
        raise NotImplementedError("contact_names is not implemented")

    def nb_soft_contacts(self):
        return 0

    def soft_contact_names(self):
        raise NotImplementedError("soft_contact_names is not implemented")

    def muscle_names(self):
        raise NotImplementedError("muscle_names is not implemented")

    def torque(self, tau_activations, q, qdot):
        raise NotImplementedError("torque is not implemented")

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints):
        raise NotImplementedError("forward_dynamics_free_floating_base is not implemented")

    def forward_dynamics_constraints_direct(self, *args):
        raise NotImplementedError("forward_dynamics_constraints_direct is not implemented")

    def non_linear_effect(self, Q, QDot, f_ext=None, f_contacts=None):
        raise NotImplementedError("non_linear_effect is not implemented")

    def contact_forces_from_forward_dynamics_constraints_direct(self, Q, QDot, Tau, f_ext=None):
        raise NotImplementedError("contact_forces_from_forward_dynamics_constraints_direct is not implemented")

    def body_inertia(self, Q, updateKin=True):
        raise NotImplementedError("body_inertia is not implemented")

    def compute_constraint_impulses_direct(self, Q, QDotPre):
        raise NotImplementedError("compute_constraint_impulses_direct is not implemented")

    def check_generalized_dimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        raise NotImplementedError("check_generalized_dimensions is not implemented")

    def state_set(self):
        raise NotImplementedError("state_set is not implemented")

    def activation_dot(self, muscle_states):
        raise NotImplementedError("activation_dot is not implemented")

    def muscular_joint_torque(self, muscle_states, q, qdot):
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

    def markers_velocity(self, Q, Qdot):
        raise NotImplementedError("marker_velocity is not implemented")

    def torque_max(self):
        raise NotImplementedError("torque_max is not implemented")

    def rigid_contact_acceleration(self, Q, Qdot, Qddot, updateKin=True):
        raise NotImplementedError("rigid_contact_acceleration is not implemented")

    def soft_contact(self, *args):
        raise NotImplementedError("soft_contact is not implemented")