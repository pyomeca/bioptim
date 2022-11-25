import biorbd_casadi as biorbd
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def deep_copy(self, *args):
        """Deep copy of the model"""

    @abstractmethod
    def add_segment(self, *args):
        """Add a segment to the model"""

    @abstractmethod
    def get_gravity(self):
        """Get the gravity vector"""

    @abstractmethod
    def set_gravity(self, newGravity):
        """Set the gravity vector"""

    @abstractmethod
    def get_body_biorbd_id(self, segmentName):
        """Get the biorbd id of a body"""

    @abstractmethod
    def get_body_rbdl_id(self, segmentName):
        """Get the rbdl id of a body"""

    @abstractmethod
    def get_body_rbdl_id_to_biorbd_id(self, idx):
        """Get the biorbd id of a body from its rbdl id"""

    @abstractmethod
    def get_body_biorbd_id_to_rbdl_id(self, idx):
        """Get the rbdl id of a body from its biorbd id"""

    @abstractmethod
    def get_dof_subtrees(self):
        """Get the dof sub trees"""

    @abstractmethod
    def get_dof_index(self, SegmentName, dofName):
        """Get the dof index"""

    @abstractmethod
    def nb_generalized_torque(self):
        """Get the number of generalized torque"""

    @abstractmethod
    def nb_segment(self):
        """Get the number of segment"""

    @abstractmethod
    def segment_index(self, segmentName):
        """Get the segment index"""

    @abstractmethod
    def nb_quat(self):
        """Get the number of quaternion"""

    @abstractmethod
    def nb_q(self):
        """Get the number of Q"""

    @abstractmethod
    def nb_qdot(self):
        """Get the number of Qdot"""

    @abstractmethod
    def nb_qddot(self):
        """Get the number of Qddot"""

    @abstractmethod
    def nb_root(self):
        """Get the number of root Dof"""

    @abstractmethod
    def update_segment_characteristics(self, idx, characteristics):
        """Update the segment characteristics"""

    @abstractmethod
    def segment(self, *args):
        """Get a segment"""

    @abstractmethod
    def segments(self, i):
        """Get a segment"""

    @abstractmethod
    def dispatched_force(self, *args):
        """Get the dispatched force"""

    @abstractmethod
    def update_kinematics_custom(self, Q=None, Qdot=None, Qddot=None):
        """Update the kinematics of the model"""

    @abstractmethod
    def all_global_jcs(self, *args):
        """Get all the Rototranslation matrix"""

    @abstractmethod
    def global_jcs(self, *args):
        """Get the Rototranslation matrix"""

    @abstractmethod
    def local_jcs(self, *args):
        """Get the Rototranslation matrix"""

    @abstractmethod
    def project_point(self, *args):
        """Project a point on the segment"""

    @abstractmethod
    def project_point_jacobian(self, *args):
        """Project a point on the segment"""

    @abstractmethod
    def mass(self):
        """Get the mass of the model"""

    @abstractmethod
    def com(self, Q, updateKin=True):
        """Get the center of mass of the model"""

    @abstractmethod
    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        """Get the center of mass of the model"""

    @abstractmethod
    def CoMbySegment(self, *args):
        """Get the center of mass of the model"""

    @abstractmethod
    def comdot(self, Q, Qdot, updateKin=True):
        """Get the center of mass velocity of the model"""

    @abstractmethod
    def comddot(self, Q, Qdot, Qddot, updateKin=True):
        """Get the center of mass acceleration of the model"""

    @abstractmethod
    def comdot_by_segment(self, *args):
        """Get the center of mass velocity of the model"""

    @abstractmethod
    def comddot_by_segment(self, *args):
        """Get the center of mass acceleration of the model"""

    @abstractmethod
    def com_jacobian(self, Q, updateKin=True):
        """Get the center of mass Jacobian of the model"""

    @abstractmethod
    def mesh_points(self, *args):
        """Get the mesh points of the model"""

    @abstractmethod
    def mesh_points_in_matrix(self, Q, updateKin=True):
        """Get the mesh points of the model"""

    @abstractmethod
    def mesh_faces(self, *args):
        """Get the mesh faces of the model"""

    @abstractmethod
    def mesh(self, *args):
        """Get the mesh of the model"""

    @abstractmethod
    def angular_momentum(self, Q, Qdot, updateKin=True):
        """Get the angular momentum of the model"""

    @abstractmethod
    def mass_matrix(self, Q, updateKin=True):
        """Get the mass matrix of the model"""

    @abstractmethod
    def mass_matrix_inverse(self, Q, updateKin=True):
        """Get the inverse of the mass matrix of the model"""

    @abstractmethod
    def calc_angular_momentum(self, *args):
        """Get the angular momentum of the model"""

    @abstractmethod
    def calc_segments_angular_momentum(self, *args):
        """Get the angular momentum of the model"""

    @abstractmethod
    def body_angular_velocity(self, Q, Qdot, updateKin=True):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def calc_mat_rot_jacobian(self, Q, segmentIdx, rotation, G, updateKin):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def jacobian_segment_rot_mat(self, Q, segmentIdx, updateKin):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def compute_qdot(self, Q, QDot, k_stab=1):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def segment_angular_velocity(self, Q, Qdot, idx, updateKin=True):
        """Get the body angular velocity of the model"""

    @abstractmethod
    def calc_kinetic_energy(self, Q, QDot, updateKin=True):
        """Get the kinetic energy of the model"""

    @abstractmethod
    def calc_potential_energy(self, Q, updateKin=True):
        """Get the potential energy of the model"""

    @abstractmethod
    def name_dof(self):
        """Get the name of the dof"""

    @abstractmethod
    def contact_names(self):
        """Get the contact names"""

    @abstractmethod
    def nb_soft_contacts(self):
        """Get the number of soft contacts"""

    @abstractmethod
    def soft_contact_names(self):
        """Get the soft contact names"""

    @abstractmethod
    def muscle_names(self):
        """Get the muscle names"""

    @abstractmethod
    def torque(self, tau_activations, q, qdot):
        """Get the muscle torque"""

    @abstractmethod
    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints):
        """compute the free floating base forward dynamics"""

    @abstractmethod
    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        """compute the forward dynamics"""

    @abstractmethod
    def forward_dynamics_constraints_direct(self, *args):
        """compute the forward dynamics with constraints"""

    @abstractmethod
    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        """compute the inverse dynamics"""

    @abstractmethod
    def non_linear_effect(self, Q, QDot, f_ext=None, f_contacts=None):
        """compute the non linear effect"""

    @abstractmethod
    def contact_forces_from_forward_dynamics_constraints_direct(self, Q, QDot, Tau, f_ext=None):
        """compute the contact forces"""

    @abstractmethod
    def body_inertia(self, Q, updateKin=True):
        """Get the inertia of the model"""

    @abstractmethod
    def compute_constraint_impulses_direct(self, Q, QDotPre):
        """compute the constraint impulses"""

    @abstractmethod
    def check_generalized_dimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        """check the dimensions of the generalized coordinates"""

    @abstractmethod
    def state_set(self):
        """Get the state set of the model"""

    @abstractmethod
    def activation_dot(self, muscle_states):
        """Get the activation derivative"""

    @abstractmethod
    def muscular_joint_torque(self, muscle_states, q, qdot):
        """Get the muscular joint torque"""

    @abstractmethod
    def get_constraints(self):
        """Get the constraints of the model"""

    @abstractmethod
    def markers(self, Q, updateKin=True):
        """Get the markers of the model"""

    @abstractmethod
    def nb_markers(self):
        """Get the number of markers of the model"""

    @abstractmethod
    def marker_index(self, name):
        """Get the index of a marker"""

    @abstractmethod
    def marker(self, i=None):
        """Get the marker index i of the model as function of coordinates"""

    @abstractmethod
    def nb_rigid_contacts(self):
        """Get the number of rigid contacts"""

    @abstractmethod
    def path(self):
        """Get the path of the model"""

    @abstractmethod
    def markers_velocity(self, Q, Qdot):
        """Get the marker velocities of the model"""

    @abstractmethod
    def rigid_contact_axis_idx(self, idx):
        """Get the rigid contact axis index"""

    @abstractmethod
    def torque_max(self):
        """Get the maximum torque"""


class BiorbdModel(Model):
    def __init__(self, biorbd_model: str | biorbd.Model):
        if isinstance(biorbd_model, str):
            self.model = biorbd.Model(biorbd_model)
        else:
            self.model = biorbd_model

    def deep_copy(self, *args):
        return self.model.DeepCopy(*args)

    def add_segment(self, *args):
        return self.model.AddSegment(self, *args)

    def get_gravity(self):
        return self.model.getGravity()

    def set_gravity(self, newGravity):
        return self.model.setGravity(newGravity)

    def get_body_biorbd_id(self, segmentName):
        return self.model.getBodyBiorbdId(segmentName)

    def get_body_rbdl_id(self, segmentName):
        return self.model.getBodyRbdlId(segmentName)

    def get_body_rbdl_id_to_biorbd_id(self, idx):
        return self.model.getBodyRbdlIdToBiorbdId(idx)

    def get_body_biorbd_id_to_rbdl_id(self, idx):
        return self.model.getBodyBiorbdIdToRbdlId(idx)

    def get_dof_subtrees(self):
        return self.model.getDofSubTrees()

    def get_dof_index(self, SegmentName, dofName):
        return self.model.getDofIndex(SegmentName, dofName)

    def nb_generalized_torque(self):
        return self.model.nbGeneralizedTorque()

    def nb_segment(self):
        return self.model.nbSegment()

    def segment_index(self, name):
        return biorbd.segment_index(self.model, name)

    def nb_quat(self):
        return self.model.nbQuat()

    def nb_q(self):
        return self.model.nbQ()

    def nb_qdot(self):
        return self.model.nbQdot()

    def nb_qddot(self):
        return self.model.nbQddot()

    def nb_root(self):
        return self.model.nbRoot()

    def update_segment_characteristics(self, idx, characteristics):
        return self.model.updateSegmentCharacteristics(idx, characteristics)

    def segment(self, *args):
        return self.model.segment(*args)

    def segments(self, i):
        return self.model.segments()

    def dispatched_force(self, *args):
        return self.model.dispatchedForce(*args)

    def update_kinematics_custom(self, Q=None, Qdot=None, Qddot=None):
        return self.model.UpdateKinematicsCustom(Q, Qdot, Qddot)

    def all_global_jcs(self, *args):
        return self.model.allGlobalJCS(*args)

    def global_jcs(self, *args):
        return self.model.globalJCS(*args)

    def local_jcs(self, *args):
        return self.model.localJCS(*args)

    def project_point(self, *args):
        return self.model.projectPoint(*args)

    def project_point_jacobian(self, *args):
        return self.model.projectPointJacobian(*args)

    def mass(self):
        return self.model.mass()

    def com(self, Q, updateKin=True):
        return self.model.CoM(Q, updateKin)

    def CoMbySegmentInMatrix(self, Q, updateKin=True):
        return self.model.CoMbySegmentInMatrix(Q, updateKin)

    def CoMbySegment(self, *args):
        return self.model.CoMbySegment(*args)

    def comdot(self, Q, Qdot, updateKin=True):
        return self.model.CoMdot(Q, Qdot, updateKin)

    def comddot(self, Q, Qdot, Qddot, updateKin=True):
        return self.model.CoMddot(Q, Qdot, Qddot, updateKin)

    def comdot_by_segment(self, *args):
        return self.model.CoMdotBySegment(*args)

    def comddot_by_segment(self, *args):
        return self.model.CoMddotBySegment(*args)

    def com_jacobian(self, Q, updateKin=True):
        return self.model.CoMJacobian(Q, updateKin)

    def mesh_points(self, *args):
        return self.model.meshPoints(*args)

    def mesh_points_in_matrix(self, Q, updateKin=True):
        return self.model.meshPointsInMatrix(Q, updateKin)

    def mesh_faces(self, *args):
        return self.model.meshFaces(*args)

    def mesh(self, *args):
        return self.model.mesh(*args)

    def angular_momentum(self, Q, Qdot, updateKin=True):
        return self.model.angularMomentum(Q, Qdot, updateKin)

    def mass_matrix(self, Q, updateKin=True):
        return self.model.massMatrix(Q, updateKin)

    def mass_matrix_inverse(self, Q, updateKin=True):
        return self.model.massMatrixInverse(Q, updateKin)

    def calc_angular_momentum(self, *args):
        return self.model.CalcAngularMomentum(*args)

    def calc_segments_angular_momentum(self, *args):
        return self.model.CalcSegmentsAngularMomentum(*args)

    def body_angular_velocity(self, Q, Qdot, updateKin=True):
        return self.model.bodyAngularVelocity(Q, Qdot, updateKin)

    def calc_mat_rot_jacobian(self, Q, segmentIdx, rotation, G, updateKin):
        return self.model.CalcMatRotJacobian(Q, segmentIdx, rotation, G, updateKin)

    def jacobian_segment_rot_mat(self, Q, segmentIdx, updateKin):
        return self.model.JacobianSegmentRotMat(Q, segmentIdx, updateKin)

    def compute_qdot(self, Q, QDot, k_stab=1):
        return self.model.computeQdot(Q, QDot, k_stab)

    def segment_angular_velocity(self, Q, Qdot, idx, updateKin=True):
        return self.model.segmentAngularVelocity(Q, Qdot, idx, updateKin)

    def calc_kinetic_energy(self, Q, QDot, updateKin=True):
        return self.model.CalcKineticEnergy(Q, QDot, updateKin)

    def calc_potential_energy(self, Q, updateKin=True):
        return self.model.CalcPotentialEnergy(Q, updateKin)

    def name_dof(self):
        return self.model.nameDof()

    def contact_names(self):
        return self.model.contactNames()

    def nb_soft_contacts(self):
        return self.model.nbSoftContacts()

    def soft_contact_names(self):
        return self.model.softContactNames()

    def muscle_names(self):
        return self.model.muscleNames()

    def nb_muscles(self):
        return self.model.nbMuscles()

    def nb_muscle_total(self):
        return self.model.nbMuscleTotal()

    def torque(self, tau_activations, q, qdot):
        return self.model.torque(tau_activations, q, qdot)

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints):
        return self.model.ForwardDynamicsFreeFloatingBase(q, qdot, qddot_joints)

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        return self.model.ForwardDynamics(q, qdot, tau, fext, f_contacts)

    def forward_dynamics_constraints_direct(self, *args):
        return self.model.ForwardDynamicsConstraintsDirect(*args)

    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None):
        return self.model.InverseDynamics(q, qdot, qddot, f_ext, f_contacts)

    def non_linear_effect(self, Q, QDot, f_ext=None, f_contacts=None):
        return self.model.NonLinearEffect(Q, QDot, f_ext, f_contacts)

    def contact_forces_from_forward_dynamics_constraints_direct(self, Q, QDot, Tau, f_ext=None):
        return self.model.ContactForcesFromForwardDynamicsConstraintsDirect(Q, QDot, Tau, f_ext)

    def body_inertia(self, Q, updateKin=True):
        return self.model.bodyInertia(Q, updateKin)

    def compute_constraint_impulses_direct(self, Q, QDotPre):
        return self.model.ComputeConstraintImpulsesDirect(Q, QDotPre)

    def check_generalized_dimensions(self, Q=None, Qdot=None, Qddot=None, torque=None):
        return self.model.checkGeneralizedDimensions(Q, Qdot, Qddot, torque)

    def state_set(self):
        return self.model.stateSet()

    def activation_dot(self, muscle_states):
        return self.model.activationDot(muscle_states)

    def muscular_joint_torque(self, muscle_states, q, qdot):
        return self.model.muscularJointTorque(muscle_states, q, qdot)

    def get_constraints(self):
        return self.model.getConstraints()

    def markers(self, Q, updateKin=True):
        return self.model.markers(Q, updateKin)

    def nb_markers(self):
        return self.model.nbMarkers()

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def marker(self, i=None):
        if i is None:
            return self.model.marker
        else:
            return self.model.marker(i)

    def nb_rigid_contacts(self):
        return self.model.nbRigidContacts()

    def nb_contacts(self):
        return self.model.nbContacts()

    def path(self):
        return self.model.path()

    def markers_velocity(self, Q, Qdot, updateKin=True):
        return self.model.markerVelocity(Q, Qdot, updateKin)

    def rigid_contact_axis_idx(self, idx):
        return self.model.rigidContactAxisIdx(idx)

    def torque_max(self):
        return self.model.torqueMax()

