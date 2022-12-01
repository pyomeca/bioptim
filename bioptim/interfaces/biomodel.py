import biorbd_casadi as biorbd
from abc import ABC, abstractmethod
from typing import List


class BioModel(ABC):
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
    def nb_tau(self):
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
    def nb_dof(self):
        """Get the number of dof"""

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
    def kinetic_energy(self, Q, QDot, updateKin=True):
        """Get the kinetic energy of the model"""

    @abstractmethod
    def potential_energy(self, Q, updateKin=True):
        """Get the potential energy of the model"""

    @abstractmethod
    def name_dof(self) -> List[str]:
        """Get the name of the dof"""

    @abstractmethod
    def contact_names(self) -> List[str]:
        """Get the name of the contacts"""

    @abstractmethod
    def nb_soft_contacts(self):
        """Get the number of soft contacts"""

    @abstractmethod
    def soft_contact_names(self):
        """Get the soft contact names"""

    @abstractmethod
    def soft_contact(self, *args):
        """Get the soft contact"""

    @abstractmethod
    def muscle_names(self) -> List[str]:
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

    @abstractmethod
    def rigid_contact_acceleration(self, Q, Qdot, Qddot, updateKin=True):
        """Get the rigid contact acceleration"""

    @abstractmethod
    def rt(self, *args):
        """Get the rototranslation matrix"""

    @abstractmethod
    def marker_names(self) -> List[str]:
        """Get the marker names"""

    @abstractmethod
    def soft_contact_name(self, i) -> str:
        """Get the soft contact name"""


class BiorbdModel(BioModel):
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

    def nb_tau(self):
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

    def kinetic_energy(self, Q, QDot, updateKin=True):
        return self.model.KineticEnergy(Q, QDot, updateKin)

    def potential_energy(self, Q, updateKin=True):
        return self.model.PotentialEnergy(Q, updateKin)

    def name_dof(self):
        return [s.to_string() for s in self.model.nameDof()]

    def contact_names(self):
        return [s.to_string() for s in self.model.contactNames()]

    def nb_soft_contacts(self):
        return self.model.nbSoftContacts()

    def soft_contact_names(self):
        return [s.to_string() for s in self.model.softContactNames()]

    def soft_contact(self, *args):
        return self.model.softContact(*args)

    def muscle_names(self):
        return [s.to_string() for s in self.model.muscleNames()]

    def nb_muscles(self):
        return self.model.nbMuscles()

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

    def markers(self, *args):
        if len(args) == 0:
            return self.model.markers
        else:
            return self.model.markers(*args)

    def nb_markers(self):
        return self.model.nbMarkers()

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def marker(self, *args):
        if len(args) > 1:
            return self.model.marker(*args)
        else:
            return self.model.marker
        # hard to interface with c++ code
        # because sometimes it used as :
        # BiorbdInterface.mx_to_cx(
        #     f"markers_{first_marker}", nlp.model.marker, nlp.states["q"], first_marker_idx
        # )
        # it will change the way we call it by model.marker()
        # else:
        #     return self.model.marker(i)

    def nb_rigid_contacts(self):
        return self.model.nbRigidContacts()

    def nb_contacts(self):
        return self.model.nbContacts()

    def path(self):
        return self.model.path()

    def markers_velocity(self, Q, Qdot, updateKin=True):
        return self.model.markersVelocity(Q, Qdot, updateKin)

    def rigid_contact_axis_idx(self, idx):
        return self.model.rigidContactAxisIdx(idx)

    def torque_max(self, *args):
        return self.model.torqueMax(*args)

    def rigid_contact_acceleration(self, Q, Qdot, Qddot, idx=None, updateKin=True):
        return self.model.rigidContactAcceleration(Q, Qdot, Qddot, idx, updateKin)

    def rt(self, *args):
        return self.model.RT(*args)

    def nb_dof(self):
        return self.model.nbDof()

    def marker_names(self):
        return [s.to_string() for s in self.model.markerNames()]

    def soft_contact_name(self, i):
        return self.model.softContactName(i).to_string()

    def apply_rt(self, *args):
        return self.model.applyRT(*args)


class CustomModel(BioModel):
    """
    This is a custom model that inherits from bioptim.BioModel
    This class is made for the user to help him create his own model
    """

    # ---- absolutely needed to be implemented ---- #
    def nb_quat(self):
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

    def rigid_contact_axis_idx(self, *args):
        raise NotImplementedError("rigid_contact_axis_idx is not implemented")

    def rt(self, *args):
        raise NotImplementedError("rt is not implemented")

    def nb_dof(self):
        raise NotImplementedError("nb_dof is not implemented")

    def marker_names(self):
        raise NotImplementedError("marker_names is not implemented")

    def soft_contact_name(self, i):
        raise NotImplementedError("soft_contact_name is not implemented")
