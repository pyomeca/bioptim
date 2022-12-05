from casadi import MX
import biorbd_casadi as biorbd
from typing import Protocol
from pathlib import Path


class BiorbdModel:
    def __init__(self, bio_model: str | biorbd.Model):
        if isinstance(bio_model, str):
            self.model = biorbd.Model(bio_model)
        else:
            self.model = bio_model

    def deep_copy(self, *args):
        return BiorbdModel(self.model.DeepCopy(*args))

    @property
    def gravity(self) -> MX:
        return self.model.getGravity()

    def set_gravity(self, newGravity) -> None:
        return self.model.setGravity(newGravity)

    @property
    def nb_tau(self) -> int:
        return self.model.nbGeneralizedTorque()

    @property
    def nb_segments(self) -> int:
        return self.model.nbSegment()

    def segment_index(self, name) -> int:
        return biorbd.segment_index(self.model, name)

    @property
    def nb_quaternions(self) -> int:
        return self.model.nbQuat()

    @property
    def nb_q(self) -> int:
        return self.model.nbQ()

    @property
    def nb_qdot(self) -> int:
        return self.model.nbQdot()

    @property
    def nb_qddot(self) -> int:
        return self.model.nbQddot()

    @property
    def nb_root(self) -> int:
        return self.model.nbRoot()

    @property
    def segments(self) -> list[biorbd.Segment]:
        return self.model.segments()

    def global_homogeneous_matrices(self, *args):
        return self.model.globalJCS(*args)

    def child_homogeneous_matrices(self, *args):
        return self.model.localJCS(*args)

    @property
    def mass(self) -> MX:
        return self.model.mass()

    def center_of_mass(self, q, updateKin=True) -> MX:
        return self.model.CoM(q, updateKin)

    def center_of_mass_velocity(self, q, qdot, updateKin=True) -> MX:
        return self.model.CoMdot(q, qdot, updateKin)

    def center_of_mass_acceleration(self, q, qdot, qddot, updateKin=True) -> MX:
        return self.model.CoMddot(q, qdot, qddot, updateKin)

    def angular_momentum(self, Q, Qdot, updateKin=True) -> MX:
        return self.model.angularMomentum(Q, Qdot, updateKin)

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        return self.model.computeQdot(q, qdot, k_stab)

    def segment_angular_velocity(self, Q, Qdot, idx, updateKin=True) -> MX:
        return self.model.segmentAngularVelocity(Q, Qdot, idx, updateKin)

    @property
    def name_dof(self) -> tuple[str]:
        return tuple(s.to_string() for s in self.model.nameDof())

    @property
    def contact_names(self) -> tuple[str]:
        return tuple(s.to_string() for s in self.model.contactNames())

    @property
    def nb_soft_contacts(self) -> int:
        return self.model.nbSoftContacts()

    @property
    def soft_contact_names(self) -> tuple[str]:
        return tuple(s.to_string() for s in self.model.softContactNames())

    def soft_contact(self, *args):
        return self.model.softContact(*args)

    @property
    def muscle_names(self) -> tuple[str]:
        return tuple(s.to_string() for s in self.model.muscleNames())

    @property
    def nb_muscles(self) -> int:
        return self.model.nbMuscles()

    def torque(self, tau_activations, q, qdot) -> MX:
        return self.model.torque(tau_activations, q, qdot)

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        return self.model.ForwardDynamicsFreeFloatingBase(q, qdot, qddot_joints)

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None) -> MX:
        return self.model.ForwardDynamics(q, qdot, tau, fext, f_contacts)

    def constrained_forward_dynamics(self, *args) -> MX:
        return self.model.ForwardDynamicsConstraintsDirect(*args)

    def inverse_dynamics(self, q, qdot, qddot, f_ext=None, f_contacts=None) -> MX:
        return self.model.InverseDynamics(q, qdot, qddot, f_ext, f_contacts)

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, fext=None):
        return self.model.ContactForcesFromForwardDynamicsConstraintsDirect(q, qdot, tau, fext)

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        return self.model.ComputeConstraintImpulsesDirect(q, qdot_pre_impact)

    def state_set(self):
        # todo to remove
        return self.model.stateSet()

    def muscle_activation_dot(self, muscle_states) -> MX:
        return self.model.activationDot(muscle_states)

    def muscle_joint_torque(self, muscle_states, q, qdot) -> MX:
        return self.model.muscularJointTorque(muscle_states, q, qdot)

    def get_constraints(self):
        # todo: to be removed
        return self.model.getConstraints()

    def markers(self, *args) -> MX:
        if len(args) == 0:
            return self.model.markers
        else:
            return self.model.markers(*args)

    @property
    def nb_markers(self) -> int:
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

    @property
    def nb_rigid_contacts(self) -> int:
        return self.model.nbRigidContacts()

    @property
    def nb_contacts(self) -> int:
        return self.model.nbContacts()

    @property
    def path(self):
        return self.model.path()
        # return Path(self.model.path())

    def marker_velocities(self, q, qdot, update_kin=True) -> MX:
        return self.model.markersVelocity(q, qdot, update_kin)

    def rigid_contact_axis_idx(self, idx) -> int:
        # todo: to be removed
        return self.model.rigidContactAxisIdx(idx)

    def tau_max(self, *args) -> MX:
        return self.model.torqueMax(*args)

    def rigid_contact_acceleration(self, q, qdot, qddot, idx=None, updateKin=True) -> MX:
        return self.model.rigidContactAcceleration(q, qdot, qddot, idx, updateKin)

    def object_homogeneous_matrix(self, *args) -> MX:
        return self.model.RT(*args)

    @property
    def nb_dof(self) -> int:
        return self.model.nbDof()

    @property
    def marker_names(self) -> tuple[str]:
        return tuple(s.to_string() for s in self.model.markerNames())

    def soft_contact_forces(self, q: MX, qdot: MX) -> MX:
        soft_contact_forces = MX.zeros(self.nb_soft_contacts * 6, 1)
        for i_sc in range(self.nb_soft_contacts):
            soft_contact = self.soft_contact(i_sc)

            soft_contact_forces[i_sc * 6 : (i_sc + 1) * 6, :] = (
                biorbd.SoftContactSphere(soft_contact).computeForceAtOrigin(self.model, q, qdot).to_mx()
            )
        return soft_contact_forces
