import biorbd_casadi as biorbd
import numpy as np
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, norm_fro
from typing import Callable

from ..utils import _var_mapping, bounds_from_ranges
from ...limits.path_conditions import Bounds
from ...misc.mapping import BiMapping, BiMappingList
from ...misc.utils import check_version

check_version(biorbd, "1.11.1", "1.12.0")


class BiorbdModel:
    """
    This class wraps the biorbd model and allows the user to call the biorbd functions from the biomodel protocol
    """

    def __init__(
        self,
        bio_model: str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        segments_to_apply_external_forces: list[str] = [],
    ):
        if not isinstance(bio_model, str) and not isinstance(bio_model, biorbd.Model):
            raise ValueError("The model should be of type 'str' or 'biorbd.Model'")

        self.model = biorbd.Model(bio_model) if isinstance(bio_model, str) else bio_model
        self._friction_coefficients = friction_coefficients
        self._segments_to_apply_external_forces = segments_to_apply_external_forces

    @property
    def name(self) -> str:
        # parse the path and split to get the .bioMod name
        return self.model.path().absolutePath().to_string().split("/")[-1]

    @property
    def path(self) -> str:
        return self.model.path().relativePath().to_string()

    def copy(self):
        return BiorbdModel(self.path)

    def serialize(self) -> tuple[Callable, dict]:
        return BiorbdModel, dict(bio_model=self.path)

    @property
    def friction_coefficients(self) -> MX | np.ndarray:
        return self._friction_coefficients

    def set_friction_coefficients(self, new_friction_coefficients) -> None:
        if np.any(new_friction_coefficients < 0):
            raise ValueError("Friction coefficients must be positive")
        return self._friction_coefficients

    @property
    def gravity(self) -> MX:
        return self.model.getGravity().to_mx()

    def set_gravity(self, new_gravity) -> None:
        self.model.setGravity(new_gravity)
        return

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
    def nb_dof(self) -> int:
        return self.model.nbDof()

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
    def segments(self) -> tuple[biorbd.Segment]:
        return self.model.segments()

    def biorbd_homogeneous_matrices_in_global(self, q, segment_idx, inverse=False) -> tuple:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        """
        rt_matrix = self.model.globalJCS(GeneralizedCoordinates(q), segment_idx)
        return rt_matrix.transpose() if inverse else rt_matrix

    def homogeneous_matrices_in_global(self, q, segment_idx, inverse=False) -> MX:
        """
        Returns the roto-translation matrix of the segment in the global reference frame.
        """
        return self.biorbd_homogeneous_matrices_in_global(q, segment_idx, inverse).to_mx()

    def homogeneous_matrices_in_child(self, segment_id) -> MX:
        return self.model.localJCS(segment_id).to_mx()

    @property
    def mass(self) -> MX:
        return self.model.mass().to_mx()

    def check_q_size(self, q):
        if q.shape[0] != self.nb_q:
            raise ValueError(f"Length of q size should be: {self.nb_q}, but got: {q.shape[0]}")

    def check_qdot_size(self, qdot):
        if qdot.shape[0] != self.nb_qdot:
            raise ValueError(f"Length of qdot size should be: {self.nb_qdot}, but got: {qdot.shape[0]}")

    def check_qddot_size(self, qddot):
        if qddot.shape[0] != self.nb_qddot:
            raise ValueError(f"Length of qddot size should be: {self.nb_qddot}, but got: {qddot.shape[0]}")

    def check_qddot_joints_size(self, qddot_joints):
        nb_qddot_joints = self.nb_q - self.nb_root
        if qddot_joints.shape[0] != nb_qddot_joints:
            raise ValueError(
                f"Length of qddot_joints size should be: {nb_qddot_joints}, but got: {qddot_joints.shape[0]}"
            )

    def check_tau_size(self, tau):
        if tau.shape[0] != self.nb_tau:
            raise ValueError(f"Length of tau size should be: {self.nb_tau}, but got: {tau.shape[0]}")

    def check_muscle_size(self, muscle):
        if isinstance(muscle, list):
            muscle_size = len(muscle)
        elif hasattr(muscle, "shape"):
            muscle_size = muscle.shape[0]
        else:
            raise TypeError("Unsupported type for muscle.")

        if muscle_size != self.nb_muscles:
            raise ValueError(f"Length of muscle size should be: {self.nb_muscles}, but got: {muscle_size}")

    def center_of_mass(self, q) -> MX:
        self.check_q_size(q)
        q_biorbd = GeneralizedCoordinates(q)
        return self.model.CoM(q_biorbd, True).to_mx()

    def center_of_mass_velocity(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.CoMdot(q_biorbd, qdot_biorbd, True).to_mx()

    def center_of_mass_acceleration(self, q, qdot, qddot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_qddot_size(qddot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)
        return self.model.CoMddot(q_biorbd, qdot_biorbd, qddot_biorbd, True).to_mx()

    def body_rotation_rate(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.bodyAngularVelocity(q_biorbd, qdot_biorbd, True).to_mx()

    def mass_matrix(self, q) -> MX:
        self.check_q_size(q)
        q_biorbd = GeneralizedCoordinates(q)
        return self.model.massMatrix(q_biorbd).to_mx()

    def non_linear_effects(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.NonLinearEffect(q_biorbd, qdot_biorbd).to_mx()

    def angular_momentum(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.angularMomentum(q_biorbd, qdot_biorbd, True).to_mx()

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        return self.model.computeQdot(
            GeneralizedCoordinates(q),
            GeneralizedCoordinates(qdot),  # mistake in biorbd
            k_stab,
        ).to_mx()

    def segment_angular_velocity(self, q, qdot, idx) -> MX:
        """
        Returns the angular velocity of the segment in the global reference frame.
        """
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.segmentAngularVelocity(q_biorbd, qdot_biorbd, idx, True).to_mx()

    def segment_orientation(self, q, idx) -> MX:
        """
        Returns the angular position of the segment in the global reference frame.
        """
        q_biorbd = GeneralizedCoordinates(q)
        rotation_matrix = self.homogeneous_matrices_in_global(q_biorbd, idx)[:3, :3]
        segment_orientation = biorbd.Rotation.toEulerAngles(
            biorbd.Rotation(
                rotation_matrix[0, 0],
                rotation_matrix[0, 1],
                rotation_matrix[0, 2],
                rotation_matrix[1, 0],
                rotation_matrix[1, 1],
                rotation_matrix[1, 2],
                rotation_matrix[2, 0],
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
            ),
            "xyz",
        ).to_mx()
        return segment_orientation

    @property
    def name_dof(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.nameDof())

    @property
    def contact_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.contactNames())

    @property
    def nb_soft_contacts(self) -> int:
        return self.model.nbSoftContacts()

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        return self.model.softContactNames()

    def soft_contact(self, soft_contact_index, *args):
        return self.model.softContact(soft_contact_index, *args)

    @property
    def muscle_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.muscleNames())

    @property
    def nb_muscles(self) -> int:
        return self.model.nbMuscles()

    def torque(self, tau_activations, q, qdot) -> MX:
        self.check_tau_size(tau_activations)
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_activation = self.model.torque(tau_activations, q_biorbd, qdot_biorbd)
        return tau_activation.to_mx()

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_qddot_joints_size(qddot_joints)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_joints_biorbd = GeneralizedAcceleration(qddot_joints)
        return self.model.ForwardDynamicsFreeFloatingBase(q_biorbd, qdot_biorbd, qddot_joints_biorbd).to_mx()

    @staticmethod
    def reorder_qddot_root_joints(qddot_root, qddot_joints) -> MX:
        return vertcat(qddot_root, qddot_joints)

    def _dispatch_forces(self, external_forces: MX, translational_forces: MX):

        if external_forces is not None and translational_forces is not None:
            raise NotImplementedError(
                "You cannot provide both external_forces and translational_forces at the same time."
            )
        elif external_forces is not None:
            if not isinstance(external_forces, MX):
                raise ValueError("external_forces should be a numpy array of shape 9 x nb_forces.")
            if external_forces.shape[0] != 9:
                raise ValueError(
                    f"external_forces has {external_forces.shape[0]} rows, it should have 9 rows (Mx, My, Mz, Fx, Fy, Fz, Px, Py, Pz). You should provide the moments, forces and points of application."
                )
            if len(self._segments_to_apply_external_forces) != external_forces.shape[1]:
                raise ValueError(
                    f"external_forces has {external_forces.shape[1]} columns and {len(self._segments_to_apply_external_forces)} segments to apply forces on, they should have the same length."
                )
        elif translational_forces is not None:
            if not isinstance(translational_forces, MX):
                raise ValueError("translational_forces should be a numpy array of shape 6 x nb_forces.")
            if translational_forces.shape[0] != 6:
                raise ValueError(
                    f"translational_forces has {translational_forces.shape[0]} rows, it should have 6 rows (Fx, Fy, Fz, Px, Py, Pz). You should provide the forces and points of application."
                )
            if len(self._segments_to_apply_external_forces) != translational_forces.shape[1]:
                raise ValueError(
                    f"translational_forces has {translational_forces.shape[1]} columns and {len(self._segments_to_apply_external_forces)} segments to apply forces on, they should have the same length."
                )

        external_forces_set = self.model.externalForceSet()

        if external_forces is not None:
            for i_element in range(external_forces.shape[1]):
                name = self._segments_to_apply_external_forces[i_element]
                values = external_forces[:6, i_element]
                point_of_application = external_forces[6:9, i_element]
                external_forces_set.add(name, values, point_of_application)

        if translational_forces is not None:
            for i_elements in range(translational_forces.shape[1]):
                name = self._segments_to_apply_external_forces[i_elements]
                values = translational_forces[:3, i_elements]
                point_of_application = translational_forces[3:6, i_elements]
                external_forces_set.addTranslationalForce(values, name, point_of_application)

        return external_forces_set

    def forward_dynamics(self, q, qdot, tau, external_forces=None, translational_forces=None) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_tau_size(tau)
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ForwardDynamics(q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set).to_mx()

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None, translational_forces=None) -> MX:
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_tau_size(tau)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ForwardDynamicsConstraintsDirect(
            q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set
        ).to_mx()

    def inverse_dynamics(self, q, qdot, qddot, external_forces=None, translational_forces=None) -> MX:
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_qddot_size(qddot)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)
        return self.model.InverseDynamics(q_biorbd, qdot_biorbd, qddot_biorbd, external_forces_set).to_mx()

    def contact_forces_from_constrained_forward_dynamics(
        self, q, qdot, tau, external_forces=None, translational_forces=None
    ) -> MX:
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_tau_size(tau)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ContactForcesFromForwardDynamicsConstraintsDirect(
            q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set
        ).to_mx()

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot_pre_impact)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_pre_impact_biorbd = GeneralizedVelocity(qdot_pre_impact)
        return self.model.ComputeConstraintImpulsesDirect(q_biorbd, qdot_pre_impact_biorbd).to_mx()

    def muscle_activation_dot(self, muscle_excitations) -> MX:
        self.check_muscle_size(muscle_excitations)
        muscle_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscle_states[k].setExcitation(muscle_excitations[k])
        return self.model.activationDot(muscle_states).to_mx()

    def muscle_length_jacobian(self, q) -> MX:
        self.check_q_size(q)
        q_biorbd = GeneralizedCoordinates(q)
        return self.model.musclesLengthJacobian(q_biorbd).to_mx()

    def muscle_velocity(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        J = self.muscle_length_jacobian(q)
        return J @ qdot

    def muscle_joint_torque(self, activations, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_muscle_size(activations)

        muscles_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscles_states[k].setActivation(activations[k])
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx()

    def markers(self, q) -> list[MX]:
        self.check_q_size(q)
        return [m.to_mx() for m in self.model.markers(GeneralizedCoordinates(q))]

    @property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def marker(self, q, index, reference_segment_index=None) -> MX:
        self.check_q_size(q)
        marker = self.model.marker(GeneralizedCoordinates(q), index)
        if reference_segment_index is not None:
            global_homogeneous_matrix = self.model.globalJCS(GeneralizedCoordinates(q), reference_segment_index)
            marker.applyRT(global_homogeneous_matrix.transpose())
        return marker.to_mx()

    @property
    def nb_rigid_contacts(self) -> int:
        """
        Returns the number of rigid contacts.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_rigid_contacts = 2
        """
        return self.model.nbRigidContacts()

    @property
    def nb_contacts(self) -> int:
        """
        Returns the number of contact index.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_contacts = 3
        """
        return self.model.nbContacts()

    def rigid_contact_index(self, contact_index) -> tuple:
        """
        Returns the axis index of this specific rigid contact.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            rigid_contact_index(0) = (1, 2)
        """
        return self.model.rigidContacts()[contact_index].availableAxesIndices()

    def marker_velocities(self, q, qdot, reference_index=None) -> list[MX]:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        if reference_index is None:
            return [
                m.to_mx()
                for m in self.model.markersVelocity(
                    GeneralizedCoordinates(q),
                    GeneralizedVelocity(qdot),
                    True,
                )
            ]

        else:
            out = []
            homogeneous_matrix_transposed = self.biorbd_homogeneous_matrices_in_global(
                GeneralizedCoordinates(q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersVelocity(GeneralizedCoordinates(q), GeneralizedVelocity(qdot)):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    out.append(m.to_mx())

            return out

    def marker_accelerations(self, q, qdot, qddot, reference_index=None) -> list[MX]:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_qddot_size(qddot)
        if reference_index is None:
            return [
                m.to_mx()
                for m in self.model.markerAcceleration(
                    GeneralizedCoordinates(q),
                    GeneralizedVelocity(qdot),
                    GeneralizedAcceleration(qddot),
                    True,
                )
            ]

        else:
            out = []
            homogeneous_matrix_transposed = self.biorbd_homogeneous_matrices_in_global(
                GeneralizedCoordinates(q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersAcceleration(
                GeneralizedCoordinates(q),
                GeneralizedVelocity(qdot),
                GeneralizedAcceleration(qddot),
            ):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    out.append(m.to_mx())

            return out

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        self.model.closeActuator()
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        torque_max, torque_min = self.model.torqueMax(q_biorbd, qdot_biorbd)
        return torque_max.to_mx(), torque_min.to_mx()

    def rigid_contact_acceleration(self, q, qdot, qddot, contact_index, contact_axis) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_qddot_size(qddot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)
        return self.model.rigidContactAcceleration(q_biorbd, qdot_biorbd, qddot_biorbd, contact_index, True).to_mx()[
            contact_axis
        ]

    def markers_jacobian(self, q) -> list[MX]:
        self.check_q_size(q)
        return [m.to_mx() for m in self.model.markersJacobian(GeneralizedCoordinates(q))]

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.to_string() for s in self.model.markerNames()])

    def soft_contact_forces(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)

        soft_contact_forces = MX.zeros(self.nb_soft_contacts * 6, 1)
        for i_sc in range(self.nb_soft_contacts):
            soft_contact = self.soft_contact(i_sc)

            soft_contact_forces[i_sc * 6 : (i_sc + 1) * 6, :] = (
                biorbd.SoftContactSphere(soft_contact).computeForceAtOrigin(self.model, q_biorbd, qdot_biorbd).to_mx()
            )

        return soft_contact_forces

    def reshape_fext_to_fcontact(self, fext: MX) -> list:
        if len(self._segments_to_apply_external_forces) == 0:
            parent_name = []
            for i in range(self.nb_rigid_contacts):
                contact = self.model.rigidContact(i)
                parent_name += [
                    self.model.segment(self.model.getBodyRbdlIdToBiorbdId(contact.parentId())).name().to_string()
                ]
            self._segments_to_apply_external_forces = parent_name

        count = 0
        f_contact_vec = MX()
        for i in range(self.nb_rigid_contacts):
            contact = self.model.rigidContact(i)
            tp = MX.zeros(6)
            used_axes = [i for i, val in enumerate(contact.axes()) if val]
            n_contacts = len(used_axes)
            tp[used_axes] = fext[count : count + n_contacts]
            tp[3:] = contact.to_mx()
            f_contact_vec = horzcat(f_contact_vec, tp)
            count += n_contacts
        return f_contact_vec

    def normalize_state_quaternions(self, x: MX | SX) -> MX | SX:
        quat_idx = self.get_quaternion_idx()

        # Normalize quaternion, if needed
        for j in range(self.nb_quaternions):
            quaternion = vertcat(
                x[quat_idx[j][3]],
                x[quat_idx[j][0]],
                x[quat_idx[j][1]],
                x[quat_idx[j][2]],
            )
            quaternion /= norm_fro(quaternion)
            x[quat_idx[j][0] : quat_idx[j][2] + 1] = quaternion[1:4]
            x[quat_idx[j][3]] = quaternion[0]

        return x

    def get_quaternion_idx(self) -> list[list[int]]:
        n_dof = 0
        quat_idx = []
        quat_number = 0
        for j in range(self.nb_segments):
            if self.segments[j].isRotationAQuaternion():
                quat_idx.append([n_dof, n_dof + 1, n_dof + 2, self.nb_dof + quat_number])
                quat_number += 1
            n_dof += self.segments[j].nbDof()
        return quat_idx

    def contact_forces(self, q, qdot, tau, external_forces: MX = None) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_tau_size(tau)
        if external_forces is not None:
            all_forces = MX()
            for i in range(external_forces.shape[1]):
                force = self.contact_forces_from_constrained_forward_dynamics(
                    q, qdot, tau, external_forces=external_forces[:, i]
                )
                all_forces = horzcat(all_forces, force)
            return all_forces
        else:
            return self.contact_forces_from_constrained_forward_dynamics(q, qdot, tau, external_forces=None)

    def passive_joint_torque(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.passiveJointTorque(q_biorbd, qdot_biorbd).to_mx()

    def ligament_joint_torque(self, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.ligamentsJointTorque(q_biorbd, qdot_biorbd).to_mx()

    def ranges_from_model(self, variable: str):
        ranges = []
        for segment in self.segments:
            if "_joints" in variable:
                if segment.parent().to_string().lower() != "root":
                    variable = variable.replace("_joints", "")
                    ranges += self._add_range(variable, segment)
            elif "_roots" in variable:
                if segment.parent().to_string().lower() == "root":
                    variable = variable.replace("_roots", "")
                    ranges += self._add_range(variable, segment)
            else:
                ranges += self._add_range(variable, segment)

        return ranges

    @staticmethod
    def _add_range(variable: str, segment: biorbd.Segment) -> list[biorbd.Range]:
        """
        Get the range of a variable for a given segment

        Parameters
        ----------
        variable: str
            The variable to get the range for such as:
             "q", "qdot", "qddot", "q_joint", "qdot_joint", "qddot_joint", "q_root", "qdot_root", "qddot_root"
        segment: biorbd.Segment
            The segment to get the range from

        Returns
        -------
        list[biorbd.Range]
            range min and max for the given variable for a given segment
        """
        ranges_map = {
            "q": [q_range for q_range in segment.QRanges()],
            "qdot": [qdot_range for qdot_range in segment.QdotRanges()],
            "qddot": [qddot_range for qddot_range in segment.QddotRanges()],
        }

        segment_variable_range = ranges_map.get(variable, None)
        if segment_variable_range is None:
            RuntimeError("Wrong variable name")

        return segment_variable_range

    def _var_mapping(
        self,
        key: str,
        range_for_mapping: int | list | tuple | range,
        mapping: BiMapping = None,
    ) -> dict:
        return _var_mapping(key, range_for_mapping, mapping)

    def bounds_from_ranges(self, variables: str | list[str], mapping: BiMapping | BiMappingList = None) -> Bounds:
        return bounds_from_ranges(self, variables, mapping)

    def lagrangian(self, q: MX | SX, qdot: MX | SX) -> MX | SX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.Lagrangian(q_biorbd, qdot_biorbd).to_mx()

    def partitioned_forward_dynamics(self, q_u, qdot_u, tau, external_forces=None, f_contacts=None, q_v_init=None):
        raise NotImplementedError("partitioned_forward_dynamics is not implemented for BiorbdModel")

    @staticmethod
    def animate(
        ocp,
        solution,
        show_now: bool = True,
        show_tracked_markers: bool = False,
        viewer: str = "pyorerun",
        n_frames: int = 0,
        **kwargs,
    ):
        if viewer == "bioviz":
            from .viewer_bioviz import animate_with_bioviz_for_loop

            return animate_with_bioviz_for_loop(ocp, solution, show_now, show_tracked_markers, n_frames, **kwargs)
        if viewer == "pyorerun":
            from .viewer_pyorerun import animate_with_pyorerun

            return animate_with_pyorerun(ocp, solution, show_now, show_tracked_markers, **kwargs)
