from typing import Any, Callable

import biorbd_casadi as biorbd
from casadi import MX, horzcat, vertcat, SX, norm_fro

from ..misc.mapping import BiMapping, BiMappingList
from ..misc.utils import check_version
from ..limits.path_conditions import Bounds

check_version(biorbd, "1.9.9", "1.10.0")


class BiorbdModel:
    def __init__(self, bio_model: str | biorbd.Model):
        if isinstance(bio_model, str):
            self.model = biorbd.Model(bio_model)
        elif isinstance(bio_model, biorbd.Model):
            self.model = bio_model
        else:
            raise RuntimeError("Type must be a 'str' or a 'biorbd.Model'")

    def deep_copy(self, *args):
        return BiorbdModel(self.model.DeepCopy(*args))

    @property
    def path(self) -> str:
        return self.model.path().relativePath().to_string()

    def copy(self):
        return BiorbdModel(self.path)

    def serialize(self) -> tuple[Callable, dict]:
        return BiorbdModel, dict(bio_model=self.path)

    @property
    def gravity(self) -> MX:
        return self.model.getGravity().to_mx()

    def set_gravity(self, new_gravity) -> None:
        return self.model.setGravity(new_gravity)

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

    def homogeneous_matrices_in_global(self, q, reference_index, inverse=False):
        val = self.model.globalJCS(q, reference_index)
        if inverse:
            return val.transpose()
        else:
            return val

    def homogeneous_matrices_in_child(self, *args):
        return self.model.localJCS(*args)

    @property
    def mass(self) -> MX:
        return self.model.mass().to_mx()

    def center_of_mass(self, q) -> MX:
        return self.model.CoM(q, True).to_mx()

    def center_of_mass_velocity(self, q, qdot) -> MX:
        return self.model.CoMdot(q, qdot, True).to_mx()

    def center_of_mass_acceleration(self, q, qdot, qddot) -> MX:
        return self.model.CoMddot(q, qdot, qddot, True).to_mx()

    def angular_momentum(self, q, qdot) -> MX:
        return self.model.angularMomentum(q, qdot, True)

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        return self.model.computeQdot(q, qdot, k_stab).to_mx()

    def segment_angular_velocity(self, q, qdot, idx) -> MX:
        return self.model.segmentAngularVelocity(q, qdot, idx, True)

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
        return tuple(s.to_string() for s in self.model.softContactNames())

    def soft_contact(self, *args):
        return self.model.softContact(*args)

    @property
    def muscle_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.muscleNames())

    @property
    def nb_muscles(self) -> int:
        return self.model.nbMuscles()

    def torque(self, tau_activations, q, qdot) -> MX:
        return self.model.torque(tau_activations, q, qdot).to_mx()

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        return self.model.ForwardDynamicsFreeFloatingBase(q, qdot, qddot_joints).to_mx()

    def forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        # external_forces = self.convert_array_to_external_forces(external_forces)
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        return self.model.ForwardDynamics(q, qdot, tau, external_forces, f_contacts).to_mx()

    def constrained_forward_dynamics(self, q, qdot, qddot, external_forces=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        return self.model.ForwardDynamicsConstraintsDirect(q, qdot, qddot, external_forces).to_mx()

    def inverse_dynamics(self, q, qdot, qddot, external_forces=None, f_contacts: biorbd.VecBiorbdVector = None) -> MX:
        # external_forces = self.convert_array_to_external_forces(external_forces)
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        return self.model.InverseDynamics(q, qdot, qddot, external_forces, f_contacts).to_mx()

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        # external_forces = self.convert_array_to_external_forces(external_forces)
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        return self.model.ContactForcesFromForwardDynamicsConstraintsDirect(q, qdot, tau, external_forces).to_mx()

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        return self.model.ComputeConstraintImpulsesDirect(q, qdot_pre_impact).to_mx()

    def muscle_activation_dot(self, muscle_excitations) -> MX:
        muscle_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscle_states[k].setExcitation(muscle_excitations[k])
        return self.model.activationDot(muscle_states).to_mx()

    def muscle_joint_torque(self, activations, q, qdot) -> MX:
        muscles_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscles_states[k].setActivation(activations[k])

        return self.model.muscularJointTorque(muscles_states, q, qdot).to_mx()

    def markers(self, q) -> Any | list[MX]:
        return [m.to_mx() for m in self.model.markers(q)]

    @property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def marker(self, q, index, reference_segment_index=None) -> MX:
        marker = self.model.marker(q, index)

        if reference_segment_index is not None:
            global_homogeneous_matrix = self.model.globalJCS(q, reference_segment_index)
            marker.applyRT(global_homogeneous_matrix.transpose())

        return marker.to_mx()

    @property
    def nb_rigid_contacts(self) -> int:
        return self.model.nbRigidContacts()

    @property
    def nb_contacts(self) -> int:
        return self.model.nbContacts()

    def marker_velocities(self, q, qdot, reference_index=None) -> MX:
        if reference_index is None:
            return horzcat(*[m.to_mx() for m in self.model.markersVelocity(q, qdot, True)])
        else:
            homogeneous_matrix_transposed = (
                biorbd.RotoTrans(),
                self.homogeneous_matrices_in_global(q, reference_index, inverse=True),
            )
        return horzcat(
            *[
                m.to_mx()
                for m in self.model.markersVelocity(q, qdot, True)
                if m.applyRT(homogeneous_matrix_transposed) is None
            ]
        )

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        torque_max, torque_min = self.model.torqueMax(q, qdot)
        return torque_max.to_mx(), torque_min.to_mx()

    def rigid_contact_acceleration(self, q, qdot, qddot, index) -> MX:
        # TODO: There is a bug here since only index 0 is call.

        if "_X" in self.contact_names[index]:
            index_direction = 0
        elif "_Y" in self.contact_names[index]:
            index_direction = 1
        elif "_Z" in self.contact_names[index]:
            index_direction = 2
        else:
            raise ValueError("Wrong index")
        return self.model.rigidContactAcceleration(q, qdot, qddot, 0, True).to_mx()[index_direction]

    @property
    def nb_dof(self) -> int:
        return self.model.nbDof()

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.markerNames())

    def soft_contact_forces(self, q, qdot) -> MX:
        soft_contact_forces = MX.zeros(self.nb_soft_contacts * 6, 1)
        for i_sc in range(self.nb_soft_contacts):
            soft_contact = self.soft_contact(i_sc)

            soft_contact_forces[i_sc * 6 : (i_sc + 1) * 6, :] = (
                biorbd.SoftContactSphere(soft_contact).computeForceAtOrigin(self.model, q, qdot).to_mx()
            )
        return soft_contact_forces

    def reshape_fext_to_fcontact(self, fext: MX) -> biorbd.VecBiorbdVector:
        count = 0
        f_contact_vec = biorbd.VecBiorbdVector()
        for ii in range(self.nb_rigid_contacts):
            n_f_contact = len(self.model.rigidContactAxisIdx(ii))
            idx = [i + count for i in range(n_f_contact)]
            f_contact_vec.append(fext[idx])
            count = count + n_f_contact

        return f_contact_vec

    def normalize_state_quaternions(self, x: MX | SX) -> MX | SX:
        quat_idx = self.get_quaternion_idx()

        # Normalize quaternion, if needed
        for j in range(self.nb_quaternions):
            quaternion = vertcat(x[quat_idx[j][3]], x[quat_idx[j][0]], x[quat_idx[j][1]], x[quat_idx[j][2]])
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

    def contact_forces(self, q, qdot, tau, external_forces: list = None) -> MX:
        if external_forces is not None and len(external_forces) != 0:
            all_forces = MX()
            for i, f_ext in enumerate(external_forces):
                force = self.contact_forces_from_constrained_forward_dynamics(q, qdot, tau, external_forces=f_ext)
                all_forces = horzcat(all_forces, force)
            return all_forces
        else:
            return self.contact_forces_from_constrained_forward_dynamics(q, qdot, tau, external_forces=None)

    def passive_joint_torque(self, q, qdot) -> MX:
        return self.model.passiveJointTorque(q, qdot).to_mx()

    def ligament_joint_torque(self, q, qdot) -> MX:
        return self.model.ligamentsJointTorque(q, qdot).to_mx()

    def _q_mapping(self, mapping: BiMapping = None) -> BiMapping:
        if mapping is None:
            mapping = {}
        if self.nb_quaternions > 0:
            if "q" in mapping and "qdot" not in mapping:
                raise RuntimeError(
                    "It is not possible to provide a q_mapping but not a qdot_mapping if the model have quaternion"
                )
            elif "q" not in mapping and "qdot" in mapping:
                raise RuntimeError(
                    "It is not possible to provide a qdot_mapping but not a q_mapping if the model have quaternion"
                )
        if "q" not in mapping:
            mapping["q"] = BiMapping(range(self.nb_q), range(self.nb_q))
        return mapping

    def _qdot_mapping(self, mapping: BiMapping = None) -> BiMapping:
        if mapping is None:
            mapping = {}
        if "qdot" not in mapping:
            if self.nb_quaternions > 0:
                mapping["qdot"] = BiMapping(range(self.nb_qdot), range(self.nb_qdot))
            else:
                if "q" not in mapping:
                    mapping["q"] = BiMapping(range(self.nb_q), range(self.nb_q))
                mapping["qdot"] = mapping["q"]
        return mapping

    def _qddot_mapping(self, mapping: BiMapping = None) -> BiMapping:
        if mapping is None:
            mapping = {}
        if "qddot" not in mapping:
            if self.nb_quaternions > 0:
                mapping["qddot"] = BiMapping(range(self.nb_qddot), range(self.nb_qddot))
            elif "qdot" not in mapping:
                if self.nb_quaternions > 0:
                    mapping["qdot"] = BiMapping(range(self.nb_qdot), range(self.nb_qdot))
                    mapping["qddot"] = mapping["qdot"]
                else:
                    if "q" not in mapping:
                        mapping["q"] = BiMapping(range(self.nb_q), range(self.nb_q))
                    mapping["qdot"] = mapping["q"]
                    mapping["qddot"] = mapping["qdot"]
            else:
                mapping["qddot"] = mapping["qdot"]
        return mapping

    def bounds_from_ranges(self, variables: str | list[str, ...], mapping: BiMapping | BiMappingList = None) -> Bounds:
        out = Bounds()
        q_ranges = []
        qdot_ranges = []
        qddot_ranges = []

        for i in range(self.nb_segments):
            segment = self.segments[i]
            for var in variables:
                if var == "q":
                    q_ranges += [q_range for q_range in segment.QRanges()]
                if var == "qdot":
                    qdot_ranges += [qdot_range for qdot_range in segment.QDotRanges()]
                if var == "qddot":
                    qddot_ranges += [qddot_range for qddot_range in segment.QDDotRanges()]

        for var in variables:
            if var == "q":
                q_mapping = self._q_mapping(mapping)
                mapping = q_mapping
                x_min = [q_ranges[i].min() for i in q_mapping["q"].to_first.map_idx]
                x_max = [q_ranges[i].max() for i in q_mapping["q"].to_first.map_idx]
                out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

            elif var == "qdot":
                qdot_mapping = self._qdot_mapping(mapping)
                mapping = qdot_mapping
                x_min = [qdot_ranges[i].min() for i in qdot_mapping["qdot"].to_first.map_idx]
                x_max = [qdot_ranges[i].max() for i in qdot_mapping["qdot"].to_first.map_idx]
                out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

            elif var == "qddot":
                qddot_mapping = self._qddot_mapping(mapping)
                mapping = qddot_mapping
                x_min = [qddot_ranges[i].min() for i in qddot_mapping["qddot"].to_first.map_idx]
                x_max = [qddot_ranges[i].max() for i in qddot_mapping["qddot"].to_first.map_idx]
                out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

        if out.shape[0] == 0:
            raise ValueError(f"Unrecognized variable ({variables}), only 'q', 'qdot' and 'qddot' are allowed")

        return out
