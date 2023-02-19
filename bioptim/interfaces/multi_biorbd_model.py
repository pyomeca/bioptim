from typing import Any, Callable

import biorbd_casadi as biorbd
from casadi import MX, vertcat, horzcat, SX, norm_fro

from .biorbd_model import BiorbdModel
from ..misc.mapping import BiMapping, BiMappingList
from ..misc.utils import check_version
from ..limits.path_conditions import Bounds

check_version(biorbd, "1.9.9", "1.10.0")

class MultiBiorbdModel:
    def __init__(self, bio_models: tuple[str | biorbd.Model, ...]):
        self.models = []
        for bio_model in bio_models:
            if isinstance(bio_model, str):
                self.models.append(biorbd.Model(bio_model))
            elif isinstance(bio_model, biorbd.Model):
                self.models.append(bio_model)
            else:
                raise RuntimeError("Type must be a tuple")

    def __getitem__(self, index):
        return self.models[index]

    def deep_copy(self, *args):
        return MultiBiorbdModel(tuple(BiorbdModel(model.DeepCopy(*args)) for model in self.models)) ## ???????

    @property
    def path(self) -> list[str]:
        return [model.path().relativePath().to_string() for model in self.models]

    def copy(self):
        return MultiBiorbdModel(tuple(path for path in self.path))

    def serialize(self) -> tuple[Callable, dict]:
        return MultiBiorbdModel, dict(bio_model=tuple(path for path in self.path))

    @property
    def gravity(self) -> MX:
        return vertcat(*(model.getGravity().to_mx() for model in self.models))

    def set_gravity(self, new_gravity, model_index=0) -> None:
        return self.models[model_index].setGravity(new_gravity)

    @property
    def nb_tau(self) -> int:
        return sum(model.nbGeneralizedTorque() for model in self.models)

    @property
    def nb_segments(self) -> int:
        return sum(model.nbSegment() for model in self.models)

    def segment_index(self, name, model_index=0) -> int:
        return biorbd.segment_index(self.models[model_index], name)

    @property
    def nb_quaternions(self) -> int:
        return sum(model.nbQuat() for model in self.models)

    @property
    def nb_q(self) -> int:
        return sum(model.nbQ() for model in self.models)

    @property
    def nb_qdot(self) -> int:
        return sum(model.nbQdot() for model in self.models)

    @property
    def nb_qddot(self) -> int:
        return sum(model.nbQddot() for model in self.models)

    @property
    def nb_root(self) -> int:
        return sum(model.nbRoot() for model in self.models)

    @property
    def segments(self) -> list[biorbd.Segment]:
        out = ()
        for model in self.models:
            out += model.segments()
        return out

    def homogeneous_matrices_in_global(self, q, reference_index, model_index=0, inverse=False):
        val = self.models[model_index].globalJCS(q, reference_index)
        if inverse:
            return val.transpose()
        else:
            return val

    def homogeneous_matrices_in_child(self, model_index=0, *args):
        return self.models[model_index].localJCS(*args)

    @property
    def mass(self) -> list[MX]:
        return vertcat(*(model.mass().to_mx() for model in self.models))

    def center_of_mass(self, q) -> MX:
        out = MX()
        current_q = 0
        for model in self.models:
            out = vertcat(out, model.CoM(q[current_q: model.nbQ() + current_q], True).to_mx())
            current_q += model.nbQ()
        return out

    def center_of_mass_velocity(self, q, qdot) -> MX:
        out = MX()
        current_q = 0
        current_qdot = 0
        for model in self.models:
            out = vertcat(out, model.CoMdot(q[current_q: model.nbQ() + current_q], qdot[current_qdot: model.nbQdot() + current_qdot], True).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
        return out

    def center_of_mass_acceleration(self, q, qdot, qddot) -> MX:
        out = MX()
        current_q = 0
        current_qdot = 0
        for model in self.models:
            out = vertcat(out, model.CoMddot(q[current_q: model.nbQ() + current_q],
                                            qdot[current_qdot: model.nbQdot() + current_qdot],
                                            qddot[current_qdot: model.nbQdot() + current_qdot], True).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
        return out

    def angular_momentum(self, q, qdot) -> MX:
        out = MX()
        current_q = 0
        current_qdot = 0
        for model in self.models:
            out = vertcat(out, model.angularMomentum(q[current_q: model.nbQ() + current_q],
                                            qdot[current_qdot: model.nbQdot() + current_qdot], True).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
        return out

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        return vertcat(*(model.computeQdot(q, qdot, k_stab).to_mx() for model in self.models))

    def segment_angular_velocity(self, q, qdot, idx) -> MX:
        out = MX()
        current_q = 0
        current_qdot = 0
        for model in self.models:
            out = vertcat(out, model.segmentAngularVelocity(q[current_q: model.nbQ() + current_q],
                                            qdot[current_qdot: model.nbQdot() + current_qdot], True).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
        return out

    @property
    def name_dof(self) -> tuple[str, ...]:
        out = []
        for model in self.models:
            for s in model.nameDof():
                out.append(s.to_string())
        return tuple(out)

    @property
    def contact_names(self) -> tuple[str, ...]:
        out = []
        for model in self.models:
            for s in model.contactNames():
                out.append(s.to_string())
        return tuple(out)

    @property
    def nb_soft_contacts(self) -> int:
        return sum(model.nbSoftContacts() for model in self.models)

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        out = []
        for model in self.models:
            for s in model.softContactNames():
                out.append(s.to_string())
        return tuple(out)

    def soft_contact(self, model_index=0, *args):
        return self.models[model_index].softContact(*args)

    @property
    def muscle_names(self) -> tuple[str, ...]:
        out = []
        for model in self.models:
            for s in model.muscleNames():
                out.append(s.to_string())
        return tuple(out)

    @property
    def nb_muscles(self) -> int:
        return sum(model.nbMuscles() for model in self.models)

    def torque(self, tau_activations, q, qdot) -> MX:
        out = MX()
        current_q = 0
        current_qdot = 0
        current_tau = 0
        for model in self.models:
            out = vertcat(out, model.torque(q[current_q: model.nbQ() + current_q],
                                            qdot[current_qdot: model.nbQdot() + current_qdot],
                                            tau_activations[current_tau: current_tau + model.nbGeneralizedTorque()]).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
            current_tau += model.nbGeneralizedTorque()
            model.closeActuator()
        return out

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        out = MX()
        current_q = 0
        current_qdot = 0
        current_qddot = 0
        for model in self.models:
            out = vertcat(out, model.ForwardDynamicsFreeFloatingBase(q[current_q: model.nbQ() + current_q],
                                            qdot[current_qdot: model.nbQdot() + current_qdot],
                                            qddot_joints[current_qddot: current_qddot + model.nbQddot() - model.nbRoot()]).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
            current_qddot += model.nbQddot() - model.nbRoot()
        return out

    def forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        current_q = 0
        current_qdot = 0
        current_tau = 0
        for model in self.models:
            out = vertcat(out, model.ForwardDynamics(q[current_q: model.nbQ() + current_q],
                                            qdot[current_qdot: model.nbQdot() + current_qdot],
                                            tau[current_tau: current_tau + model.nbGeneralizedTorque()],
                                            external_forces,
                                            f_contacts).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
            current_tau += model.nbGeneralizedTorque()
        return out

    def constrained_forward_dynamics(self, q, qdot, qddot, external_forces=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        current_q = 0
        current_qdot = 0
        current_qddot = 0
        for model in self.models:
            out = vertcat(out, model.ForwardDynamicsConstraintsDirect(q[current_q: model.nbQ() + current_q],
                                                     qdot[current_qdot: model.nbQdot() + current_qdot],
                                                     qddot[current_qddot: current_qddot + model.nbQddot()],
                                                     external_forces).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
            current_qddot += model.nbQddot()
        return out

    def inverse_dynamics(self, q, qdot, qddot, external_forces=None, f_contacts=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        current_q = 0
        current_qdot = 0
        current_qddot = 0
        for model in self.models:
            out = vertcat(out, model.InverseDynamics(q[current_q: model.nbQ() + current_q],
                                                                      qdot[
                                                                      current_qdot: model.nbQdot() + current_qdot],
                                                                      qddot[
                                                                      current_qddot: current_qddot + model.nbQddot()],
                                                                      external_forces,
                                                                      f_contacts).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
            current_qddot += model.nbQddot()
        return out

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        current_q = 0
        current_qdot = 0
        current_tau = 0
        for model in self.models:
            out = vertcat(out, model.ForwardDynamics(q[current_q: model.nbQ() + current_q],
                                                     qdot[current_qdot: model.nbQdot() + current_qdot],
                                                     tau[current_tau: current_tau + model.nbGeneralizedTorque()],
                                                     external_forces).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
            current_tau += model.nbGeneralizedTorque()
        return out

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        out = MX()
        current_q = 0
        current_qdot = 0
        for model in self.models:
            out = vertcat(out, model.ComputeConstraintImpulsesDirect(q[current_q: model.nbQ() + current_q],
                                            qdot_pre_impact[current_qdot: model.nbQdot() + current_qdot]).to_mx())
            current_q += model.nbQ()
            current_qdot += model.nbQdot()
        return out

    def muscle_activation_dot(self, muscle_excitations) -> MX:
        out = MX()
        for model in self.models:
            muscle_states = model.stateSet()
            for k in range(model.nbMuscles()):
                muscle_states[k].setExcitation(muscle_excitations[k])
            out = vertcat(out, model.activationDot(muscle_states).to_mx())
        return out

    def muscle_joint_torque(self, activations, q, qdot) -> MX:
        out = MX()
        for model in self.models:
            muscles_states = model.stateSet()
            for k in range(model.nbMuscles()):
                muscles_states[k].setActivation(activations[k])
            out = vertcat(out, model.muscularJointTorque(muscles_states, q, qdot).to_mx())
        return out

    def markers(self, q) -> Any | list[MX]:
        out = []
        current_q = 0
        for model in self.models:
            for m in model.markers(q[current_q: model.nbQ() + current_q]):
                out.append(m.to_mx())
            current_q += model.nbQ()
        return out

    @property
    def nb_markers(self) -> int:
        return sum(model.nbMarkers() for model in self.models)

    def marker_index(self, name, model_index=0):
        return biorbd.marker_index(self.models[model_index], name)

    def marker(self, q, index, model_index=0, reference_segment_index=None) -> MX:
        marker = self.models[model_index].marker(q, index)
        if reference_segment_index is not None:
            global_homogeneous_matrix = self.models[model_index].globalJCS(q, reference_segment_index)
            marker.applyRT(global_homogeneous_matrix.transpose())
        return marker.to_mx()

    @property
    def nb_rigid_contacts(self) -> int:
        return sum(model.nbRigidContacts() for model in self.models)

    @property
    def nb_contacts(self) -> int:
        return sum(model.nbContacts() for model in self.models)

    def marker_velocities(self, q, qdot, reference_index=None, model_index=None) -> MX:
        if model_index is None and reference_index is not None or model_index is not None and reference_index is None:
            raise RuntimeError("If reference_index is specified, model_index must be also specified")

        if reference_index is None:
            out = MX()
            current_q = 0
            current_qdot = 0
            for model in self.models:
                out = vertcat(out, horzcat(*[m.to_mx() for m in model.markersVelocity(q[current_q: model.nbQ() + current_q],
                                                                                qdot[current_qdot: model.nbQdot() + current_qdot], True)]))
                current_q += model.nbQ()
                current_qdot += model.nbQdot()

        else:
            out = MX()
            homogeneous_matrix_transposed = self.homogeneous_matrices_in_global(q, reference_index, model_index=model_index, inverse=True)
            for m in self.models[model_index].markersVelocity(q, qdot):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    out = horzcat(out, m.to_mx())

        return out

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        out_max = MX()
        out_min = MX()
        for model in self.models:
            model.closeActuator()
            torque_max, torque_min = model.torqueMax(q, qdot)
            out_max = vertcat(out_max, torque_max.to_mx())
            out_min = vertcat(out_min, torque_min.to_mx())
        return out_max, out_min

    def rigid_contact_acceleration(self, q, qdot, qddot, index) -> MX:
        # TODO: add rigid_contact_acceleration, There is a bug in biorbd_model on this function.
        raise NotImplementedError("rigid_contact_acceleration is not implemented yet for multi models")

    @property
    def nb_dof(self) -> int:
        return sum(model.nbDof() for model in self.models)

    @property
    def marker_names(self) -> tuple[str, ...]:
        out = []
        for model in self.models:
            for s in model.markerNames():
                out.append(s.to_string())
        return tuple(out)

    def soft_contact_forces(self, q, qdot) -> MX:
        out = MX()
        for model in self.models:
            soft_contact_forces = MX.zeros(self.nb_soft_contacts * 6, 1)
            for i_sc in range(self.nb_soft_contacts):
                soft_contact = self.soft_contact(i_sc)

                soft_contact_forces[i_sc * 6 : (i_sc + 1) * 6, :] = (
                    biorbd.SoftContactSphere(soft_contact).computeForceAtOrigin(self.model, q, qdot).to_mx()
                )
            out = vertcat(out, soft_contact_forces)
        return out

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
        return vertcat(*(model.passiveJointTorque(q, qdot).to_mx() for model in self.models))

    def _q_mapping(self, mapping: BiMapping | BiMappingList = None) -> BiMapping:
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

        for model in self.models:
            for i in range(model.nbSegment()):
                segment = model.segment(i)
                for var in variables:
                    if var == "q":
                        q_ranges += [q_range for q_range in segment.QRanges()]

        for var in variables:
            if var == "q":
                q_mapping = self._q_mapping(mapping)
                mapping = q_mapping
                x_min = [q_ranges[i].min() for i in q_mapping["q"].to_first.map_idx]
                x_max = [q_ranges[i].max() for i in q_mapping["q"].to_first.map_idx]
                out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

        for model in self.models:
            for i in range(model.nbSegment()):
                segment = model.segment(i)
                for var in variables:
                    if var == "qdot":
                        qdot_ranges += [qdot_range for qdot_range in segment.QDotRanges()]

        for var in variables:
            if var == "qdot":
                qdot_mapping = self._qdot_mapping(mapping)
                mapping = qdot_mapping
                x_min = [qdot_ranges[i].min() for i in qdot_mapping["qdot"].to_first.map_idx]
                x_max = [qdot_ranges[i].max() for i in qdot_mapping["qdot"].to_first.map_idx]
                out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

        if out.shape[0] == 0:
            raise ValueError(f"Unrecognized variable ({variables}), only 'q', 'qdot' and 'qddot' are allowed")

        return out