from typing import Callable, Any

import biorbd_casadi as biorbd
from casadi import SX, MX, vertcat, horzcat, norm_fro

from ..misc.utils import check_version
from ..limits.path_conditions import Bounds
from ..misc.mapping import BiMapping, BiMappingList

check_version(biorbd, "1.9.9", "1.10.0")


class MultiBiorbdModel:
    """
    This class allows to define multiple biorbd models for the same phase.
    """

    def __init__(self, bio_model: tuple[str | biorbd.Model, ...]):
        self.models = []
        if not isinstance(bio_model, tuple):
            raise RuntimeError("The models must be a 'str', 'biorbd.Model' or a tuple of 'str' or 'biorbd.Model'")

        for model in bio_model:
            if isinstance(model, str):
                self.models.append(biorbd.Model(model))
            elif isinstance(model, biorbd.Model):
                self.models.append(model)
            else:
                raise RuntimeError("The models should be of type 'str' or 'biorbd.Model'")

    def __getitem__(self, index):
        return self.models[index]

    # def deep_copy(self, *args):
    #     return MultiBiorbdModel(tuple(MultiBiorbdModel(model.DeepCopy(*args)) for model in self.models)) #@pariterre ??

    @property
    def path(self) -> list[str]:
        return [model.path().relativePath().to_string() for model in self.models]

    def copy(self):
        return MultiBiorbdModel(tuple(path for path in self.path))

    def serialize(self) -> tuple[Callable, dict]:
        return MultiBiorbdModel, dict(bio_model=tuple(path for path in self.path))

    def variable_index(self, variable: str, model_index: int) -> range:
        if variable == "q":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nbQ()
            return range(current_idx, current_idx + self.models[model_index].nbQ())
        elif variable == "qdot":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nbQdot()
            return range(current_idx, current_idx + self.models[model_index].nbQdot())
        elif variable == "qddot":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nbQddot()
            return range(current_idx, current_idx + self.models[model_index].nbQddot())
        elif variable == "qddot_joints":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nbQddot() - model.nbRoot()
            return range(
                current_idx, current_idx + self.models[model_index].nbQddot() - self.models[model_index].nbRoot()
            )
        elif variable == "tau":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nbGeneralizedTorque()
            return range(current_idx, current_idx + self.models[model_index].nbGeneralizedTorque())
        elif variable == "contact":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nbRigidContacts()
            return range(current_idx, current_idx + self.models[model_index].nbRigidContacts())

    def transform_to_generalized_coordinates(self, q: MX):
        return biorbd.GeneralizedCoordinates(q)

    def transform_to_generalized_velocities(self, qdot: MX):
        return biorbd.GeneralizedVelocity(qdot)

    def transform_to_generalized_torques(self, tau: MX):
        return biorbd.GeneralizedTorque(tau)

    def transform_to_generalized_accelerations(self, qddot: MX):
        return biorbd.GeneralizedAcceleration(qddot)

    @property
    def gravity(self) -> MX:
        return vertcat(*(model.getGravity().to_mx() for model in self.models))

    def set_gravity(self, new_gravity) -> None:
        for model in self.models:
            model.setGravity(new_gravity)
        return

    @property
    def nb_tau(self) -> int:
        return sum(model.nbGeneralizedTorque() for model in self.models)

    @property
    def nb_segments(self) -> int:
        return sum(model.nbSegment() for model in self.models)

    def segment_index(self, name) -> int:
        raise NotImplementedError("segment_index is not implemented for MultiBiorbdModel")

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

    def homogeneous_matrices_in_global(self, q, reference_index, inverse=False):
        raise NotImplementedError("homogeneous_matrices_in_global is not implemented for MultiBiorbdModel")

    def homogeneous_matrices_in_child(self, *args):
        raise NotImplementedError("homogeneous_matrices_in_child is not implemented for MultiBiorbdModel")

    @property
    def mass(self) -> list[MX]:
        return vertcat(*(model.mass().to_mx() for model in self.models))

    def center_of_mass(self, q) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            out = vertcat(out, model.CoM(q_biorbd, True).to_mx())
        return out

    def center_of_mass_velocity(self, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            out = vertcat(
                out,
                model.CoMdot(q_biorbd, qdot_biorbd, True).to_mx(),
            )
        return out

    def center_of_mass_acceleration(self, q, qdot, qddot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            qddot_biorbd = self.transform_to_generalized_accelerations(qddot[self.variable_index("qddot", i)])
            out = vertcat(
                out,
                model.CoMddot(
                    q_biorbd,
                    qdot_biorbd,
                    qddot_biorbd,
                    True,
                ).to_mx(),
            )
        return out

    def angular_momentum(self, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            out = vertcat(
                out,
                model.angularMomentum(q_biorbd, qdot_biorbd, True).to_mx(),
            )
        return out

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_coordinates(
                qdot[self.variable_index("qdot", i)]
            )  # Due to a bug in biorbd
            out = vertcat(
                out,
                model.computeQdot(q_biorbd, qdot_biorbd, k_stab).to_mx(),
            )
        return out

    def segment_angular_velocity(self, q, qdot, idx) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            out = vertcat(
                out,
                model.segmentAngularVelocity(
                    q_biorbd,
                    qdot_biorbd,
                    idx,
                    True,
                ).to_mx(),
            )
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

    def soft_contact(self, soft_contact_index, *args):
        current_number_of_soft_contacts = 0
        out = []
        for model in self.models:
            if soft_contact_index < current_number_of_soft_contacts + model.nbSoftContacts():
                out = model.softContact(soft_contact_index - current_number_of_soft_contacts, *args)
                break
            current_number_of_soft_contacts += model.nbSoftContacts()
        return out

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
        for i, model in enumerate(self.models):
            out = vertcat(
                out,
                model.torque(
                    tau_activations,
                    q,
                    qdot,
                ).to_mx(),
            )
            model.closeActuator()
        return out

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            qddot_joints_biorbd = self.transform_to_generalized_accelerations(
                qddot_joints[self.variable_index("qddot_joints", i)]
            )
            out = vertcat(
                out,
                model.ForwardDynamicsFreeFloatingBase(
                    q_biorbd,
                    qdot_biorbd,
                    qddot_joints_biorbd,
                ).to_mx(),
            )
        return out

    def forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            tau_biorbd = self.transform_to_generalized_torques(tau[self.variable_index("tau", i)])
            out = vertcat(
                out,
                model.ForwardDynamics(
                    q_biorbd,
                    qdot_biorbd,
                    tau_biorbd,
                    external_forces,
                    f_contacts,
                ).to_mx(),
            )
        return out

    def constrained_forward_dynamics(self, q, qdot, qddot, external_forces=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            qddot_biorbd = self.transform_to_generalized_torques(
                qddot[self.variable_index("qddot", i)]
            )  # Due to a bug in biorbd
            out = vertcat(
                out,
                model.ForwardDynamicsConstraintsDirect(
                    q_biorbd,
                    qdot_biorbd,
                    qddot_biorbd,
                    external_forces,
                ).to_mx(),
            )
        return out

    def inverse_dynamics(self, q, qdot, qddot, external_forces=None, f_contacts=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            qddot_biorbd = self.transform_to_generalized_accelerations(qddot[self.variable_index("qddot", i)])
            out = vertcat(
                out,
                model.InverseDynamics(
                    q_biorbd,
                    qdot_biorbd,
                    qddot_biorbd,
                    external_forces,
                    f_contacts,
                ).to_mx(),
            )
        return out

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            tau_biorbd = self.transform_to_generalized_torques(tau[self.variable_index("tau", i)])
            out = vertcat(
                out,
                model.ContactForcesFromForwardDynamicsConstraintsDirect(
                    q_biorbd,
                    qdot_biorbd,
                    tau_biorbd,
                    external_forces,
                ).to_mx(),
            )
        return out

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_pre_impact_biorbd = self.transform_to_generalized_velocities(
                qdot_pre_impact[self.variable_index("qdot", i)]
            )
            out = vertcat(
                out,
                model.ComputeConstraintImpulsesDirect(
                    q_biorbd,
                    qdot_pre_impact_biorbd,
                ).to_mx(),
            )
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
            q_biorbd = self.transform_to_generalized_coordinates(q)
            qdot_biorbd = self.transform_to_generalized_velocities(qdot)
            out = vertcat(out, model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx())
        return out

    def markers(self, q) -> Any | list[MX]:
        out = []
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            for m in model.markers(q_biorbd):
                out.append(m.to_mx())
        return out

    @property
    def nb_markers(self) -> int:
        return sum(model.nbMarkers() for model in self.models)

    def marker_index(self, name):
        raise NotImplementedError("marker_index is not implemented yet for MultiBiorbdModel")

    def marker(self, q, index, reference_segment_index=None) -> MX:
        raise NotImplementedError("marker is not implemented yet for MultiBiorbdModel")

    @property
    def nb_rigid_contacts(self) -> int:
        """
        Returns the number of rigid contacts.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_rigid_contacts = 2
        """
        return sum(model.nbRigidContacts() for model in self.models)

    @property
    def nb_contacts(self) -> int:
        """
        Returns the number of contact index.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_contacts = 3
        """
        return sum(model.nbContacts() for model in self.models)

    def rigid_contact_index(self, contact_index) -> tuple:
        """
        Returns the axis index of this specific rigid contact.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            rigid_contact_index(0) = (1, 2)
        """
        for i, model in enumerate(self.models):
            if contact_index in self.variable_index("contact", i):
                model_selected = model
        return model_selected.rigidContactAxisIdx(contact_index)

    def marker_velocities(self, q, qdot, reference_index=None) -> MX:
        if reference_index is not None:
            raise RuntimeError("marker_velocities is not implemented yet with reference_index for MultiBiorbdModel")

        out = MX()
        for i, model in enumerate(self.models):
            q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
            qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
            out = vertcat(
                out,
                horzcat(
                    *[
                        m.to_mx()
                        for m in model.markersVelocity(
                            q_biorbd,
                            qdot_biorbd,
                            True,
                        )
                    ]
                ),
            )
        return out

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        out_max = MX()
        out_min = MX()
        for model in self.models:
            model.closeActuator()
            q_biorbd = self.transform_to_generalized_coordinates(q)
            qdot_biorbd = self.transform_to_generalized_velocities(qdot)
            torque_max, torque_min = model.torqueMax(q_biorbd, qdot_biorbd)
            out_max = vertcat(out_max, torque_max.to_mx())
            out_min = vertcat(out_min, torque_min.to_mx())
        return out_max, out_min

    def rigid_contact_acceleration(self, q, qdot, qddot, contact_index, contact_axis) -> MX:
        for i, model in enumerate(self.models):
            if contact_index in self.variable_index("contact", i):
                model_selected = model
        q_biorbd = self.transform_to_generalized_coordinates(q[self.variable_index("q", i)])
        qdot_biorbd = self.transform_to_generalized_velocities(qdot[self.variable_index("qdot", i)])
        qddot_biorbd = self.transform_to_generalized_accelerations(qddot[self.variable_index("qddot", i)])
        return model.rigidContactAcceleration(q_biorbd, qdot_biorbd, qddot_biorbd, contact_index, True).to_mx()[
            contact_axis
        ]

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
        q_biorbd = self.transform_to_generalized_coordinates(q)
        qdot_biorbd = self.transform_to_generalized_velocities(qdot)
        out = MX()
        for model in self.models:
            soft_contact_forces = MX.zeros(self.nb_soft_contacts * 6, 1)
            for i_sc in range(self.nb_soft_contacts):
                soft_contact = self.soft_contact(i_sc)

                soft_contact_forces[i_sc * 6 : (i_sc + 1) * 6, :] = (
                    biorbd.SoftContactSphere(soft_contact).computeForceAtOrigin(model, q_biorbd, qdot_biorbd).to_mx()
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
        q_biorbd = self.transform_to_generalized_coordinates(q)
        qdot_biorbd = self.transform_to_generalized_velocities(qdot)
        return vertcat(*(model.passiveJointTorque(q_biorbd, qdot_biorbd).to_mx() for model in self.models))

    def ligament_joint_torque(self, q, qdot) -> MX:
        q_biorbd = self.transform_to_generalized_coordinates(q)
        qdot_biorbd = self.transform_to_generalized_velocities(qdot)
        return self.model.ligamentsJointTorque(q_biorbd, qdot_biorbd).to_mx()

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

        for model in self.models:
            for i in range(model.nbSegment()):
                segment = model.segment(i)
                for var in variables:
                    if var == "q":
                        q_ranges += [q_range for q_range in segment.QRanges()]
                    elif var == "qdot":
                        qdot_ranges += [qdot_range for qdot_range in segment.QDotRanges()]
                    elif var == "qddot":
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


class BiorbdModel(MultiBiorbdModel):
    """
    This class allows to define a biorbd model.
    """

    def __init__(self, bio_model: str | biorbd.Model):
        if not isinstance(bio_model, str) and not isinstance(bio_model, biorbd.Model):
            raise RuntimeError("The model should be of type 'str' or 'biorbd.Model'")

        super(BiorbdModel, self).__init__(tuple([bio_model]))

    @property
    def model(self):
        """
        Returns the first model for retro-compatibility with single model definition

        Returns
        -------
        The states data
        """

        return self.models[0]

    def segment_index(self, name) -> int:
        return biorbd.segment_index(self.model, name)

    def homogeneous_matrices_in_global(self, q, reference_index, inverse=False):
        val = self.model.globalJCS(self.transform_to_generalized_coordinates(q), reference_index)
        if inverse:
            return val.transpose()
        else:
            return val

    def homogeneous_matrices_in_child(self, *args):
        return self.model.localJCS(*args)

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def marker(self, q, index, reference_segment_index=None) -> MX:
        marker = self.model.marker(self.transform_to_generalized_coordinates(q), index)
        if reference_segment_index is not None:
            global_homogeneous_matrix = self.model.globalJCS(self.transform_to_generalized_coordinates(q), reference_segment_index)
            marker.applyRT(global_homogeneous_matrix.transpose())
        return marker.to_mx()

    def marker_velocities(self, q, qdot, reference_index=None) -> MX:
        if reference_index is None:
            return horzcat(
                *[
                    m.to_mx()
                    for m in self.model.markersVelocity(
                        self.transform_to_generalized_coordinates(q),
                        self.transform_to_generalized_velocities(qdot),
                        True,
                    )
                ]
            )

        else:
            out = MX()
            homogeneous_matrix_transposed = self.homogeneous_matrices_in_global(
                self.transform_to_generalized_coordinates(q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersVelocity(self.transform_to_generalized_coordinates(q), self.transform_to_generalized_velocities(qdot)):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    out = horzcat(out, m.to_mx())

            return out

    def segment_angular_velocity(self, q, qdot, segment_index) -> MX:
        q_biorbd = self.transform_to_generalized_coordinates(q)
        qdot_biorbd = self.transform_to_generalized_velocities(qdot)
        return self.model.segmentAngularVelocity(q_biorbd, qdot_biorbd, segment_index, True).to_mx()

    @property
    def path(self) -> list[str]:
        # This is for retro compatibility with bioviz in animate
        return self.model.path().relativePath().to_string()

    def copy(self):
        return BiorbdModel(self.path)

    def serialize(self) -> tuple[Callable, dict]:
        return BiorbdModel, dict(bio_model=self.path)
