from typing import Callable, Any

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
)
from casadi import SX, MX, vertcat

from ...misc.utils import check_version
from ...limits.path_conditions import Bounds
from ...misc.mapping import BiMapping, BiMappingList
from ..utils import _var_mapping, bounds_from_ranges
from ...optimization.solution.solution_data import SolutionMerge
from ..utils import bounds_from_ranges
from .biorbd_model import BiorbdModel


class MultiBiorbdModel:
    """
    This class allows to define multiple biorbd models for the same phase.


    Attributes
    ----------
    models : list[BiorbdModel]
        The list of biorbd models to be handled in the optimal control program.
    extra_models : list[BiorbdModel]
        A list of extra biorbd models stored in the class for further use.

    Methods
    -------
    variable_index()
        Get the index of the variables in the global vector for a given model index.
    nb_models()
        Get the number of models.
    nb_extra_models()
        Get the number of extra models.

    """

    def __init__(
        self,
        bio_model: tuple[str | biorbd.Model | BiorbdModel, ...],
        extra_bio_models: tuple[str | biorbd.Model | BiorbdModel, ...] = (),
    ):
        self.models = []
        if not isinstance(bio_model, tuple):
            raise ValueError("The models must be a 'str', 'biorbd.Model', 'bioptim.BiorbdModel'" " or a tuple of those")

        for model in bio_model:
            if isinstance(model, str):
                self.models.append(BiorbdModel(model))
            elif isinstance(model, biorbd.Model):
                self.models.append(BiorbdModel(model))
            elif isinstance(model, BiorbdModel):
                self.models.append(model)
            else:
                raise ValueError("The models should be of type 'str', 'biorbd.Model' or 'bioptim.BiorbdModel'")

        if not isinstance(extra_bio_models, tuple):
            raise ValueError("The models must be a 'str', 'biorbd.Model', 'bioptim.BiorbdModel'" " or a tuple of those")

        self.extra_models = []
        for model in extra_bio_models:
            if isinstance(model, str):
                self.extra_models.append(BiorbdModel(model))
            elif isinstance(model, biorbd.Model):
                self.extra_models.append(BiorbdModel(model))
            elif isinstance(model, BiorbdModel):
                self.extra_models.append(model)
            else:
                raise ValueError("The models should be of type 'str', 'biorbd.Model' or 'bioptim.BiorbdModel'")

    def __getitem__(self, index):
        return self.models[index]

    def deep_copy(self, *args):
        raise NotImplementedError("Deep copy is not implemented yet for MultiBiorbdModel class")

    @property
    def path(self) -> (list[str], list[str]):
        return [model.path for model in self.models], [model.path for model in self.extra_models]

    def copy(self):
        return MultiBiorbdModel(tuple(self.path[0]), tuple(self.path[1]))

    def serialize(self) -> tuple[Callable, dict]:
        return MultiBiorbdModel, dict(bio_model=tuple(self.path[0]), extra_bio_models=tuple(self.path[1]))

    def variable_index(self, variable: str, model_index: int) -> range:
        """
        Get the index of the variables in the global vector for a given model index

        Parameters
        ----------
        variable: str
            The variable to get the index from such as 'q', 'qdot', 'qddot', 'tau', 'contact', 'markers'
        model_index: int
            The index of the model to get the index from

        Returns
        -------
        range
            The index of the variable in the global vector
        """
        current_idx = 0

        if variable == "q":
            for model in self.models[:model_index]:
                current_idx += model.nb_q
            return range(current_idx, current_idx + self.models[model_index].nb_q)

        elif variable == "qdot":
            for model in self.models[:model_index]:
                current_idx += model.nb_qdot
            return range(current_idx, current_idx + self.models[model_index].nb_qdot)

        elif variable == "qddot":
            for model in self.models[:model_index]:
                current_idx += model.nb_qddot
            return range(current_idx, current_idx + self.models[model_index].nb_qddot)

        elif variable == "qddot_joints":
            for model in self.models[:model_index]:
                current_idx += model.nb_qddot - model.nb_root
            return range(
                current_idx, current_idx + self.models[model_index].nb_qddot - self.models[model_index].nb_root
            )

        elif variable == "qddot_root":
            for model in self.models[:model_index]:
                current_idx += model.nb_root
            return range(current_idx, current_idx + self.models[model_index].nb_root)

        elif variable == "tau":
            for model in self.models[:model_index]:
                current_idx += model.nb_tau
            return range(current_idx, current_idx + self.models[model_index].nb_tau)

        elif variable == "contact":
            for model in self.models[:model_index]:
                current_idx += model.nb_rigid_contacts
            return range(current_idx, current_idx + self.models[model_index].nb_rigid_contacts)

        elif variable == "markers":
            for model in self.models[:model_index]:
                current_idx += model.nb_markers
            return range(current_idx, current_idx + self.models[model_index].nb_markers)

        elif variable == "segment":
            for model in self.models[:model_index]:
                current_idx += model.nb_segments
            return range(current_idx, current_idx + self.models[model_index].nb_segments)

        else:
            raise ValueError(
                "The variable must be 'q', 'qdot', 'qddot', 'tau', 'contact' or 'markers'" f" and {variable} was sent."
            )

    def global_variable_id(self, variable: str, model_index: int, model_variable_id: int) -> int:
        """
        Get the id of the variable in the global vector for a given model index

        Parameters
        ----------
        variable: str
            The variable to get the index from such as 'q', 'qdot', 'qddot', 'tau', 'contact', 'markers'
        model_index: int
            The index of the model to get the index from
        model_variable_id: int
            The id of the variable in the model vector

        Returns
        -------
        int
            The id of the variable in the global vector
        """
        return self.variable_index(variable, model_index)[model_variable_id]

    def local_variable_id(self, variable: str, global_index: int) -> tuple[int, int]:
        """
        Get the id of the variable in the local vector and the model index for a given index of the global vector

        Parameters
        ----------
        variable: str
            The variable to get the index from such as 'q', 'qdot', 'qddot', 'tau', 'contact', 'markers'
        global_index: int
            The index of the variable in the global vector

        Returns
        -------
        tuple(int, int)
            The id of the variable in the local vector and the model index
        """

        for model_id, model in enumerate(self.models):
            if global_index in self.variable_index(variable, model_id):
                return global_index - self.variable_index(variable, model_id)[0], model_id

    @property
    def nb_models(self) -> int:
        """
        Get the number of models

        Returns
        -------
        int
            The number of models
        """
        return len(self.models)

    @property
    def nb_extra_models(self) -> int:
        """
        Get the number of extra models

        Returns
        -------
        int
            The number of extra models
        """
        return len(self.extra_models)

    @property
    def gravity(self) -> MX:
        return vertcat(*(model.gravity for model in self.models))

    def set_gravity(self, new_gravity) -> None:
        for model in self.models:
            model.set_gravity(new_gravity)
        return

    @property
    def nb_tau(self) -> int:
        return sum(model.nb_tau for model in self.models)

    @property
    def nb_segments(self) -> int:
        return sum(model.nb_segments for model in self.models)

    def segment_index(self, name) -> int:
        raise NotImplementedError("segment_index is not implemented for MultiBiorbdModel")

    @property
    def nb_quaternions(self) -> int:
        return sum(model.nb_quaternions for model in self.models)

    @property
    def nb_q(self) -> int:
        return sum(model.nb_q for model in self.models)

    @property
    def nb_qdot(self) -> int:
        return sum(model.nb_qdot for model in self.models)

    @property
    def nb_qddot(self) -> int:
        return sum(model.nb_qddot for model in self.models)

    @property
    def nb_root(self) -> int:
        return sum(model.nb_root for model in self.models)

    @property
    def segments(self) -> tuple[biorbd.Segment, ...]:
        out = ()
        for model in self.models:
            out += model.segments
        return out

    def biorbd_homogeneous_matrices_in_global(self, q, segment_idx, inverse=False) -> biorbd.RotoTrans:
        local_segment_id, model_id = self.local_variable_id("segment", segment_idx)
        q_model = q[self.variable_index("q", model_id)]
        return self.models[model_id].homogeneous_matrices_in_global(q_model, local_segment_id, inverse)

    def homogeneous_matrices_in_global(self, q, segment_idx, inverse=False) -> MX:
        return self.biorbd_homogeneous_matrices_in_global(q, segment_idx, inverse).to_mx()

    def homogeneous_matrices_in_child(self, segment_id) -> MX:
        local_id, model_id = self.local_variable_id("segment", segment_id)
        return self.models[model_id].homogeneous_matrices_in_child(local_id)

    @property
    def mass(self) -> MX:
        return vertcat(*(model.mass for model in self.models))

    def center_of_mass(self, q) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            out = vertcat(out, model.center_of_mass(q_model))
        return out

    def center_of_mass_velocity(self, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(
                out,
                model.center_of_mass_velocity(q_model, qdot_model),
            )
        return out

    def center_of_mass_acceleration(self, q, qdot, qddot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            qddot_model = qddot[self.variable_index("qddot", i)]
            out = vertcat(
                out,
                model.center_of_mass_acceleration(q_model, qdot_model, qddot_model),
            )
        return out

    def mass_matrix(self, q) -> list[MX]:
        out = []
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            out += [model.mass_matrix(q_model)]
        return out

    def non_linear_effects(self, q, qdot) -> list[MX]:
        out = []
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out += [model.non_linear_effects(q_model, qdot_model)]
        return out

    def angular_momentum(self, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(
                out,
                model.angular_momentum(q_model, qdot_model),
            )
        return out

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(
                out,
                model.reshape_qdot(q_model, qdot_model, k_stab),
            )
        return out

    def segment_angular_velocity(self, q, qdot, idx) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(
                out,
                model.segment_angular_velocity(q_model, qdot_model, idx),
            )
        return out

    @property
    def name_dof(self) -> tuple[str, ...]:
        return tuple([dof for model in self.models for dof in model.name_dof])

    @property
    def contact_names(self) -> tuple[str, ...]:
        return tuple([contact for model in self.models for contact in model.contact_names])

    @property
    def nb_soft_contacts(self) -> int:
        return sum(model.nb_soft_contacts for model in self.models)

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        return tuple([contact for model in self.models for contact in model.soft_contact_names])

    def soft_contact(self, soft_contact_index, *args):
        current_number_of_soft_contacts = 0
        out = []
        for model in self.models:
            if soft_contact_index < current_number_of_soft_contacts + model.nb_soft_contacts:
                out = model.soft_contact(soft_contact_index - current_number_of_soft_contacts, *args)
                break
            current_number_of_soft_contacts += model.nb_soft_contacts
        return out

    @property
    def muscle_names(self) -> tuple[str, ...]:
        return tuple([muscle for model in self.models for muscle in model.muscle_names])

    @property
    def nb_muscles(self) -> int:
        return sum(model.nb_muscles for model in self.models)

    def torque(self, tau_activations, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            tau_activations_model = tau_activations[self.variable_index("tau", i)]
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(
                out,
                model.torque(
                    tau_activations_model,
                    q_model,
                    qdot_model,
                ),
            )
        return out

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            qddot_joints_model = qddot_joints[self.variable_index("qddot_joints", i)]
            out = vertcat(
                out,
                model.forward_dynamics_free_floating_base(
                    q_model,
                    qdot_model,
                    qddot_joints_model,
                ),
            )
        return out

    def reorder_qddot_root_joints(self, qddot_root, qddot_joints):
        out = MX()
        for i, model in enumerate(self.models):
            qddot_root_model = qddot_root[self.variable_index("qddot_root", i)]
            qddot_joints_model = qddot_joints[self.variable_index("qddot_joints", i)]
            out = vertcat(
                out,
                model.reorder_qddot_root_joints(qddot_root_model, qddot_joints_model),
            )

        return out

    def forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        if f_contacts is not None or external_forces is not None:
            raise NotImplementedError(
                "External forces and contact forces are not implemented yet for MultiBiorbdModel."
            )

        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            tau_model = tau[self.variable_index("tau", i)]
            out = vertcat(
                out,
                model.forward_dynamics(
                    q_model,
                    qdot_model,
                    tau_model,
                    external_forces,
                    f_contacts,
                ),
            )
        return out

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet for MultiBiorbdModel.")

        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            tau_model = tau[self.variable_index("qddot", i)]  # Due to a bug in biorbd
            out = vertcat(
                out,
                model.constrained_forward_dynamics(
                    q_model,
                    qdot_model,
                    tau_model,
                    external_forces,
                ),
            )
        return out

    def inverse_dynamics(self, q, qdot, qddot, external_forces=None, f_contacts=None) -> MX:
        if f_contacts is not None or external_forces is not None:
            raise NotImplementedError(
                "External forces and contact forces are not implemented yet for MultiBiorbdModel."
            )

        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            qddot_model = qddot[self.variable_index("qddot", i)]
            out = vertcat(
                out,
                model.inverse_dynamics(
                    q_model,
                    qdot_model,
                    qddot_model,
                    external_forces,
                    f_contacts,
                ),
            )
        return out

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet for MultiBiorbdModel.")
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            tau_model = tau[self.variable_index("qddot", i)]  # Due to a bug in biorbd
            out = vertcat(
                out,
                model.contact_forces_from_constrained_forward_dynamics(
                    q_model,
                    qdot_model,
                    tau_model,
                    external_forces,
                ),
            )
        return out

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_pre_impact_model = qdot_pre_impact[self.variable_index("qdot", i)]
            out = vertcat(
                out,
                model.qdot_from_impact(
                    q_model,
                    qdot_pre_impact_model,
                ),
            )
        return out

    def muscle_activation_dot(self, muscle_excitations) -> MX:
        out = MX()
        for model in self.models:
            muscle_states = model.model.stateSet()  # still call from Biorbd
            for k in range(model.nb_muscles):
                muscle_states[k].setExcitation(muscle_excitations[k])
            out = vertcat(out, model.model.activationDot(muscle_states).to_mx())
        return out

    def muscle_joint_torque(self, activations, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            muscles_states = model.model.stateSet()  # still call from Biorbd
            for k in range(model.nb_muscles):
                muscles_states[k].setActivation(activations[k])
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(out, model.model.muscularJointTorque(muscles_states, q_model, qdot_model).to_mx())
        return out

    def markers(self, q) -> Any | list[MX]:
        out = []
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            out.append(model.markers(q_model))
        return [item for sublist in out for item in sublist]

    @property
    def nb_markers(self) -> int:
        return sum(model.nb_markers for model in self.models)

    def marker_index(self, name):
        for i, model in enumerate(self.models):
            if name in model.marker_names:
                marker_id = biorbd.marker_index(model.model, name)
                return self.variable_index("markers", model_index=i)[marker_id]

        raise ValueError(f"{name} is not in the MultiBiorbdModel")

    def marker(self, q, index, reference_segment_index=None) -> MX:
        local_marker_id, model_id = self.local_variable_id("markers", index)
        q_model = q[self.variable_index("q", model_id)]

        return self.models[model_id].marker(q_model, local_marker_id, reference_segment_index)

    @property
    def nb_rigid_contacts(self) -> int:
        """
        Returns the number of rigid contacts.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_rigid_contacts = 2
        """
        return sum(model.nb_rigid_contacts for model in self.models)

    @property
    def nb_contacts(self) -> int:
        """
        Returns the number of contact index.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            nb_contacts = 3
        """
        return sum(model.nb_contacts for model in self.models)

    def rigid_contact_index(self, contact_index) -> tuple:
        """
        Returns the axis index of this specific rigid contact.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            rigid_contact_index(0) = (1, 2)
        """

        model_selected = None
        for i, model in enumerate(self.models):
            if contact_index in self.variable_index("contact", i):
                model_selected = model
            # Note: may not work if the contact_index is not in the first model
        return model_selected.rigid_contact_index(contact_index)

    def marker_velocities(self, q, qdot, reference_index=None) -> list[MX]:
        if reference_index is not None:
            raise RuntimeError("marker_velocities is not implemented yet with reference_index for MultiBiorbdModel")

        out = []
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out.extend(
                model.marker_velocities(q_model, qdot_model, reference_index),
            )
        return out

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        out_max = MX()
        out_min = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            torque_max, torque_min = model.tau_max(q_model, qdot_model)
            out_max = vertcat(out_max, torque_max)
            out_min = vertcat(out_min, torque_min)
        return out_max, out_min

    def rigid_contact_acceleration(self, q, qdot, qddot, contact_index, contact_axis) -> MX:
        model_selected = None
        model_idx = -1
        for i, model in enumerate(self.models):
            if contact_index in self.variable_index("contact", i):
                model_selected = model
                model_idx = i
        q_model = q[self.variable_index("q", model_idx)]
        qdot_model = qdot[self.variable_index("qdot", model_idx)]
        qddot_model = qddot[self.variable_index("qddot", model_idx)]
        return model_selected.rigid_contact_acceleration(q_model, qdot_model, qddot_model, contact_index, contact_axis)

    @property
    def nb_dof(self) -> int:
        return sum(model.nb_dof for model in self.models)

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([name for model in self.models for name in model.marker_names])

    def soft_contact_forces(self, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            soft_contact_forces = model.soft_contact_forces(q_model, qdot_model)
            out = vertcat(out, soft_contact_forces)
        return out

    def reshape_fext_to_fcontact(self, fext: MX) -> biorbd.VecBiorbdVector:
        raise NotImplementedError("reshape_fext_to_fcontact is not implemented yet for MultiBiorbdModel")

    def normalize_state_quaternions(self, x: MX | SX) -> MX | SX:
        all_q_normalized = MX()
        for i, model in enumerate(self.models):
            q_model = x[self.variable_index("q", i)]  # quaternions are only in q
            q_normalized = model.normalize_state_quaternions(q_model)
            all_q_normalized = vertcat(all_q_normalized, q_normalized)
        idx_first_qdot = self.nb_q  # assuming x = [q, qdot]
        x_normalized = vertcat(all_q_normalized, x[idx_first_qdot:])

        return x_normalized

    def contact_forces(self, q, qdot, tau, external_forces: list = None) -> MX:
        if external_forces is not None:
            raise NotImplementedError("contact_forces is not implemented yet with external_forces for MultiBiorbdModel")

        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            tau_model = tau[self.variable_index("tau", i)]

            contact_forces = model.contact_forces(q_model, qdot_model, tau_model, external_forces)
            out = vertcat(out, contact_forces)

        return out

    def passive_joint_torque(self, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(out, model.passive_joint_torque(q_model, qdot_model))
        return out

    def ligament_joint_torque(self, q, qdot) -> MX:
        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(out, model.ligament_joint_torque(q_model, qdot_model))
        return out

    def ranges_from_model(self, variable: str):
        return [the_range for model in self.models for the_range in model.ranges_from_model(variable)]

    def bounds_from_ranges(self, variables: str | list[str, ...], mapping: BiMapping | BiMappingList = None) -> Bounds:
        return bounds_from_ranges(self, variables, mapping)

    def _var_mapping(self, key: str, range_for_mapping: int | list | tuple | range, mapping: BiMapping = None) -> dict:
        return _var_mapping(key, range_for_mapping, mapping)

    def lagrangian(self):
        raise NotImplementedError("lagrangian is not implemented yet for MultiBiorbdModel")

    def partitioned_forward_dynamics(self, q_u, qdot_u, tau, external_forces=None, f_contacts=None, q_v_init=None):
        raise NotImplementedError("partitioned_forward_dynamics is not implemented yet for MultiBiorbdModel")

    @staticmethod
    def animate(solution: Any, show_now: bool = True, tracked_markers: list = None, **kwargs: Any) -> None | list:
        try:
            import bioviz
        except ModuleNotFoundError:
            raise RuntimeError("bioviz must be install to animate the model")

        check_version(bioviz, "2.3.0", "2.4.0")

        states = solution.stepwise_states(to_merge=SolutionMerge.NODES)
        if not isinstance(states, (list, tuple)):
            states = [states]

        if tracked_markers is None:
            tracked_markers = [None] * len(states)

        all_bioviz = []
        for idx_phase, data in enumerate(states):
            # This calls each of the function that modify the internal dynamic model based on the parameters
            nlp = solution.ocp.nlp[idx_phase]

            if isinstance(nlp.model, MultiBiorbdModel):
                if nlp.model.nb_models > 1:
                    raise NotImplementedError(
                        f"Animation is only implemented for MultiBiorbdModel with 1 model."
                        f" There are {nlp.model.nb_models} models in the phase {idx_phase}."
                    )
                else:
                    model = nlp.model.models[0]

            biorbd_model: BiorbdModel = model

            all_bioviz.append(bioviz.Viz(biorbd_model.path, **kwargs))
            all_bioviz[-1].load_movement(solution.ocp.nlp[idx_phase].variable_mappings["q"].to_second.map(data["q"]))

            if tracked_markers[idx_phase] is not None:
                all_bioviz[-1].load_experimental_markers(tracked_markers[idx_phase])

        if show_now:
            b_is_visible = [True] * len(all_bioviz)
            while sum(b_is_visible):
                for i, b in enumerate(all_bioviz):
                    if b.vtk_window.is_active:
                        b.update()
                    else:
                        b_is_visible[i] = False
            return None
        else:
            return all_bioviz
