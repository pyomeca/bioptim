from typing import Callable, Any

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, norm_fro
import numpy as np
from ...limits.path_conditions import Bounds
from ...misc.utils import check_version
from ...misc.mapping import BiMapping, BiMappingList
from ..utils import _q_mapping, _qdot_mapping, _qddot_mapping, bounds_from_ranges

check_version(biorbd, "1.10.0", "1.11.0")


class BiorbdModel:
    """
    This class wraps the biorbd model and allows the user to call the biorbd functions from the biomodel protocol
    """

    def __init__(self, bio_model: str | biorbd.Model, friction_coefficients: np.ndarray = None):
        if not isinstance(bio_model, str) and not isinstance(bio_model, biorbd.Model):
            raise ValueError("The model should be of type 'str' or 'biorbd.Model'")

        self.model = biorbd.Model(bio_model) if isinstance(bio_model, str) else bio_model
        self._friction_coefficients = friction_coefficients

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

    def homogeneous_matrices_in_global(self, q, segment_id, inverse=False) -> tuple:
        # Todo: one of the last ouput of BiorbdModel which is not a MX but a biorbd object
        rt_matrix = self.model.globalJCS(GeneralizedCoordinates(q), segment_id)
        return rt_matrix.transpose() if inverse else rt_matrix

    def homogeneous_matrices_in_child(self, segment_id) -> MX:
        return self.model.localJCS(segment_id).to_mx()

    @property
    def mass(self) -> MX:
        return self.model.mass().to_mx()

    def center_of_mass(self, q) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        return self.model.CoM(q_biorbd, True).to_mx()

    def center_of_mass_velocity(self, q, qdot) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.CoMdot(q_biorbd, qdot_biorbd, True).to_mx()

    def center_of_mass_acceleration(self, q, qdot, qddot) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)
        return self.model.CoMddot(q_biorbd, qdot_biorbd, qddot_biorbd, True).to_mx()

    def body_rotation_rate(self, q, qdot) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.bodyAngularVelocity(q_biorbd, qdot_biorbd, True).to_mx()

    def mass_matrix(self, q) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        return self.model.massMatrix(q_biorbd).to_mx()

    def non_linear_effects(self, q, qdot) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.NonLinearEffect(q_biorbd, qdot_biorbd).to_mx()

    def angular_momentum(self, q, qdot) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.angularMomentum(q_biorbd, qdot_biorbd, True).to_mx()

    def reshape_qdot(self, q, qdot, k_stab=1) -> MX:
        return self.model.computeQdot(
            GeneralizedCoordinates(q),
            GeneralizedCoordinates(qdot),  # mistake in biorbd
            k_stab,
        ).to_mx()

    def segment_angular_velocity(self, q, qdot, idx) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.segmentAngularVelocity(q_biorbd, qdot_biorbd, idx, True).to_mx()

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
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.torque(tau_activations, q_biorbd, qdot_biorbd).to_mx()

    def forward_dynamics_free_floating_base(self, q, qdot, qddot_joints) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_joints_biorbd = GeneralizedAcceleration(qddot_joints)
        return self.model.ForwardDynamicsFreeFloatingBase(q_biorbd, qdot_biorbd, qddot_joints_biorbd).to_mx()

    @staticmethod
    def reorder_qddot_root_joints(qddot_root, qddot_joints) -> MX:
        return vertcat(qddot_root, qddot_joints)

    def _dispatch_forces(self, external_forces, translational_forces):
        def extract_elements(e) -> tuple[str, Any] | tuple[Any, str, Any]:
            value_message = ValueError(
                "The external_forces at each frame should be of the form: [segment_name, spatial_vector],\n"
                "where the segment_name is a str corresponding to the name of the parent and the spatial_vector\n"
                "is a 6 element vectors (Mx, My, Mz, Fx, Fy, Fz) of the type tuple, list, np.ndarray or MX."
            )
            if not isinstance(e, (list, tuple)) and len(e) < 2:
                raise value_message

            name = e[0]
            if not isinstance(name, str):
                raise value_message

            values = e[1]
            if isinstance(values, (list, tuple)):
                values = np.array(values)
            if not isinstance(values, (np.ndarray, MX)):
                raise value_message

            # If it is a force, we are done
            if len(e) < 3:
                return name, values

            # If it is a contact point, add it
            point_of_application = e[2]
            if isinstance(point_of_application, (list, tuple)):
                point_of_application = np.array(point_of_application)
            if not isinstance(point_of_application, (np.ndarray, MX)):
                raise value_message
            return values, name, point_of_application

        external_forces_set = self.model.externalForceSet()

        if external_forces is not None:
            for elements in external_forces:
                name, values = extract_elements(elements)
                external_forces_set.add(name, values)

        if translational_forces is not None:
            for elements in translational_forces:
                values, name, point_of_application = extract_elements(elements)
                external_forces_set.addTranslationalForce(values, name, point_of_application)

        return external_forces_set

    def forward_dynamics(self, q, qdot, tau, external_forces=None, translational_forces=None) -> MX:
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ForwardDynamics(q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set).to_mx()

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None, translational_forces=None) -> MX:
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ForwardDynamicsConstraintsDirect(
            q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set
        ).to_mx()

    def inverse_dynamics(self, q, qdot, qddot, external_forces=None, translational_forces=None) -> MX:
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)
        return self.model.InverseDynamics(q_biorbd, qdot_biorbd, qddot_biorbd, external_forces_set).to_mx()

    def contact_forces_from_constrained_forward_dynamics(
        self, q, qdot, tau, external_forces=None, translational_forces=None
    ) -> MX:
        external_forces_set = self._dispatch_forces(external_forces, translational_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ContactForcesFromForwardDynamicsConstraintsDirect(
            q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set
        ).to_mx()

    def qdot_from_impact(self, q, qdot_pre_impact) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_pre_impact_biorbd = GeneralizedVelocity(qdot_pre_impact)
        return self.model.ComputeConstraintImpulsesDirect(q_biorbd, qdot_pre_impact_biorbd).to_mx()

    def muscle_activation_dot(self, muscle_excitations) -> MX:
        muscle_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscle_states[k].setExcitation(muscle_excitations[k])
        return self.model.activationDot(muscle_states).to_mx()

    def muscle_length_jacobian(self, q) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        return self.model.musclesLengthJacobian(q_biorbd).to_mx()

    def muscle_velocity(self, q, qdot) -> MX:
        J = self.muscle_length_jacobian(q)
        return J @ qdot

    def muscle_joint_torque(self, activations, q, qdot) -> MX:
        muscles_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscles_states[k].setActivation(activations[k])
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx()

    def markers(self, q) -> list[MX]:
        return [m.to_mx() for m in self.model.markers(GeneralizedCoordinates(q))]

    @property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def marker(self, q, index, reference_segment_index=None) -> MX:
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
            homogeneous_matrix_transposed = self.homogeneous_matrices_in_global(
                GeneralizedCoordinates(q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersVelocity(GeneralizedCoordinates(q), GeneralizedVelocity(qdot)):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    out.append(m.to_mx())

            return out

    def marker_accelerations(self, q, qdot, qddot, reference_index=None) -> list[MX]:
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
            homogeneous_matrix_transposed = self.homogeneous_matrices_in_global(
                GeneralizedCoordinates(q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersAcceleration(
                GeneralizedCoordinates(q), GeneralizedVelocity(qdot), GeneralizedAcceleration(qddot)
            ):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    out.append(m.to_mx())

            return out

    def tau_max(self, q, qdot) -> tuple[MX, MX]:
        self.model.closeActuator()
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        torque_max, torque_min = self.model.torqueMax(q_biorbd, qdot_biorbd)
        return torque_max.to_mx(), torque_min.to_mx()

    def rigid_contact_acceleration(self, q, qdot, qddot, contact_index, contact_axis) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)
        return self.model.rigidContactAcceleration(q_biorbd, qdot_biorbd, qddot_biorbd, contact_index, True).to_mx()[
            contact_axis
        ]

    def markers_jacobian(self, q) -> list[MX]:
        return [m.to_mx() for m in self.model.markersJacobian(GeneralizedCoordinates(q))]

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.to_string() for s in self.model.markerNames()])

    def soft_contact_forces(self, q, qdot) -> MX:
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
        count = 0
        f_contact_vec = []
        for i in range(self.nb_rigid_contacts):
            contact = self.model.rigidContact(i)
            parent_name = self.model.segment(self.model.getBodyRbdlIdToBiorbdId(contact.parentId())).name().to_string()

            tp = MX.zeros(3)
            used_axes = [i for i, val in enumerate(contact.axes()) if val]
            n_contacts = len(used_axes)
            tp[used_axes] = fext[count : count + n_contacts]
            f_contact_vec.append([parent_name, tp, contact.to_mx()])
            count += n_contacts
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
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.passiveJointTorque(q_biorbd, qdot_biorbd).to_mx()

    def ligament_joint_torque(self, q, qdot) -> MX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.ligamentsJointTorque(q_biorbd, qdot_biorbd).to_mx()

    def ranges_from_model(self, variable: str):
        q_ranges = []
        qdot_ranges = []
        qddot_ranges = []

        for segment in self.segments:
            if variable == "q":
                q_ranges += [q_range for q_range in segment.QRanges()]
            elif variable == "qdot":
                qdot_ranges += [qdot_range for qdot_range in segment.QDotRanges()]
            elif variable == "qddot":
                qddot_ranges += [qddot_range for qddot_range in segment.QDDotRanges()]

        if variable == "q":
            return q_ranges
        elif variable == "qdot":
            return qdot_ranges
        elif variable == "qddot":
            return qddot_ranges
        else:
            raise RuntimeError("Wrong variable name")

    def _q_mapping(self, mapping: BiMapping = None) -> dict:
        return _q_mapping(self, mapping)

    def _qdot_mapping(self, mapping: BiMapping = None) -> dict:
        return _qdot_mapping(self, mapping)

    def _qddot_mapping(self, mapping: BiMapping = None) -> dict:
        return _qddot_mapping(self, mapping)

    def bounds_from_ranges(self, variables: str | list[str, ...], mapping: BiMapping | BiMappingList = None) -> Bounds:
        return bounds_from_ranges(self, variables, mapping)

    def lagrangian(self, q: MX | SX, qdot: MX | SX) -> MX | SX:
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.Lagrangian(q_biorbd, qdot_biorbd).to_mx()

    def partitioned_forward_dynamics(self, q_u, qdot_u, tau, external_forces=None, f_contacts=None, q_v_init=None):
        raise NotImplementedError("partitioned_forward_dynamics is not implemented for BiorbdModel")

    @staticmethod
    def animate(
        solution: Any, show_now: bool = True, tracked_markers: list[np.ndarray, ...] = None, **kwargs: Any
    ) -> None | list:
        try:
            import bioviz
        except ModuleNotFoundError:
            raise RuntimeError("bioviz must be install to animate the model")

        check_version(bioviz, "2.0.0", "2.4.0")

        states = solution.states
        if not isinstance(states, (list, tuple)):
            states = [states]

        if tracked_markers is None:
            tracked_markers = [None] * len(states)

        all_bioviz = []
        for idx_phase, data in enumerate(states):
            if not isinstance(solution.ocp.nlp[idx_phase].model, BiorbdModel):
                raise NotImplementedError("Animation is only implemented for biorbd models")

            # This calls each of the function that modify the internal dynamic model based on the parameters
            nlp = solution.ocp.nlp[idx_phase]

            # noinspection PyTypeChecker
            biorbd_model: BiorbdModel = nlp.model

            all_bioviz.append(bioviz.Viz(biorbd_model.path, **kwargs))
            if "q" in solution.ocp.nlp[idx_phase].variable_mappings:
                q = solution.ocp.nlp[idx_phase].variable_mappings["q"].to_second.map(data["q"])
            else:
                q = vertcat(data["q_roots"], data["q_joints"]).T
            all_bioviz[-1].load_movement(q)

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
