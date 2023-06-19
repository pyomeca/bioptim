from typing import Callable, Any

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, DM, vertcat, horzcat, norm_fro, Function

from ..misc.enums import ControlType, QuadratureRule
from ..misc.utils import check_version
from ..limits.path_conditions import Bounds
from ..misc.mapping import BiMapping, BiMappingList

check_version(biorbd, "1.9.9", "1.10.0")


def _q_mapping(model, mapping: BiMapping = None) -> dict:
    """
    This function returns a standard mapping for the q states if None
    and checks if the model has quaternions
    """
    if mapping is None:
        mapping = {}
    if model.nb_quaternions > 0:
        if "q" in mapping and "qdot" not in mapping:
            raise RuntimeError(
                "It is not possible to provide a q_mapping but not a qdot_mapping if the model have quaternion"
            )
        elif "q" not in mapping and "qdot" in mapping:
            raise RuntimeError(
                "It is not possible to provide a qdot_mapping but not a q_mapping if the model have quaternion"
            )
    if "q" not in mapping:
        mapping["q"] = BiMapping(range(model.nb_q), range(model.nb_q))
    return mapping


def _qdot_mapping(model, mapping: BiMapping = None) -> dict:
    """
    This function returns a standard mapping for the qdot states if None
    and checks if the model has quaternions
    """
    if mapping is None:
        mapping = {}
    if "qdot" not in mapping:
        mapping["qdot"] = BiMapping(range(model.nb_qdot), range(model.nb_qdot))

    return mapping


def _qddot_mapping(model, mapping: BiMapping = None) -> dict:
    """
    This function returns a standard mapping for the qddot states if None
    and checks if the model has quaternions
    """
    if mapping is None:
        mapping = {}
    if "qddot" not in mapping:
        mapping["qddot"] = BiMapping(range(model.nb_qddot), range(model.nb_qddot))

    return mapping


def bounds_from_ranges(model, variables: str | list[str, ...], mapping: BiMapping | BiMappingList = None) -> Bounds:
    """
    Generate bounds from the ranges of the model

    Parameters
    ----------
    model: bio_model
        such as BiorbdModel or MultiBiorbdModel
    variables: str | list[str, ...]
        The variables to generate the bounds from, such as "q", "qdot", "qddot", or ["q", "qdot"],
    mapping: BiMapping | BiMappingList
        The mapping to use to generate the bounds. If None, the default mapping is built

    Returns
    -------
    Bounds
        The bounds generated from the ranges of the model
    """
    out = Bounds()

    q_ranges = model.ranges_from_model("q") if "q" in variables else None
    qdot_ranges = model.ranges_from_model("qdot") if "qdot" in variables else None
    qddot_ranges = model.ranges_from_model("qddot") if "qddot" in variables else None

    for var in variables:
        if var == "q":
            q_mapping = _q_mapping(model, mapping)
            mapping = q_mapping
            x_min = [q_ranges[i].min() for i in q_mapping["q"].to_first.map_idx]
            x_max = [q_ranges[i].max() for i in q_mapping["q"].to_first.map_idx]
            out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))
        elif var == "qdot":
            qdot_mapping = _qdot_mapping(model, mapping)
            mapping = qdot_mapping
            x_min = [qdot_ranges[i].min() for i in qdot_mapping["qdot"].to_first.map_idx]
            x_max = [qdot_ranges[i].max() for i in qdot_mapping["qdot"].to_first.map_idx]
            out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))
        elif var == "qddot":
            qddot_mapping = _qddot_mapping(model, mapping)
            mapping = qddot_mapping
            x_min = [qddot_ranges[i].min() for i in qddot_mapping["qddot"].to_first.map_idx]
            x_max = [qddot_ranges[i].max() for i in qddot_mapping["qddot"].to_first.map_idx]
            out.concatenate(Bounds(min_bound=x_min, max_bound=x_max))

    if out.shape[0] == 0:
        raise ValueError(f"Unrecognized variable ({variables}), only 'q', 'qdot' and 'qddot' are allowed")

    return out


class BiorbdModel:
    """
    This class allows to define a biorbd model.
    """

    def __init__(self, bio_model: str | biorbd.Model):
        if not isinstance(bio_model, str) and not isinstance(bio_model, biorbd.Model):
            raise ValueError("The model should be of type 'str' or 'biorbd.Model'")

        self.model = biorbd.Model(bio_model) if isinstance(bio_model, str) else bio_model

    @property
    def path(self) -> str:
        return self.model.path().relativePath().to_string()

    def copy(self):
        return BiorbdModel(self.path)

    def serialize(self) -> tuple[Callable, dict]:
        return BiorbdModel, dict(bio_model=self.path)

    def set_gravity(self, new_gravity) -> None:
        self.model.setGravity(new_gravity)
        return

    @property
    def gravity(self) -> MX:
        return self.model.getGravity().to_mx()

    @property
    def nb_segments(self) -> int:
        return self.model.nbSegment()

    @property
    def nb_tau(self) -> int:
        return self.model.nbGeneralizedTorque()

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
    def segments(self) -> tuple[biorbd.Segment]:
        return self.model.segments()

    def homogeneous_matrices_in_global(self, q, reference_index, inverse=False):
        val = self.model.globalJCS(GeneralizedCoordinates(q), reference_index)
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

    def forward_dynamics(self, q, qdot, tau, external_forces=None, f_contacts=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ForwardDynamics(q_biorbd, qdot_biorbd, tau_biorbd, external_forces, f_contacts).to_mx()

    def constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ForwardDynamicsConstraintsDirect(q_biorbd, qdot_biorbd, tau_biorbd, external_forces).to_mx()

    def inverse_dynamics(self, q, qdot, qddot, external_forces=None, f_contacts=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        qddot_biorbd = GeneralizedAcceleration(qddot)
        return self.model.InverseDynamics(q_biorbd, qdot_biorbd, qddot_biorbd, external_forces, f_contacts).to_mx()

    def contact_forces_from_constrained_forward_dynamics(self, q, qdot, tau, external_forces=None) -> MX:
        if external_forces is not None:
            external_forces = biorbd.to_spatial_vector(external_forces)

        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        tau_biorbd = GeneralizedTorque(tau)
        return self.model.ContactForcesFromForwardDynamicsConstraintsDirect(
            q_biorbd, qdot_biorbd, tau_biorbd, external_forces
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

    def muscle_joint_torque(self, activations, q, qdot) -> MX:
        muscles_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscles_states[k].setActivation(activations[k])
        q_biorbd = GeneralizedCoordinates(q)
        qdot_biorbd = GeneralizedVelocity(qdot)
        return self.model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx()

    def markers(self, q) -> Any | list[MX]:
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
        return self.model.rigidContactAxisIdx(contact_index)

    def marker_velocities(self, q, qdot, reference_index=None) -> MX:
        if reference_index is None:
            return horzcat(
                *[
                    m.to_mx()
                    for m in self.model.markersVelocity(
                        GeneralizedCoordinates(q),
                        GeneralizedVelocity(qdot),
                        True,
                    )
                ]
            )

        else:
            out = MX()
            homogeneous_matrix_transposed = self.homogeneous_matrices_in_global(
                GeneralizedCoordinates(q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersVelocity(GeneralizedCoordinates(q), GeneralizedVelocity(qdot)):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    out = horzcat(out, m.to_mx())

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

    @property
    def nb_dof(self) -> int:
        return self.model.nbDof()

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
        return self.model.KineticEnergy(q, qdot).to_mx() - self.model.PotentialEnergy(q).to_mx()

    def set_dependencies(self, dependent_joint_index: list, independent_joint_index: list):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    @property
    def nb_independent_joints(self):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    @property
    def nb_dependent_joints(self):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def add_holonomic_constraint(
        self,
        constraint: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    @property
    def nb_holonomic_constraints(self):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def holonomic_constraints(self, q: MX):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def holonomic_constraints_jacobian(self, q: MX):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def holonomic_constraints_derivative(self, q: MX, qdot: MX):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def partitioned_mass_matrix(self, q):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def partitioned_non_linear_effect(self, q, qdot, f_ext=None, f_contacts=None):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def partitioned_q(self, q):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def partitioned_qdot(self, qdot):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def partitioned_tau(self, tau):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def partitioned_constrained_jacobian(self, q):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def forward_dynamics_constrained_independent(self, u, udot, tau, external_forces=None, f_contacts=None) -> MX:
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def coupling_matrix(self, q: MX) -> MX:
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def q_from_u_and_v(self, u: MX, v: MX) -> MX:
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def compute_v_from_u(self, u: MX):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def compute_v_from_u_numeric(self, u: DM, v_init=None):
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def partitioned_forward_dynamics(self, u, udot, tau, external_forces=None, f_contacts=None) -> MX:
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def dae_inverse_dynamics(
        self, q, qdot, qddot, tau, lagrange_multipliers, external_forces=None, f_contacts=None
    ) -> MX:
        raise NotImplementedError("This function is not implemented for BiorbdModel use BiorbdModelHolonomic instead.")

    def discrete_lagrangian(
        self,
        q1: MX | SX,
        q2: MX | SX,
        time_step: MX | SX,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
    ) -> MX | SX:
        raise NotImplementedError(
            "This function is not implemented for BiorbdModel use VariationalBiorbdModel instead."
        )

    @staticmethod
    def control_approximation(
        control_minus: MX | SX,
        control_plus: MX | SX,
        time_step: float,
        control_type: ControlType = ControlType.CONSTANT,
        discrete_approximation: QuadratureRule = QuadratureRule.MIDPOINT,
    ):
        raise NotImplementedError(
            "This function is not implemented for BiorbdModel use VariationalBiorbdModel instead."
        )

    @staticmethod
    def compute_holonomic_discrete_constraints_jacobian(
        jac: Function, time_step: MX | SX, q: MX | SX
    ) -> MX | SX | None:
        raise NotImplementedError(
            "This function is not implemented for BiorbdModel use VariationalBiorbdModel instead."
        )

    def discrete_euler_lagrange_equations(
        self,
        time_step: MX | SX,
        q_prev: MX | SX,
        q_cur: MX | SX,
        q_next: MX | SX,
        control_prev: MX | SX,
        control_cur: MX | SX,
        control_next: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas: MX | SX = None,
    ) -> MX | SX:
        raise NotImplementedError(
            "This function is not implemented for BiorbdModel use VariationalBiorbdModel instead."
        )

    def compute_initial_states(
        self,
        time_step: MX | SX,
        q0: MX | SX,
        qdot0: MX | SX,
        q1: MX | SX,
        control0: MX | SX,
        control1: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas0: MX | SX = None,
    ):
        raise NotImplementedError(
            "This function is not implemented for BiorbdModel use VariationalBiorbdModel instead."
        )

    def compute_final_states(
        self,
        time_step: MX | SX,
        q_penultimate: MX | SX,
        q_ultimate: MX | SX,
        q_dot_ultimate: MX | SX,
        control_penultimate: MX | SX,
        control_ultimate: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdasN: MX | SX = None,
    ):
        raise NotImplementedError(
            "This function is not implemented for BiorbdModel use VariationalBiorbdModel instead."
        )


class MultiBiorbdModel:
    """
    This class allows to define multiple biorbd models for the same phase.
    """

    def __init__(self, bio_model: tuple[str | biorbd.Model | BiorbdModel, ...]):
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

    def __getitem__(self, index):
        return self.models[index]

    def deep_copy(self, *args):
        raise NotImplementedError("Deep copy is not implemented yet for MultiBiorbdModel class")

    @property
    def path(self) -> list[str]:
        return [model.path for model in self.models]

    def copy(self):
        return MultiBiorbdModel(tuple(self.path))

    def serialize(self) -> tuple[Callable, dict]:
        return MultiBiorbdModel, dict(bio_model=tuple(self.path))

    def variable_index(self, variable: str, model_index: int) -> range:
        """
        Get the index of the variables in the global vector for a given model index

        Parameters
        ----------
        variable: str
            The variable to get the index from such as 'q', 'qdot', 'qddot', 'tau', 'contact'
        model_index: int
            The index of the model to get the index from

        Returns
        -------
        range
            The index of the variable in the global vector
        """
        if variable == "q":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nb_q
            return range(current_idx, current_idx + self.models[model_index].nb_q)
        elif variable == "qdot":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nb_qdot
            return range(current_idx, current_idx + self.models[model_index].nb_qdot)
        elif variable == "qddot":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nb_qddot
            return range(current_idx, current_idx + self.models[model_index].nb_qddot)
        elif variable == "qddot_joints":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nb_qddot - model.nb_root
            return range(
                current_idx, current_idx + self.models[model_index].nb_qddot - self.models[model_index].nb_root
            )
        elif variable == "qddot_root":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nb_root
            return range(current_idx, current_idx + self.models[model_index].nb_root)
        elif variable == "tau":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nb_tau
            return range(current_idx, current_idx + self.models[model_index].nb_tau)
        elif variable == "contact":
            current_idx = 0
            for model in self.models[:model_index]:
                current_idx += model.nb_rigid_contacts
            return range(current_idx, current_idx + self.models[model_index].nb_rigid_contacts)

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

    def homogeneous_matrices_in_global(self, q, reference_index, inverse=False):
        raise NotImplementedError("homogeneous_matrices_in_global is not implemented for MultiBiorbdModel")

    def homogeneous_matrices_in_child(self, *args) -> tuple:
        raise NotImplementedError("homogeneous_matrices_in_child is not implemented for MultiBiorbdModel")

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
        for model in self.models:
            muscles_states = model.model.stateSet()  # still call from Biorbd
            for k in range(model.nb_muscles):
                muscles_states[k].setActivation(activations[k])
            q_biorbd = GeneralizedCoordinates(q)
            qdot_biorbd = GeneralizedVelocity(qdot)
            out = vertcat(out, model.model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx())
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

    def marker_velocities(self, q, qdot, reference_index=None) -> MX:
        if reference_index is not None:
            raise RuntimeError("marker_velocities is not implemented yet with reference_index for MultiBiorbdModel")

        out = MX()
        for i, model in enumerate(self.models):
            q_model = q[self.variable_index("q", i)]
            qdot_model = qdot[self.variable_index("qdot", i)]
            out = vertcat(
                out,
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

    def _q_mapping(self, mapping: BiMapping = None) -> dict:
        return _q_mapping(self, mapping)

    def _qdot_mapping(self, mapping: BiMapping = None) -> dict:
        return _qdot_mapping(self, mapping)

    def _qddot_mapping(self, mapping: BiMapping = None) -> dict:
        return _qddot_mapping(self, mapping)

    def lagrangian(self):
        raise NotImplementedError("lagrangian is not implemented yet for MultiBiorbdModel")

    def set_dependencies(self, dependent_joint_index: list, independent_joint_index: list):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    @property
    def nb_independent_joints(self):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    @property
    def nb_dependent_joints(self):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def add_holonomic_constraint(
        self,
        constraint: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    @property
    def nb_holonomic_constraints(self):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def holonomic_constraints(self, q: MX):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def holonomic_constraints_jacobian(self, q: MX):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def holonomic_constraints_derivative(self, q: MX, qdot: MX):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def partitioned_mass_matrix(self, q):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def partitioned_non_linear_effect(self, q, qdot, f_ext=None, f_contacts=None):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def partitioned_q(self, q):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def partitioned_qdot(self, qdot):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def partitioned_tau(self, tau):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def partitioned_constrained_jacobian(self, q):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def forward_dynamics_constrained_independent(self, u, udot, tau, external_forces=None, f_contacts=None) -> MX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def coupling_matrix(self, q: MX) -> MX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def q_from_u_and_v(self, u: MX, v: MX) -> MX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def compute_v_from_u(self, u: MX):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def compute_v_from_u_numeric(self, u: DM, v_init=None):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def partitioned_forward_dynamics(self, u, udot, tau, external_forces=None, f_contacts=None) -> MX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def dae_inverse_dynamics(
        self, q, qdot, qddot, tau, lagrange_multipliers, external_forces=None, f_contacts=None
    ) -> MX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use BiorbdModelHolonomic instead."
        )

    def discrete_lagrangian(
        self,
        q1: MX | SX,
        q2: MX | SX,
        time_step: MX | SX,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
    ) -> MX | SX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use VariationalBiorbdModel instead."
        )

    @staticmethod
    def control_approximation(
        control_minus: MX | SX,
        control_plus: MX | SX,
        time_step: float,
        control_type: ControlType = ControlType.CONSTANT,
        discrete_approximation: QuadratureRule = QuadratureRule.MIDPOINT,
    ):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use VariationalBiorbdModel instead."
        )

    @staticmethod
    def compute_holonomic_discrete_constraints_jacobian(
        jac: Function, time_step: MX | SX, q: MX | SX
    ) -> MX | SX | None:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use VariationalBiorbdModel instead."
        )

    def discrete_euler_lagrange_equations(
        self,
        time_step: MX | SX,
        q_prev: MX | SX,
        q_cur: MX | SX,
        q_next: MX | SX,
        control_prev: MX | SX,
        control_cur: MX | SX,
        control_next: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas: MX | SX = None,
    ) -> MX | SX:
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use VariationalBiorbdModel instead."
        )

    def compute_initial_states(
        self,
        time_step: MX | SX,
        q0: MX | SX,
        qdot0: MX | SX,
        q1: MX | SX,
        control0: MX | SX,
        control1: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas0: MX | SX = None,
    ):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use VariationalBiorbdModel instead."
        )

    def compute_final_states(
        self,
        time_step: MX | SX,
        q_penultimate: MX | SX,
        q_ultimate: MX | SX,
        q_dot_ultimate: MX | SX,
        control_penultimate: MX | SX,
        control_ultimate: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdasN: MX | SX = None,
    ):
        raise NotImplementedError(
            "This function is not implemented for MultiBiorbdModel use VariationalBiorbdModel instead."
        )
