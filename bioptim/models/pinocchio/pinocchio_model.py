from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from casadi import DM, MX, Function, horzcat, vertcat

from ..biorbd.external_forces import ExternalForceSetTimeSeries, ExternalForceSetVariables
from ..utils import _var_mapping, bounds_from_ranges, cache_function
from ...limits.path_conditions import Bounds
from ...misc.mapping import BiMapping, BiMappingList
from ...optimization.parameters import ParameterList


@dataclass(frozen=True)
class _Range:
    min_bound: float
    max_bound: float

    def min(self) -> float:
        return self.min_bound

    def max(self) -> float:
        return self.max_bound


def _import_pinocchio():
    try:
        import pinocchio as pin
        import pinocchio.casadi as cpin
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PinocchioModel requires the optional dependency 'pinocchio'. "
            "Install the Pinocchio Python bindings to use this model backend."
        ) from e

    return pin, cpin


class PinocchioModel:
    """
    Pinocchio implementation of the bioptim biomodel protocol.

    The implementation intentionally mirrors the BiorbdModel public surface where Pinocchio exposes equivalent
    rigid-body algorithms. Muscle, ligament, passive torque, rigid contact and biorbd external-force APIs are not
    available yet.
    """

    def __init__(
        self,
        bio_model: str | object,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        marker_names: tuple[str, ...] | list[str] = None,
        root_joint: object = None,
    ):
        if external_force_set is not None:
            raise NotImplementedError("External forces are not implemented yet for PinocchioModel")

        if parameters is not None:
            raise NotImplementedError("Parameter callbacks are not implemented yet for PinocchioModel")

        if isinstance(bio_model, str):
            self.pin, self.cpin = _import_pinocchio()
            self.model = (
                self.pin.buildModelFromUrdf(bio_model, root_joint)
                if root_joint is not None
                else self.pin.buildModelFromUrdf(bio_model)
            )
            self._path = str(Path(bio_model).absolute())
        elif bio_model.__class__.__module__.startswith("pinocchio"):
            self.pin, self.cpin = _import_pinocchio()
            self.model = bio_model
            self._path = ""
        else:
            raise ValueError("The model should be of type 'str' or 'pinocchio.Model'")

        self.casadi_model = self.cpin.Model(self.model)
        self.data = self.casadi_model.createData()
        self._friction_coefficients = friction_coefficients
        self.external_force_set = None
        self.external_forces = MX.sym("external_forces_mx", 0, 1)
        self.parameters = MX()
        self._marker_names = tuple(marker_names) if marker_names is not None else self._default_marker_names()
        self._symbolic_variables()
        self._cached_functions = {}

    def _symbolic_variables(self):
        self.q = MX.sym("q_mx", self.nb_q, 1)
        self.qdot = MX.sym("qdot_mx", self.nb_qdot, 1)
        self.qddot = MX.sym("qddot_mx", self.nb_qddot, 1)
        self.qddot_joints = MX.sym("qddot_joints_mx", self.nb_qddot - self.nb_root, 1)
        self.tau = MX.sym("tau_mx", self.nb_tau, 1)
        self.muscle = MX.sym("muscle_mx", 0, 1)
        self.activations = MX.sym("activations_mx", 0, 1)

    def _default_marker_names(self) -> tuple[str, ...]:
        return tuple(frame.name for frame in self.model.frames[1:])

    @property
    def name(self) -> str:
        return Path(self.path).name if self.path else "pinocchio_model"

    @property
    def path(self) -> str:
        return self._path

    def copy(self):
        if self.path:
            return PinocchioModel(
                self.path,
                friction_coefficients=self.friction_coefficients,
                marker_names=self.marker_names,
            )
        return PinocchioModel(
            self.model,
            friction_coefficients=self.friction_coefficients,
            marker_names=self.marker_names,
        )

    def serialize(self) -> tuple[Callable, dict]:
        bio_model = self.path if self.path else self.model
        return PinocchioModel, dict(bio_model=bio_model, marker_names=self.marker_names)

    @property
    def friction_coefficients(self) -> MX | np.ndarray:
        return self._friction_coefficients

    def set_friction_coefficients(self, new_friction_coefficients) -> None:
        if isinstance(new_friction_coefficients, (DM, np.ndarray)) and np.any(new_friction_coefficients < 0):
            raise ValueError("Friction coefficients must be positive")
        self._friction_coefficients = new_friction_coefficients

    @cache_function
    def gravity(self) -> Function:
        gravity = MX(self.model.gravity.linear)
        return Function("gravity", [self.parameters], [gravity], ["parameters"], ["gravity"])

    def set_gravity(self, new_gravity) -> None:
        self.model.gravity.linear = np.asarray(new_gravity, dtype=float).reshape(3)
        self.casadi_model = self.cpin.Model(self.model)
        self.data = self.casadi_model.createData()
        self._cached_functions = {}

    @property
    def nb_tau(self) -> int:
        return self.model.nv

    @property
    def nb_segments(self) -> int:
        return self.model.njoints - 1

    def segment_index(self, name) -> int:
        if name not in self.model.names:
            raise ValueError(f"{name} is not a segment name")
        return list(self.model.names).index(name)

    @property
    def nb_quaternions(self) -> int:
        return max(self.nb_q - self.nb_qdot, 0)

    @property
    def nb_dof(self) -> int:
        return self.model.nv

    @property
    def nb_q(self) -> int:
        return self.model.nq

    @property
    def nb_qdot(self) -> int:
        return self.model.nv

    @property
    def nb_qddot(self) -> int:
        return self.model.nv

    @property
    def nb_root(self) -> int:
        if self.model.njoints <= 1:
            return 0
        first_joint = self.model.joints[1]
        return first_joint.nv if first_joint.nq == 7 and first_joint.nv == 6 else 0

    @property
    def segments(self) -> tuple:
        return tuple(self.model.joints[1:])

    @staticmethod
    def _unsupported(name: str):
        raise NotImplementedError(f"{name} is not implemented for PinocchioModel")

    @cache_function
    def rotation_matrix_to_euler_angles(self, sequence: str) -> Function:
        self._unsupported("rotation_matrix_to_euler_angles")

    def _frame_placement(self, frame_index: int, q: MX = None):
        data = self.casadi_model.createData()
        self.cpin.framesForwardKinematics(self.casadi_model, data, self.q if q is None else q)
        return data.oMf[frame_index]

    def _frame_translation(self, frame_index: int, q: MX = None) -> MX:
        return self._frame_placement(frame_index, q).translation

    def _frame_homogeneous_matrix(self, frame_index: int, q: MX = None) -> MX:
        placement = self._frame_placement(frame_index, q)
        out = MX.eye(4)
        out[:3, :3] = placement.rotation
        out[:3, 3] = placement.translation
        return out

    @cache_function
    def homogeneous_matrices_in_global(self, segment_index: int, inverse: bool = False) -> Function:
        matrix = self._frame_homogeneous_matrix(segment_index)
        if inverse:
            rotation = matrix[:3, :3]
            translation = matrix[:3, 3]
            matrix = MX.eye(4)
            matrix[:3, :3] = rotation.T
            matrix[:3, 3] = -rotation.T @ translation

        return Function(
            "homogeneous_matrices_in_global",
            [self.q, self.parameters],
            [matrix],
            ["q", "parameters"],
            ["Joint coordinate system RT matrix in global"],
        )

    @cache_function
    def homogeneous_matrices_in_child(self, segment_id) -> Function:
        placement = self.casadi_model.frames[segment_id].placement
        matrix = MX.eye(4)
        matrix[:3, :3] = placement.rotation
        matrix[:3, 3] = placement.translation
        return Function(
            "homogeneous_matrices_in_child",
            [self.parameters],
            [matrix],
            ["parameters"],
            ["Joint coordinate system RT matrix in local"],
        )

    @cache_function
    def mass(self) -> Function:
        model_mass = sum(inertia.mass for inertia in self.model.inertias)
        return Function("mass", [self.parameters], [MX(model_mass)], ["parameters"], ["mass"])

    @cache_function
    def rt(self, rt_index) -> Function:
        return self.homogeneous_matrices_in_global(rt_index)

    @cache_function
    def center_of_mass(self) -> Function:
        data = self.casadi_model.createData()
        com = self.cpin.centerOfMass(self.casadi_model, data, self.q)
        return Function("center_of_mass", [self.q, self.parameters], [com], ["q", "parameters"], ["Center of mass"])

    @cache_function
    def center_of_mass_velocity(self) -> Function:
        data = self.casadi_model.createData()
        jacobian = self.cpin.jacobianCenterOfMass(self.casadi_model, data, self.q)
        com_velocity = jacobian @ self.qdot
        return Function(
            "center_of_mass_velocity",
            [self.q, self.qdot, self.parameters],
            [com_velocity],
            ["q", "qdot", "parameters"],
            ["Center of mass velocity"],
        )

    @cache_function
    def center_of_mass_acceleration(self) -> Function:
        data = self.casadi_model.createData()
        self.cpin.centerOfMass(self.casadi_model, data, self.q, self.qdot, self.qddot)
        com_acceleration = data.acom[0]
        return Function(
            "center_of_mass_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [com_acceleration],
            ["q", "qdot", "qddot", "parameters"],
            ["Center of mass acceleration"],
        )

    @cache_function
    def body_rotation_rate(self) -> Function:
        self._unsupported("body_rotation_rate")

    @cache_function
    def mass_matrix(self) -> Function:
        data = self.casadi_model.createData()
        mass_matrix = self.cpin.crba(self.casadi_model, data, self.q)
        return Function("mass_matrix", [self.q, self.parameters], [mass_matrix], ["q", "parameters"], ["Mass matrix"])

    @cache_function
    def non_linear_effects(self) -> Function:
        data = self.casadi_model.createData()
        effects = self.cpin.nonLinearEffects(self.casadi_model, data, self.q, self.qdot)
        return Function(
            "non_linear_effects",
            [self.q, self.qdot, self.parameters],
            [effects],
            ["q", "qdot", "parameters"],
            ["Non linear effects"],
        )

    @cache_function
    def angular_momentum(self) -> Function:
        data = self.casadi_model.createData()
        self.cpin.computeCentroidalMomentum(self.casadi_model, data, self.q, self.qdot)
        return Function(
            "angular_momentum",
            [self.q, self.qdot, self.parameters],
            [data.hg.angular],
            ["q", "qdot", "parameters"],
            ["Angular momentum"],
        )

    @cache_function
    def reshape_qdot(self, k_stab=1) -> Function:
        return Function(
            "reshape_qdot",
            [self.q, self.qdot, self.parameters],
            [self.qdot],
            ["q", "qdot", "parameters"],
            ["Reshaped qdot"],
        )

    @cache_function
    def segment_angular_velocity(self, idx) -> Function:
        data = self.casadi_model.createData()
        self.cpin.forwardKinematics(self.casadi_model, data, self.q, self.qdot)
        velocity = self.cpin.getVelocity(self.casadi_model, data, idx, self.cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return Function(
            "segment_angular_velocity",
            [self.q, self.qdot, self.parameters],
            [velocity.angular],
            ["q", "qdot", "parameters"],
            ["Segment angular velocity"],
        )

    @cache_function
    def segment_orientation(self, idx: int, sequence: str = "xyz") -> Function:
        self._unsupported("segment_orientation")

    @property
    def name_dof(self) -> tuple[str, ...]:
        names = []
        for joint_id in range(1, self.model.njoints):
            joint_name = self.model.names[joint_id]
            joint_nv = self.model.joints[joint_id].nv
            names.extend([joint_name] if joint_nv == 1 else [f"{joint_name}_{i}" for i in range(joint_nv)])
        return tuple(names)

    @property
    def contact_names(self) -> tuple[str, ...]:
        return ()

    @property
    def nb_soft_contacts(self) -> int:
        return 0

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        return ()

    def soft_contact(self, soft_contact_index, *args):
        self._unsupported("soft_contact")

    @property
    def muscle_names(self) -> tuple[str, ...]:
        return ()

    @property
    def nb_muscles(self) -> int:
        return 0

    @cache_function
    def torque(self) -> Function:
        return Function(
            "torque_activation",
            [self.tau, self.q, self.qdot, self.parameters],
            [self.tau],
            ["tau", "q", "qdot", "parameters"],
            ["Torque from tau activations"],
        )

    @cache_function
    def forward_dynamics_free_floating_base(self) -> Function:
        self._unsupported("forward_dynamics_free_floating_base")

    @staticmethod
    def reorder_qddot_root_joints(qddot_root, qddot_joints) -> MX:
        return vertcat(qddot_root, qddot_joints)

    @cache_function
    def forward_dynamics(self, with_contact: bool = False) -> Function:
        if with_contact:
            self._unsupported("forward_dynamics with contact")
        data = self.casadi_model.createData()
        qddot = self.cpin.aba(self.casadi_model, data, self.q, self.qdot, self.tau)
        return Function(
            "forward_dynamics",
            [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
            [qddot],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["qddot"],
        )

    @cache_function
    def inverse_dynamics(self, with_contact: bool = False) -> Function:
        if with_contact:
            self._unsupported("inverse_dynamics with contact")
        data = self.casadi_model.createData()
        tau = self.cpin.rnea(self.casadi_model, data, self.q, self.qdot, self.qddot)
        return Function(
            "inverse_dynamics",
            [self.q, self.qdot, self.qddot, self.external_forces, self.parameters],
            [tau],
            ["q", "qdot", "qddot", "external_forces", "parameters"],
            ["tau"],
        )

    @cache_function
    def forward_dynamics_derivatives(self) -> Function:
        data = self.casadi_model.createData()
        self.cpin.computeABADerivatives(self.casadi_model, data, self.q, self.qdot, self.tau)
        return Function(
            "forward_dynamics_derivatives",
            [self.q, self.qdot, self.tau, self.parameters],
            [data.ddq_dq, data.ddq_dv, data.Minv],
            ["q", "qdot", "tau", "parameters"],
            ["ddq_dq", "ddq_dqdot", "ddq_dtau"],
        )

    @cache_function
    def inverse_dynamics_derivatives(self) -> Function:
        data = self.casadi_model.createData()
        derivatives = self.cpin.computeRNEADerivatives(self.casadi_model, data, self.q, self.qdot, self.qddot)
        if derivatives is not None:
            dtau_dq, dtau_dv, dtau_da = derivatives
        else:
            dtau_dq, dtau_dv, dtau_da = data.dtau_dq, data.dtau_dv, data.M
        return Function(
            "inverse_dynamics_derivatives",
            [self.q, self.qdot, self.qddot, self.parameters],
            [dtau_dq, dtau_dv, dtau_da],
            ["q", "qdot", "qddot", "parameters"],
            ["dtau_dq", "dtau_dqdot", "dtau_dqddot"],
        )

    @cache_function
    def contact_forces_from_constrained_forward_dynamics(self) -> Function:
        self._unsupported("contact_forces_from_constrained_forward_dynamics")

    @cache_function
    def rigid_contact_position(self, index: int) -> Function:
        self._unsupported("rigid_contact_position")

    @cache_function
    def forces_on_each_rigid_contact_point(self) -> Function:
        self._unsupported("forces_on_each_rigid_contact_point")

    @cache_function
    def qdot_from_impact(self) -> Function:
        self._unsupported("qdot_from_impact")

    @cache_function
    def muscle_activation_dot(self) -> Function:
        self._unsupported("muscle_activation_dot")

    @cache_function
    def muscle_length_jacobian(self) -> Function:
        self._unsupported("muscle_length_jacobian")

    @cache_function
    def muscle_velocity(self) -> Function:
        self._unsupported("muscle_velocity")

    @cache_function
    def muscle_joint_torque(self) -> Function:
        self._unsupported("muscle_joint_torque")

    @property
    def marker_names(self) -> tuple[str, ...]:
        return self._marker_names

    @property
    def nb_markers(self) -> int:
        return len(self.marker_names)

    def marker_index(self, name):
        if name not in self.marker_names:
            raise ValueError(f"{name} is not a marker name")
        return self.marker_names.index(name)

    def _marker_frame_index(self, index: int) -> int:
        return self.model.getFrameId(self.marker_names[index])

    def contact_index(self, name):
        self._unsupported("contact_index")

    @cache_function
    def markers(self) -> Function:
        markers = horzcat(*[self._frame_translation(self._marker_frame_index(i)) for i in range(self.nb_markers)])
        return Function("markers", [self.q, self.parameters], [markers], ["q", "parameters"], ["markers"])

    @cache_function
    def marker(self, index: int, reference_segment_index: int = None) -> Function:
        position = self._frame_translation(self._marker_frame_index(index))
        if reference_segment_index is not None:
            reference_matrix = self._frame_homogeneous_matrix(reference_segment_index)
            position = reference_matrix[:3, :3].T @ (position - reference_matrix[:3, 3])
        return Function("marker", [self.q, self.parameters], [position], ["q", "parameters"], ["marker"])

    @property
    def nb_rigid_contacts(self) -> int:
        return 0

    @property
    def nb_contacts(self) -> int:
        return 0

    def rigid_contact_index(self, contact_index) -> tuple:
        self._unsupported("rigid_contact_index")

    @cache_function
    def markers_velocities(self, reference_index=None) -> Function:
        velocities = horzcat(
            *[self.marker_velocity(i)(self.q, self.qdot, self.parameters) for i in range(self.nb_markers)]
        )
        return Function(
            "markers_velocities",
            [self.q, self.qdot, self.parameters],
            [velocities],
            ["q", "qdot", "parameters"],
            ["markers_velocities"],
        )

    @cache_function
    def marker_velocity(self, marker_index: int) -> Function:
        data = self.casadi_model.createData()
        self.cpin.forwardKinematics(self.casadi_model, data, self.q, self.qdot)
        self.cpin.updateFramePlacements(self.casadi_model, data)
        velocity = self.cpin.getFrameVelocity(
            self.casadi_model,
            data,
            self._marker_frame_index(marker_index),
            self.cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        return Function(
            "marker_velocity",
            [self.q, self.qdot, self.parameters],
            [velocity],
            ["q", "qdot", "parameters"],
            ["marker_velocity"],
        )

    @cache_function
    def markers_accelerations(self, reference_index=None) -> Function:
        accelerations = horzcat(
            *[
                self.marker_acceleration(i)(self.q, self.qdot, self.qddot, self.parameters)
                for i in range(self.nb_markers)
            ]
        )
        return Function(
            "markers_accelerations",
            [self.q, self.qdot, self.qddot, self.parameters],
            [accelerations],
            ["q", "qdot", "qddot", "parameters"],
            ["markers_accelerations"],
        )

    @cache_function
    def marker_acceleration(self, marker_index: int) -> Function:
        data = self.casadi_model.createData()
        self.cpin.forwardKinematics(self.casadi_model, data, self.q, self.qdot, self.qddot)
        self.cpin.updateFramePlacements(self.casadi_model, data)
        acceleration = self.cpin.getFrameAcceleration(
            self.casadi_model,
            data,
            self._marker_frame_index(marker_index),
            self.cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        return Function(
            "marker_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [acceleration],
            ["q", "qdot", "qddot", "parameters"],
            ["marker_acceleration"],
        )

    @cache_function
    def tau_max(self) -> Function:
        return Function(
            "tau_max",
            [self.q, self.qdot, self.parameters],
            [MX(self.model.effortLimit), -MX(self.model.effortLimit)],
            ["q", "qdot", "parameters"],
            ["tau_max", "tau_min"],
        )

    @cache_function
    def rigid_contact_acceleration(self, contact_index, contact_axis) -> Function:
        self._unsupported("rigid_contact_acceleration")

    @cache_function
    def markers_jacobian(self) -> Function:
        jacobians = []
        for marker_index in range(self.nb_markers):
            data = self.casadi_model.createData()
            try:
                jacobian = self.cpin.computeFrameJacobian(
                    self.casadi_model,
                    data,
                    self.q,
                    self._marker_frame_index(marker_index),
                    self.cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
            except TypeError:
                jacobian = MX.zeros(6, self.nb_qdot)
                self.cpin.computeFrameJacobian(
                    self.casadi_model,
                    data,
                    self.q,
                    self._marker_frame_index(marker_index),
                    self.cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    jacobian,
                )
            jacobians.append(jacobian[:3, :])
        return Function(
            "markers_jacobian",
            [self.q, self.parameters],
            jacobians,
            ["q", "parameters"],
            ["markers_jacobian"],
        )

    @cache_function
    def soft_contact_forces(self) -> Function:
        self._unsupported("soft_contact_forces")

    @cache_function
    def normalize_state_quaternions(self) -> Function:
        normalized_q = self.cpin.normalize(self.casadi_model, self.q) if self.nb_quaternions else self.q
        return Function("normalize_state_quaternions", [self.q], [normalized_q], ["q"], ["q_normalized"])

    def get_quaternion_idx(self) -> list[list[int]]:
        if not self.nb_quaternions:
            return []
        self._unsupported("get_quaternion_idx")

    @cache_function
    def rigid_contact_forces(self) -> Function:
        self._unsupported("rigid_contact_forces")

    @cache_function
    def passive_joint_torque(self) -> Function:
        self._unsupported("passive_joint_torque")

    @cache_function
    def ligament_joint_torque(self) -> Function:
        self._unsupported("ligament_joint_torque")

    def ranges_from_model(self, variable: str):
        if variable in ("q", "q_roots", "q_joints"):
            lower = self.model.lowerPositionLimit
            upper = self.model.upperPositionLimit
        elif variable in ("qdot", "qdot_roots", "qdot_joints"):
            lower = -self.model.velocityLimit
            upper = self.model.velocityLimit
        elif variable in ("qddot", "qddot_joints"):
            lower = np.full(self.nb_qddot, -np.inf)
            upper = np.full(self.nb_qddot, np.inf)
        else:
            raise RuntimeError("Wrong variable name")

        if "_joints" in variable and self.nb_root:
            lower = lower[self.nb_root :]
            upper = upper[self.nb_root :]
        elif "_roots" in variable:
            lower = lower[: self.nb_root]
            upper = upper[: self.nb_root]

        return [_Range(float(min_bound), float(max_bound)) for min_bound, max_bound in zip(lower, upper)]

    def _var_mapping(
        self,
        key: str,
        range_for_mapping: int | list | tuple | range,
        mapping: BiMapping = None,
    ) -> dict:
        return _var_mapping(key, range_for_mapping, mapping)

    def bounds_from_ranges(self, variables: str | list[str], mapping: BiMapping | BiMappingList = None) -> Bounds:
        return bounds_from_ranges(self, variables, mapping)

    @cache_function
    def lagrangian(self) -> Function:
        data = self.casadi_model.createData()
        kinetic = self.cpin.computeKineticEnergy(self.casadi_model, data, self.q, self.qdot)
        potential = self.cpin.computePotentialEnergy(self.casadi_model, data, self.q)
        return Function("lagrangian", [self.q, self.qdot], [kinetic - potential], ["q", "qdot"], ["lagrangian"])

    def partitioned_forward_dynamics(self):
        self._unsupported("partitioned_forward_dynamics")

    @staticmethod
    def animate(*args, **kwargs):
        raise NotImplementedError("Animation is not implemented for PinocchioModel")
