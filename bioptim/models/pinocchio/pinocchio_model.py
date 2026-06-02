from typing import Callable

from casadi import SX, Function, horzcat, vertcat
import numpy as np

from ..utils import _var_mapping, bounds_from_ranges, cache_function, Range
from ...limits.path_conditions import Bounds
from ...misc.enums import ContactType
from ...misc.mapping import BiMapping, BiMappingList
from ...optimization.parameters import Parameter, ParameterList


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
        parameters: ParameterList = None,
        **kwargs,
    ):
        super().__init__(**kwargs)  # For multiple inheritance compatibility
        self._pinocchio_module, self._casadi_pinocchio_module = _import_pinocchio()

        if isinstance(bio_model, str):
            self._model = self._pinocchio_module.buildModelFromUrdf(bio_model)
        elif isinstance(bio_model, self._pinocchio_module.Model):
            self._model = bio_model
        else:
            raise ValueError("The model should be of type 'str' or 'pinocchio.Model'")

        self._casadi_model = self._casadi_pinocchio_module.Model(self._model)
        self._data = self._casadi_model.createData()

        if parameters is not None:
            for param_key in parameters:
                parameters[param_key].apply_parameter(self)
        self.parameters = parameters.sx if parameters else SX()

        self._marker_names = self._extract_marker_names()
        self._symbolic_variables()
        self._cached_functions = {}

    def _symbolic_variables(self):
        self.q = SX.sym("q_sx", self.nb_q, 1)
        self.qdot = SX.sym("qdot_sx", self.nb_qdot, 1)
        self.qddot = SX.sym("qddot_sx", self.nb_qddot, 1)
        self.qddot_joints = SX.sym("qddot_joints_sx", self.nb_qddot - self.nb_root, 1)
        self.tau = SX.sym("tau_sx", self.nb_tau, 1)
        self.external_forces = SX.sym("external_forces_sx", 0, 1)
        self.muscle = SX.sym("muscle_sx", 0, 1)
        self.activations = SX.sym("activations_sx", 0, 1)

    def _extract_marker_names(self) -> tuple[str, ...]:
        return tuple(
            frame.name for frame in self._model.frames[1:] if frame.type == self._pinocchio_module.FrameType.FIXED_JOINT
        )

    @property
    def name(self) -> str:
        return self._model.name

    def copy(self):
        if self.path:
            return PinocchioModel(
                self.path,
                parameters=self.parameters,
            )
        return PinocchioModel(
            self._model,
            parameters=self.parameters,
        )

    def serialize(self) -> tuple[Callable, dict]:
        bio_model = self.path if self.path else self._model
        return PinocchioModel, dict(bio_model=bio_model, parameters=self.parameters)

    @cache_function
    def gravity(self) -> Function:
        gravity = SX(self._model.gravity.linear)
        return Function("gravity", [self.parameters], [gravity], ["parameters"], ["gravity"])

    def set_gravity(self, new_gravity: Parameter | SX | np.ndarray) -> None:
        new_value = new_gravity.sx if isinstance(new_gravity, Parameter) else new_gravity
        self._model.gravity.linear = np.asarray(new_value, dtype=float).reshape(3)

    @property
    def nb_tau(self) -> int:
        return self._model.nv

    @property
    def nb_segments(self) -> int:
        return self._model.njoints - 1

    def segment_index(self, name) -> int:
        if name not in self._model.names:
            raise ValueError(f"{name} is not a segment name")
        return list(self._model.names).index(name)

    @property
    def nb_quaternions(self) -> int:
        return max(self.nb_q - self.nb_qdot, 0)

    @property
    def nb_dof(self) -> int:
        return self._model.nv

    @property
    def name_dofs(self) -> tuple[str, ...]:
        names = []
        for joint_id in range(1, self._model.njoints):
            joint_name = self._model.names[joint_id]
            joint_nv = self._model.joints[joint_id].nv
            names.extend([joint_name] if joint_nv == 1 else [f"{joint_name}_{i}" for i in range(joint_nv)])
        return tuple(names)

    @property
    def nb_q(self) -> int:
        return self._model.nq

    @property
    def nb_qdot(self) -> int:
        return self._model.nv

    @property
    def nb_qddot(self) -> int:
        return self._model.nv

    @property
    def nb_root(self) -> int:
        if self._model.njoints <= 1:
            return 0
        first_joint = self._model.joints[1]
        return first_joint.nv if first_joint.nq == 7 and first_joint.nv == 6 else 0

    @property
    def segments(self) -> tuple:
        return tuple(self._model.joints[1:])

    def _frame_placement(self, frame_index: int, q: SX = None):
        self._casadi_pinocchio_module.framesForwardKinematics(
            self._casadi_model, self._data, self.q if q is None else q
        )
        return self._data.oMf[frame_index]

    def _frame_translation(self, frame_index: int, q: SX = None) -> SX:
        return self._frame_placement(frame_index, q).translation

    def _frame_homogeneous_matrix(self, frame_index: int, q: SX = None) -> SX:
        placement = self._frame_placement(frame_index, q)
        out = SX.eye(4)
        out[:3, :3] = placement.rotation
        out[:3, 3] = placement.translation
        return out

    @cache_function
    def homogeneous_matrices_in_global(self, segment_index: int, inverse: bool = False) -> Function:
        matrix = self._frame_homogeneous_matrix(segment_index)
        if inverse:
            rotation = matrix[:3, :3]
            translation = matrix[:3, 3]
            matrix = SX.eye(4)
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
        placement = self._casadi_model.frames[segment_id].placement
        matrix = SX.eye(4)
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
        model_mass = sum(inertia.mass for inertia in self._model.inertias)
        return Function("mass", [self.parameters], [SX(model_mass)], ["parameters"], ["mass"])

    @cache_function
    def rt(self, rt_index) -> Function:
        return self.homogeneous_matrices_in_global(rt_index)

    @cache_function
    def center_of_mass(self) -> Function:
        com = self._casadi_pinocchio_module.centerOfMass(self._casadi_model, self._data, self.q)
        return Function("center_of_mass", [self.q, self.parameters], [com], ["q", "parameters"], ["Center of mass"])

    @cache_function
    def center_of_mass_velocity(self) -> Function:
        jacobian = self._casadi_pinocchio_module.jacobianCenterOfMass(self._casadi_model, self._data, self.q)
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
        self._casadi_pinocchio_module.centerOfMass(self._casadi_model, self._data, self.q, self.qdot, self.qddot)
        com_acceleration = self._data.acom[0]
        return Function(
            "center_of_mass_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [com_acceleration],
            ["q", "qdot", "qddot", "parameters"],
            ["Center of mass acceleration"],
        )

    @cache_function
    def mass_matrix(self) -> Function:
        mass_matrix = self._casadi_pinocchio_module.crba(self._casadi_model, self._data, self.q)
        return Function("mass_matrix", [self.q, self.parameters], [mass_matrix], ["q", "parameters"], ["Mass matrix"])

    @cache_function
    def non_linear_effects(self) -> Function:
        effects = self._casadi_pinocchio_module.nonLinearEffects(self._casadi_model, self._data, self.q, self.qdot)
        return Function(
            "non_linear_effects",
            [self.q, self.qdot, self.parameters],
            [effects],
            ["q", "qdot", "parameters"],
            ["Non linear effects"],
        )

    @cache_function
    def angular_momentum(self) -> Function:
        self._casadi_pinocchio_module.computeCentroidalMomentum(self._casadi_model, self._data, self.q, self.qdot)
        return Function(
            "angular_momentum",
            [self.q, self.qdot, self.parameters],
            [self._data.hg.angular],
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
        self._casadi_pinocchio_module.forwardKinematics(self._casadi_model, self._data, self.q, self.qdot)
        velocity = self._casadi_pinocchio_module.getVelocity(
            self._casadi_model, self._data, idx, self._casadi_pinocchio_module.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return Function(
            "segment_angular_velocity",
            [self.q, self.qdot, self.parameters],
            [velocity.angular],
            ["q", "qdot", "parameters"],
            ["Segment angular velocity"],
        )

    @property
    def friction_coefficients(self) -> SX | np.ndarray:
        return None

    @property
    def nb_passive_joint_torques(self) -> int:
        return 0

    @property
    def contact_types(self) -> list[ContactType]:
        return []

    @property
    def nb_soft_contacts(self) -> int:
        return 0

    @property
    def nb_muscles(self) -> int:
        return 0

    @property
    def nb_ligaments(self) -> int:
        return 0

    @staticmethod
    def reorder_qddot_root_joints(qddot_root: SX, qddot_joints: SX) -> SX:
        return vertcat(qddot_root, qddot_joints)

    @cache_function
    def forward_dynamics(self, with_contact: bool = False) -> Function:
        if with_contact:
            raise NotImplementedError("forward_dynamics with contact is not implemented yet for PinocchioModel")

        qddot = self._casadi_pinocchio_module.aba(self._casadi_model, self._data, self.q, self.qdot, self.tau)
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
            raise NotImplementedError("inverse_dynamics with contact is not implemented yet for PinocchioModel")

        tau = self._casadi_pinocchio_module.rnea(self._casadi_model, self._data, self.q, self.qdot, self.qddot)
        return Function(
            "inverse_dynamics",
            [self.q, self.qdot, self.qddot, self.external_forces, self.parameters],
            [tau],
            ["q", "qdot", "qddot", "external_forces", "parameters"],
            ["tau"],
        )

    @cache_function
    def forward_dynamics_derivatives(self) -> Function:
        self._casadi_pinocchio_module.computeABADerivatives(self._casadi_model, self._data, self.q, self.qdot, self.tau)
        return Function(
            "forward_dynamics_derivatives",
            [self.q, self.qdot, self.tau, self.parameters],
            [self._data.ddq_dq, self._data.ddq_dv, self._data.Minv],
            ["q", "qdot", "tau", "parameters"],
            ["ddq_dq", "ddq_dqdot", "ddq_dtau"],
        )

    @cache_function
    def inverse_dynamics_derivatives(self) -> Function:
        derivatives = self._casadi_pinocchio_module.computeRNEADerivatives(
            self._casadi_model, self._data, self.q, self.qdot, self.qddot
        )
        if derivatives is not None:
            dtau_dq, dtau_dv, dtau_da = derivatives
        else:
            dtau_dq, dtau_dv, dtau_da = self._data.dtau_dq, self._data.dtau_dv, self._data.M
        return Function(
            "inverse_dynamics_derivatives",
            [self.q, self.qdot, self.qddot, self.parameters],
            [dtau_dq, dtau_dv, dtau_da],
            ["q", "qdot", "qddot", "parameters"],
            ["dtau_dq", "dtau_dqdot", "dtau_dqddot"],
        )

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
        return self._model.getFrameId(self.marker_names[index])

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
        self._casadi_pinocchio_module.forwardKinematics(self._casadi_model, self._data, self.q, self.qdot)
        self._casadi_pinocchio_module.updateFramePlacements(self._casadi_model, self._data)
        velocity = self._casadi_pinocchio_module.getFrameVelocity(
            self._casadi_model,
            self._data,
            self._marker_frame_index(marker_index),
            self._casadi_pinocchio_module.ReferenceFrame.LOCAL_WORLD_ALIGNED,
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
        self._casadi_pinocchio_module.forwardKinematics(self._casadi_model, self._data, self.q, self.qdot, self.qddot)
        self._casadi_pinocchio_module.updateFramePlacements(self._casadi_model, self._data)
        acceleration = self._casadi_pinocchio_module.getFrameAcceleration(
            self._casadi_model,
            self._data,
            self._marker_frame_index(marker_index),
            self._casadi_pinocchio_module.ReferenceFrame.LOCAL_WORLD_ALIGNED,
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
            [SX(self._model.effortLimit), -SX(self._model.effortLimit)],
            ["q", "qdot", "parameters"],
            ["tau_max", "tau_min"],
        )

    @cache_function
    def markers_jacobian(self) -> Function:
        jacobians = []
        for marker_index in range(self.nb_markers):
            try:
                jacobian = self._casadi_pinocchio_module.computeFrameJacobian(
                    self._casadi_model,
                    self._data,
                    self.q,
                    self._marker_frame_index(marker_index),
                    self._casadi_pinocchio_module.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
            except TypeError:
                jacobian = SX.zeros(6, self.nb_qdot)
                self._casadi_pinocchio_module.computeFrameJacobian(
                    self._casadi_model,
                    self._data,
                    self.q,
                    self._marker_frame_index(marker_index),
                    self._casadi_pinocchio_module.ReferenceFrame.LOCAL_WORLD_ALIGNED,
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

    def ranges_from_model(self, variable: str):
        if variable in ("q", "q_roots", "q_joints"):
            lower = self._model.lowerPositionLimit
            upper = self._model.upperPositionLimit
        elif variable in ("qdot", "qdot_roots", "qdot_joints"):
            lower = -self._model.velocityLimit
            upper = self._model.velocityLimit
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

        return [Range(float(min_bound), float(max_bound)) for min_bound, max_bound in zip(lower, upper)]

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
        kinetic = self._casadi_pinocchio_module.computeKineticEnergy(self._casadi_model, self._data, self.q, self.qdot)
        potential = self._casadi_pinocchio_module.computePotentialEnergy(self._casadi_model, self._data, self.q)
        return Function("lagrangian", [self.q, self.qdot], [kinetic - potential], ["q", "qdot"], ["lagrangian"])

    @staticmethod
    def animate(*args, **kwargs):
        raise NotImplementedError("Animation is not implemented for PinocchioModel")
