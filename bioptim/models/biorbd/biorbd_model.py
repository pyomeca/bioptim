import biorbd_casadi as biorbd
import numpy as np
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function
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

        # Declaration of MX variables of the right shape for the creation of CasADi Functions
        self.q = MX.sym("q_mx", self.nb_q, 1)
        self.qdot = MX.sym("qdot_mx", self.nb_qdot, 1)
        self.qddot = MX.sym("qddot_mx", self.nb_qddot, 1)
        self.qddot_joints = MX.sym("qddot_joints_mx", self.nb_qddot - self.nb_root, 1)
        self.tau = MX.sym("tau_mx", self.nb_tau, 1)
        self.muscle = MX.sym("muscle_mx", self.nb_muscles, 1)
        self.external_forces = MX.sym("external_forces_mx", 9, len(segments_to_apply_external_forces))

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
    def friction_coefficients(self) -> MX | SX | np.ndarray:
        return self._friction_coefficients

    def set_friction_coefficients(self, new_friction_coefficients) -> None:
        if np.any(new_friction_coefficients < 0):
            raise ValueError("Friction coefficients must be positive")
        return self._friction_coefficients

    @property
    def gravity(self) -> Function:
        """
        Returns the gravity of the model.
        Since the gravity is self-defined in the model, you need to provide the type of the output when calling the function like this:
        model.gravity()(MX() / SX())
        """
        biorbd_return = self.model.getGravity().to_mx()
        casadi_fun = Function(
            "gravity",
            [MX()],
            [biorbd_return],
        )
        return casadi_fun

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

    def rotation_matrix_to_euler_angles(self, sequence) -> Function:
        """
        Returns the rotation matrix to euler angles function.
        """
        r = MX.sym("r_mx", 3, 3)
        # @Pariterre: is this the right order?
        r_matrix = biorbd.Rotation(r[0, 0], r[0, 1], r[0, 2],
                                    r[0, 0], r[0, 1], r[0, 2],
                                    r[0, 0], r[0, 1], r[0, 2])
        biorbd_return = biorbd.Rotation.toEulerAngles(r_matrix, sequence).to_mx()
        casadi_fun = Function(
            "rotation_matrix_to_euler_angles",
            [r],
            [biorbd_return],
        )
        return casadi_fun

    def biorbd_homogeneous_matrices_in_global(self, q, segment_idx, inverse=False) -> tuple:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        TODO: Charbie fix this with ApplyRT wrapper
        """
        rt_matrix = self.model.globalJCS(GeneralizedCoordinates(q), segment_idx)
        return rt_matrix.transpose() if inverse else rt_matrix

    def homogeneous_matrices_in_global(self, segment_idx, inverse=False) -> Function:
        """
        Returns the roto-translation matrix of the segment in the global reference frame.
        """
        biorbd_return = self.biorbd_homogeneous_matrices_in_global(self.q, segment_idx, inverse).to_mx()
        casadi_fun = Function(
            "homogeneous_matrices_in_global",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

    def homogeneous_matrices_in_child(self, segment_id) -> Function:
        """
        Returns the roto-translation matrix of the segment in the child reference frame.
        Since the homogeneous matrix is self-defined in the model, you need to provide the type of the output when calling the function like this:
        model.homogeneous_matrices_in_child(segment_id)(MX() / SX())
        """
        biorbd_return = self.model.localJCS(segment_id).to_mx()
        casadi_fun = Function(
            "homogeneous_matrices_in_child",
            [MX()],
            [biorbd_return],
        )
        return casadi_fun

    @property
    def mass(self) -> Function:
        """
        Returns the mass of the model.
        Since the mass is self-defined in the model, you need to provide the type of the output when calling the function like this:
        model.mass()(MX() / SX())
        """
        biorbd_return = self.model.mass().to_mx()
        casadi_fun = Function(
            "mass",
            [MX()],
            [biorbd_return],
        )
        return casadi_fun

    def rt(self, rt_index) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.RT(q_biorbd, rt_index).to_mx()
        casadi_fun = Function(
            "rt",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

    def center_of_mass(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.CoM(q_biorbd, True).to_mx()
        casadi_fun = Function(
            "center_of_mass",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

    def center_of_mass_velocity(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.CoMdot(q_biorbd, qdot_biorbd, True).to_mx()
        casadi_fun = Function(
            "center_of_mass_velocity",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def center_of_mass_acceleration(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_biorbd = GeneralizedAcceleration(self.qddot)
        biorbd_return = self.model.CoMddot(q_biorbd, qdot_biorbd, qddot_biorbd, True).to_mx()
        casadi_fun = Function(
            "center_of_mass_acceleration",
            [self.q, self.qdot, self.tau],
            [biorbd_return],
        )
        return casadi_fun

    def body_rotation_rate(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.bodyAngularVelocity(q_biorbd, qdot_biorbd, True).to_mx()
        casadi_fun = Function(
            "body_rotation_rate",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def mass_matrix(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.massMatrix(q_biorbd).to_mx()
        casadi_fun = Function(
            "mass_matrix",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

    def non_linear_effects(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.NonLinearEffect(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "non_linear_effects",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def angular_momentum(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.angularMomentum(q_biorbd, qdot_biorbd, True).to_mx()
        casadi_fun = Function(
            "angular_momentum",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def reshape_qdot(self, k_stab=1) -> Function:
        biorbd_return = self.model.computeQdot(
            GeneralizedCoordinates(self.q),
            GeneralizedCoordinates(self.qdot),  # mistake in biorbd
            k_stab,
        ).to_mx()
        casadi_fun = Function(
            "reshape_qdot",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def segment_angular_velocity(self, idx) -> Function:
        """
        Returns the angular velocity of the segment in the global reference frame.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.segmentAngularVelocity(q_biorbd, qdot_biorbd, idx, True).to_mx()
        casadi_fun = Function(
            "segment_angular_velocity",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def segment_orientation(self, idx) -> Function:
        """
        Returns the angular position of the segment in the global reference frame.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        rotation_matrix = self.homogeneous_matrices_in_global(q_biorbd, idx)[:3, :3]
        biorbd_return = biorbd.Rotation.toEulerAngles(
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
        casadi_fun = Function(
            "segment_orientation",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

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

    def torque(self) -> Function:
        """
        Returns the torque from the torque_activations.
        Note that tau_activation should be between 0 and 1.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        tau_activations_biorbd = self.tau  # TODO: Charbie check this
        biorbd_return = self.model.torque(tau_activations_biorbd, q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "torque_activation",
            [self.tau, self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def forward_dynamics_free_floating_base(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_joints_biorbd = GeneralizedAcceleration(self.qddot_joints)
        biorbd_return = self.model.ForwardDynamicsFreeFloatingBase(q_biorbd, qdot_biorbd, qddot_joints_biorbd).to_mx()
        casadi_fun = Function(
            "forward_dynamics_free_floating_base",
            [self.q, self.qdot, self.qddot_joints],
            [biorbd_return],
        )
        return casadi_fun

    @staticmethod
    def reorder_qddot_root_joints(qddot_root, qddot_joints) -> MX | SX:
        return vertcat(qddot_root, qddot_joints)

    def _dispatch_forces(self, external_forces: MX):

        if external_forces is not None:
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

        external_forces_set = self.model.externalForceSet()

        if external_forces is not None:
            for i_element in range(external_forces.shape[1]):
                name = self._segments_to_apply_external_forces[i_element]
                values = external_forces[:6, i_element]
                point_of_application = external_forces[6:9, i_element]
                external_forces_set.add(name, values, point_of_application)

        return external_forces_set

    def forward_dynamics(self, with_contact: bool=False) -> Function:
        external_forces_set = self._dispatch_forces(self.external_forces)

        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        tau_biorbd = GeneralizedTorque(self.tau)

        if with_contact:
            biorbd_return = self.model.ForwardDynamicsConstraintsDirect(
                q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set
            ).to_mx()
            casadi_fun = Function(
                "constrained_forward_dynamics",
                [self.q, self.qdot, self.tau, self.external_forces],
                [biorbd_return],
            )
        else:
            biorbd_return = self.model.ForwardDynamics(q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set).to_mx()
            casadi_fun = Function(
                "forward_dynamics",
                [self.q, self.qdot, self.tau, self.external_forces],
                [biorbd_return],
            )
        return casadi_fun

    def inverse_dynamics(self, with_contact: bool=False) -> Function:
        # @ipuch: I do not understand what is happening here? Do we have f_ext or it is just the contact forces?
        if with_contact:
            f_ext = self.reshape_fext_to_fcontact(self.external_forces)
        else:
            f_ext = self.external_forces

        external_forces_set = self._dispatch_forces(f_ext)

        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_biorbd = GeneralizedAcceleration(self.qddot)
        biorbd_return = self.model.InverseDynamics(q_biorbd, qdot_biorbd, qddot_biorbd, external_forces_set).to_mx()
        casadi_fun = Function(
            "inverse_dynamics",
            [self.q, self.qdot, self.qddot, self.external_forces],
            [biorbd_return],
        )
        return casadi_fun

    def contact_forces_from_constrained_forward_dynamics(self) -> Function:
        external_forces_set = self._dispatch_forces(self.external_forces)

        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        tau_biorbd = GeneralizedTorque(self.tau)
        biorbd_return = self.model.ContactForcesFromForwardDynamicsConstraintsDirect(
            q_biorbd, qdot_biorbd, tau_biorbd, external_forces_set
        ).to_mx()
        casadi_fun = Function(
            "contact_forces_from_constrained_forward_dynamics",
            [self.q, self.qdot, self.tau, self.external_forces],
            [biorbd_return],
        )
        return casadi_fun

    def qdot_from_impact(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_pre_impact_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.ComputeConstraintImpulsesDirect(q_biorbd, qdot_pre_impact_biorbd).to_mx()
        casadi_fun = Function(
            "qdot_from_impact",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def muscle_activation_dot(self) -> Function:
        muscle_excitation_biorbd = self.muscle
        muscle_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscle_states[k].setExcitation(muscle_excitation_biorbd[k])
        biorbd_return = self.model.activationDot(muscle_states).to_mx()
        casadi_fun = Function(
            "muscle_activation_dot",
            [self.muscle],
            [biorbd_return],
        )
        return casadi_fun

    def muscle_length_jacobian(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.musclesLengthJacobian(q_biorbd).to_mx()
        casadi_fun = Function(
            "muscle_length_jacobian",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

    def muscle_velocity(self) -> Function:
        J = self.muscle_length_jacobian()(self.q)
        biorbd_return = J @ self.qdot
        casadi_fun = Function(
            "muscle_velocity",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def muscle_joint_torque(self) -> Function:
        muscles_states = self.model.stateSet()
        muscles_activations = self.muscle
        for k in range(self.model.nbMuscles()):
            muscles_states[k].setActivation(muscles_activations[k])
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.muscularJointTorque(muscles_states, q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "muscle_joint_torque",
            [self.muscle, self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def markers(self) -> list[MX]:
        biorbd_return = [m.to_mx() for m in self.model.markers(GeneralizedCoordinates(self.q))]
        casadi_fun = Function(
            "markers",
            [self.q],
            biorbd_return,
        )
        return casadi_fun

    @property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def marker(self, index, reference_segment_index=None) -> Function:
        marker = self.model.marker(GeneralizedCoordinates(self.q), index)
        if reference_segment_index is not None:
            global_homogeneous_matrix = self.model.globalJCS(GeneralizedCoordinates(self.q), reference_segment_index)
            marker.applyRT(global_homogeneous_matrix.transpose())
        biorbd_return = marker.to_mx()
        casadi_fun = Function(
            "marker",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

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

    def marker_velocities(self, reference_index=None) -> list[MX]:
        if reference_index is None:
            biorbd_return = [
                m.to_mx()
                for m in self.model.markersVelocity(
                    GeneralizedCoordinates(self.q),
                    GeneralizedVelocity(self.qdot),
                    True,
                )
            ]

        else:
            biorbd_return = []
            homogeneous_matrix_transposed = self.biorbd_homogeneous_matrices_in_global(
                GeneralizedCoordinates(self.q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersVelocity(GeneralizedCoordinates(self.q), GeneralizedVelocity(self.qdot)):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    biorbd_return.append(m.to_mx())

        casadi_fun = Function(
            "marker_velocities",
            [self.q, self.qdot],
            biorbd_return,
        )
        return casadi_fun

    def marker_accelerations(self, reference_index=None) -> list[MX]:
        if reference_index is None:
            biorbd_return = [
                m.to_mx()
                for m in self.model.markerAcceleration(
                    GeneralizedCoordinates(self.q),
                    GeneralizedVelocity(self.qdot),
                    GeneralizedAcceleration(self.qddot),
                    True,
                )
            ]

        else:
            biorbd_return = []
            homogeneous_matrix_transposed = self.biorbd_homogeneous_matrices_in_global(
                GeneralizedCoordinates(self.q),
                reference_index,
                inverse=True,
            )
            for m in self.model.markersAcceleration(
                GeneralizedCoordinates(self.q),
                GeneralizedVelocity(self.qdot),
                GeneralizedAcceleration(self.qddot),
            ):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    biorbd_return.append(m.to_mx())

        casadi_fun = Function(
            "marker_accelerations",
            [self.q, self.qdot, self.qddot],
            biorbd_return,
        )
        return casadi_fun

    def tau_max(self) -> tuple[MX, MX]:
        self.model.closeActuator()
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        torque_max, torque_min = self.model.torqueMax(q_biorbd, qdot_biorbd)
        casadi_fun = Function(
            "tau_max",
            [self.q, self.qdot],
            [torque_max.to_mx(), torque_min.to_mx()],
        )
        return casadi_fun

    def rigid_contact_acceleration(self, contact_index, contact_axis) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_biorbd = GeneralizedAcceleration(self.qddot)
        biorbd_return = self.model.rigidContactAcceleration(
            q_biorbd, qdot_biorbd, qddot_biorbd, contact_index, True
        ).to_mx()[contact_axis]
        casadi_fun = Function(
            "rigid_contact_acceleration",
            [self.q, self.qdot, self.qddot],
            [biorbd_return],
        )
        return casadi_fun

    def markers_jacobian(self) -> list[MX]:
        biorbd_return = [m.to_mx() for m in self.model.markersJacobian(GeneralizedCoordinates(self.q))]
        casadi_fun = Function(
            "markers_jacobian",
            [self.q],
            biorbd_return,
        )
        return casadi_fun

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.to_string() for s in self.model.markerNames()])

    def soft_contact_forces(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)

        biorbd_return = MX.zeros(self.nb_soft_contacts * 6, 1)
        for i_sc in range(self.nb_soft_contacts):
            soft_contact = self.soft_contact(i_sc)

            biorbd_return[i_sc * 6 : (i_sc + 1) * 6, :] = (
                biorbd.SoftContactSphere(soft_contact).computeForceAtOrigin(self.model, q_biorbd, qdot_biorbd).to_mx()
            )

        casadi_fun = Function(
            "soft_contact_forces",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def reshape_fext_to_fcontact(self, fext: MX | SX) -> list:
        if len(self._segments_to_apply_external_forces) == 0:
            parent_name = []
            for i in range(self.nb_rigid_contacts):
                contact = self.model.rigidContact(i)
                parent_name += [
                    self.model.segment(self.model.getBodyRbdlIdToBiorbdId(contact.parentId())).name().to_string()
                ]
            self._segments_to_apply_external_forces = parent_name

        fext_sym = MX.sym("Fext", fext.shape[0], fext.shape[1])
        count = 0
        f_contact_vec = MX()
        for i in range(self.nb_rigid_contacts):
            contact = self.model.rigidContact(i)
            tp = MX.zeros(6)
            used_axes = [i for i, val in enumerate(contact.axes()) if val]
            n_contacts = len(used_axes)
            tp[used_axes] = fext_sym[count : count + n_contacts]
            tp[3:] = contact.to_mx()
            f_contact_vec = horzcat(f_contact_vec, tp)
            count += n_contacts

        casadi_fun_evaluated = Function(
            "reshape_fext_to_fcontact",
            [fext_sym],
            [f_contact_vec],
        )(fext)
        return casadi_fun_evaluated

    def normalize_state_quaternions(self) -> Function:

        quat_idx = self.get_quaternion_idx()
        biorbd_return = MX.zeros(self.nb_q)
        biorbd_return[:] = self.q

        # Normalize quaternion, if needed
        for j in range(self.nb_quaternions):
            quaternion = vertcat(
                self.q[quat_idx[j][3]],
                self.q[quat_idx[j][0]],
                self.q[quat_idx[j][1]],
                self.q[quat_idx[j][2]],
            )
            quaternion /= norm_fro(quaternion)
            biorbd_return[quat_idx[j][0] : quat_idx[j][2] + 1] = quaternion[1:4]
            biorbd_return[quat_idx[j][3]] = quaternion[0]

        casadi_fun = Function(
            "soft_contact_forces",
            [self.q],
            [biorbd_return],
        )
        return casadi_fun

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

    def contact_forces(self, external_forces: MX = None) -> Function:
        if external_forces is not None:
            for i in range(external_forces.shape[1]):
                force = self.contact_forces_from_constrained_forward_dynamics()(
                    self.q, self.qdot, self.tau, external_forces[:, i]
                )
                biorbd_return = force if i == 0 else horzcat(biorbd_return, force)
                casadi_fun = Function(
                    "contact_forces",
                    [self.q, self.qdot, self.tau, self.external_forces],
                    [biorbd_return],
                )
        else:
            biorbd_return = self.contact_forces_from_constrained_forward_dynamics()(
                self.q, self.qdot, self.tau, MX()
            )
            casadi_fun = Function(
                "contact_forces",
                [self.q, self.qdot, self.tau],
                [biorbd_return],
            )
        return casadi_fun

    def passive_joint_torque(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.passiveJointTorque(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "passive_joint_torque",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def ligament_joint_torque(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.ligamentsJointTorque(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "ligament_joint_torque",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

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

    def lagrangian(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.Lagrangian(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "lagrangian",
            [self.q, self.qdot],
            [biorbd_return],
        )
        return casadi_fun

    def partitioned_forward_dynamics(self, external_forces=None, f_contacts=None, q_v_init=None):
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
