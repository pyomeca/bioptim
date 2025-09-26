from typing import Callable

import biorbd_casadi as biorbd
import numpy as np
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function, DM

from bioptim.models.biorbd.external_forces import (
    ExternalForceSetTimeSeries,
    ExternalForceSetVariables,
)
from ..utils import _var_mapping, bounds_from_ranges, cache_function, check_contacts
from ...limits.path_conditions import Bounds
from ...misc.mapping import BiMapping, BiMappingList
from ...misc.enums import ContactType
from ...misc.utils import check_version
from ...optimization.parameters import ParameterList

from ...misc.parameters_types import Int, IntTuple, CX, CXOptional

check_version(biorbd, "1.11.1", "1.13.0")


class BiorbdModel:
    """
    This class wraps the biorbd model and allows the user to call the biorbd functions from the biomodel protocol
    """

    def __init__(
        self,
        bio_model: str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
    ):
        """
        Parameters
        ----------
        bio_model: str | biorbd.Model
            The path to the bioMod file or the biorbd.Model
        friction_coefficients: np.ndarray
            The friction coefficients
        parameters: ParameterList
            The parameters to add to the model. The function will call the callback with the unscaled version of the
            parameters. The user can use this callback to modify the model.
        external_force_set: ExternalForceSetTimeSeries
            The external forces to add to the model
        contact_types: list[ContactType] | tuple[ContactType]
            The type of contacts tu use in the model's dynamics
        """

        if not isinstance(bio_model, str) and not isinstance(bio_model, biorbd.Model):
            raise ValueError("The model should be of type 'str' or 'biorbd.Model'")

        self.model = biorbd.Model(bio_model) if isinstance(bio_model, str) else bio_model

        check_contacts(contact_types, self)
        self.contact_types = contact_types

        if parameters is not None:
            for param_key in parameters:
                parameters[param_key].apply_parameter(self)
        self._friction_coefficients = friction_coefficients

        self.external_force_set = self._set_external_force_set(external_force_set)
        self._symbolic_variables()
        self.biorbd_external_forces_set = self._dispatch_forces() if self.external_force_set else None

        # TODO: remove mx (the MX parameters should be created inside the BiorbdModel)
        self.parameters = parameters.mx if parameters else MX()

        self._cached_functions = {}

    def _symbolic_variables(self):
        """Declaration of MX variables of the right shape for the creation of CasADi Functions"""
        self.q = MX.sym("q_mx", self.nb_q, 1)
        self.qdot = MX.sym("qdot_mx", self.nb_qdot, 1)
        self.qddot = MX.sym("qddot_mx", self.nb_qddot, 1)
        self.qddot_joints = MX.sym("qddot_joints_mx", self.nb_qddot - self.nb_root, 1)
        self.tau = MX.sym("tau_mx", self.nb_tau, 1)
        self.muscle = MX.sym("muscle_mx", self.nb_muscles, 1)
        self.activations = MX.sym("activations_mx", self.nb_muscles, 1)
        self.external_forces = MX.sym(
            "external_forces_mx",
            self.external_force_set.nb_external_forces_components if self.external_force_set else 0,
            1,
        )

    def _set_external_force_set(self, external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables):
        """
        It checks the external forces and binds them to the model.
        """
        if (
            ContactType.RIGID_IMPLICIT in self.contact_types
            or ContactType.SOFT_IMPLICIT in self.contact_types
            or ContactType.SOFT_EXPLICIT in self.contact_types
        ) and isinstance(external_force_set, ExternalForceSetTimeSeries):
            raise NotImplementedError(
                f"Your contact_types {self.contact_types} is not supported yet with external_force_set of type ExternalForceSetTimeSeries."
            )

        if external_force_set is None:
            if len(self.contact_types) > 0 and ContactType.RIGID_EXPLICIT not in self.contact_types:
                new_external_force_set = ExternalForceSetVariables()
                if ContactType.RIGID_IMPLICIT in self.contact_types:
                    for i_contact in range(self.nb_rigid_contacts):
                        new_external_force_set.add(
                            force_name=f"implicit_rigid_contact_{i_contact}",
                            segment=self.rigid_contact_segment(i_contact),
                            use_point_of_application=True,
                        )
                elif ContactType.SOFT_IMPLICIT in self.contact_types:
                    for i_contact in range(self.nb_soft_contacts):
                        new_external_force_set.add(
                            force_name=f"implicit_soft_contact_{i_contact}",
                            segment=self.soft_contact_segment(i_contact),
                            use_point_of_application=True,
                        )
                elif ContactType.SOFT_EXPLICIT in self.contact_types:
                    for i_contact in range(self.nb_soft_contacts):
                        new_external_force_set.add(
                            force_name=f"explicit_soft_contact_{i_contact}",
                            segment=self.soft_contact_segment(i_contact),
                            use_point_of_application=True,
                        )
            else:
                return None
        else:
            new_external_force_set = external_force_set

        new_external_force_set.check_segment_names(tuple([s.name().to_string() for s in self.model.segments()]))
        new_external_force_set.check_all_string_points_of_application(self.marker_names)
        new_external_force_set.bind()

        return new_external_force_set

    @property
    def name(self) -> str:
        # parse the path and split to get the .bioMod name
        return self.model.path().absolutePath().to_string().split("/")[-1]

    @property
    def path(self) -> str:
        return self.model.path().absolutePath().to_string()

    def copy(self):
        return BiorbdModel(self.path)

    def serialize(self) -> tuple[Callable, dict]:
        return BiorbdModel, dict(bio_model=self.path, external_force_set=self.external_force_set)

    @property
    def friction_coefficients(self) -> MX | SX | np.ndarray:
        return self._friction_coefficients

    def set_friction_coefficients(self, new_friction_coefficients) -> None:
        if isinstance(new_friction_coefficients, (DM, np.ndarray)) and np.any(new_friction_coefficients < 0):
            raise ValueError("Friction coefficients must be positive")
        self._friction_coefficients = new_friction_coefficients

    @cache_function
    def gravity(self) -> Function:
        """
        Returns the gravity of the model.
        Since the gravity is self-defined in the model, you need to provide the type of the output when calling the function like this:
        model.gravity()(MX() / SX())
        """
        biorbd_return = self.model.getGravity().to_mx()
        casadi_fun = Function(
            "gravity",
            [self.parameters],
            [biorbd_return],
            ["parameters"],
            ["gravity"],
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
    def nb_ligaments(self) -> int:
        return self.model.nbLigaments()

    @property
    def nb_passive_joint_torques(self) -> int:
        return self.model.nbPassiveTorques()

    @property
    def segments(self) -> tuple[biorbd.Segment]:
        return self.model.segments()

    @cache_function
    def rotation_matrix_to_euler_angles(self, sequence: str) -> Function:
        """
        Returns the rotation matrix to euler angles function.
        """
        r = MX.sym("r_mx", 3, 3)
        r_matrix = biorbd.Rotation(r[0, 0], r[0, 1], r[0, 2], r[1, 0], r[1, 1], r[1, 2], r[2, 0], r[2, 1], r[2, 2])
        biorbd_return = biorbd.Rotation.toEulerAngles(r_matrix, sequence).to_mx()
        casadi_fun = Function(
            "rotation_matrix_to_euler_angles",
            [r],
            [biorbd_return],
            ["Rotation matrix"],
            ["Euler angles"],
        )
        return casadi_fun

    @cache_function
    def homogeneous_matrices_in_global(self, segment_index: int, inverse=False) -> Function:
        """
        Returns the roto-translation matrix of the segment in the global reference frame.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        jcs = self.model.globalJCS(q_biorbd, segment_index)
        biorbd_return = jcs.transpose().to_mx() if inverse else jcs.to_mx()
        casadi_fun = Function(
            "homogeneous_matrices_in_global",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["Joint coordinate system RT matrix in global"],
        )
        return casadi_fun

    @cache_function
    def homogeneous_matrices_in_child(self, segment_id) -> Function:
        """
        Returns the roto-translation matrix of the segment in the child reference frame.
        Since the homogeneous matrix is self-defined in the model, you need to provide the type of the output when calling the function like this:
        model.homogeneous_matrices_in_child(segment_id)(MX() / SX())
        """
        biorbd_return = self.model.localJCS(segment_id).to_mx()
        casadi_fun = Function(
            "homogeneous_matrices_in_child",
            [self.parameters],
            [biorbd_return],
            ["parameters"],
            ["Joint coordinate system RT matrix in local"],
        )
        return casadi_fun

    @cache_function
    def mass(self) -> Function:
        """
        Returns the mass of the model.
        Since the mass is self-defined in the model, you need to provide the type of the output when calling the function like this:
        model.mass()(MX() / SX())
        """
        biorbd_return = self.model.mass().to_mx()
        casadi_fun = Function(
            "mass",
            [self.parameters],
            [biorbd_return],
            ["parameters"],
            ["mass"],
        )
        return casadi_fun

    @cache_function
    def rt(self, rt_index) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.RT(q_biorbd, rt_index).to_mx()
        casadi_fun = Function(
            "rt",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["RT matrix"],
        )
        return casadi_fun

    @cache_function
    def center_of_mass(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.CoM(q_biorbd, True).to_mx()
        casadi_fun = Function(
            "center_of_mass",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["Center of mass"],
        )
        return casadi_fun

    @cache_function
    def center_of_mass_velocity(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.CoMdot(q_biorbd, qdot_biorbd, True).to_mx()
        casadi_fun = Function(
            "center_of_mass_velocity",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["Center of mass velocity"],
        )
        return casadi_fun

    @cache_function
    def center_of_mass_acceleration(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_biorbd = GeneralizedAcceleration(self.qddot)
        biorbd_return = self.model.CoMddot(q_biorbd, qdot_biorbd, qddot_biorbd, True).to_mx()
        casadi_fun = Function(
            "center_of_mass_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot", "parameters"],
            ["Center of mass acceleration"],
        )
        return casadi_fun

    @cache_function
    def body_rotation_rate(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.bodyAngularVelocity(q_biorbd, qdot_biorbd, True).to_mx()
        casadi_fun = Function(
            "body_rotation_rate",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["Body rotation rate"],
        )
        return casadi_fun

    @cache_function
    def mass_matrix(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.massMatrix(q_biorbd).to_mx()
        casadi_fun = Function(
            "mass_matrix",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["Mass matrix"],
        )
        return casadi_fun

    @cache_function
    def non_linear_effects(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.NonLinearEffect(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "non_linear_effects",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["Non linear effects"],
        )
        return casadi_fun

    @cache_function
    def angular_momentum(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.angularMomentum(q_biorbd, qdot_biorbd, True).to_mx()
        casadi_fun = Function(
            "angular_momentum",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["Angular momentum"],
        )
        return casadi_fun

    @cache_function
    def reshape_qdot(self, k_stab=1) -> Function:
        biorbd_return = self.model.computeQdot(
            GeneralizedCoordinates(self.q),
            GeneralizedCoordinates(self.qdot),  # mistake in biorbd
            k_stab,
        ).to_mx()
        casadi_fun = Function(
            "reshape_qdot",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["Reshaped qdot"],
        )
        return casadi_fun

    @cache_function
    def segment_angular_velocity(self, idx) -> Function:
        """
        Returns the angular velocity of the segment in the global reference frame.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.segmentAngularVelocity(q_biorbd, qdot_biorbd, idx, True).to_mx()
        casadi_fun = Function(
            "segment_angular_velocity",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["Segment angular velocity"],
        )
        return casadi_fun

    @cache_function
    def segment_orientation(self, idx: int, sequence: str = "xyz") -> Function:
        """
        Returns the angular position of the segment in the global reference frame.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        rotation_matrix = self.homogeneous_matrices_in_global(idx)(q_biorbd, self.parameters)[:3, :3]
        biorbd_return = self.rotation_matrix_to_euler_angles(sequence=sequence)(rotation_matrix)
        casadi_fun = Function(
            "segment_orientation",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["Segment orientation"],
        )
        return casadi_fun

    @property
    def name_dof(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.nameDof())

    @property
    def rigid_contact_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.contactNames())

    @property
    def nb_soft_contacts(self) -> int:
        return self.model.nbSoftContacts()

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.softContactNames())

    def soft_contact(self, soft_contact_index, *args):
        return self.model.softContact(soft_contact_index, *args)

    @property
    def muscle_names(self) -> tuple[str, ...]:
        return tuple(s.to_string() for s in self.model.muscleNames())

    @property
    def nb_muscles(self) -> int:
        return self.model.nbMuscles()

    @cache_function
    def torque(self) -> Function:
        """
        Returns the torque from the torque_activations.
        Note that tau_activation should be between 0 and 1.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        tau_activations_biorbd = self.tau
        biorbd_return = self.model.torque(tau_activations_biorbd, q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "torque_activation",
            [self.tau, self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["tau", "q", "qdot", "parameters"],
            ["Torque from tau activations"],
        )
        return casadi_fun

    @cache_function
    def forward_dynamics_free_floating_base(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_joints_biorbd = GeneralizedAcceleration(self.qddot_joints)
        biorbd_return = self.model.ForwardDynamicsFreeFloatingBase(q_biorbd, qdot_biorbd, qddot_joints_biorbd).to_mx()
        casadi_fun = Function(
            "forward_dynamics_free_floating_base",
            [self.q, self.qdot, self.qddot_joints, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot_joints", "parameters"],
            ["qddot_root and qddot_joints"],
        )
        return casadi_fun

    @staticmethod
    def reorder_qddot_root_joints(qddot_root, qddot_joints) -> MX | SX:
        return vertcat(qddot_root, qddot_joints)

    def _dispatch_forces(self) -> biorbd.ExternalForceSet:
        """Dispatch the symbolic MX into the biorbd external forces object"""
        biorbd_external_forces = self.model.externalForceSet()

        # "type of external force": (function to call, number of force components)
        force_mapping = {
            "in_global": (self.external_force_set.add_global_force, 6),
            "torque_in_global": (self.external_force_set.add_torque_global, 3),
            "translational_in_global": (self.external_force_set.add_translational_global, 3),
            "in_local": (self.external_force_set.add_local_force, 6),
            "torque_in_local": (self.external_force_set.add_torque_local, 3),
        }

        symbolic_counter = 0
        for force_type, val in force_mapping.items():
            add_force_func, num_force_components = val
            symbolic_counter = self._dispatch_forces_of_type(
                force_type, add_force_func, num_force_components, symbolic_counter, biorbd_external_forces
            )

        return biorbd_external_forces

    def _dispatch_forces_of_type(
        self,
        force_type: str,
        add_force_func: "Callable",
        num_force_components: int,
        symbolic_counter: int,
        biorbd_external_forces: "biorbd.ExternalForces",
    ) -> int:
        """
        Helper method to dispatch forces of a specific external forces.

        Parameters
        ----------
        force_type: str
            The type of external force to dispatch among in_global, torque_in_global, translational_in_global, in_local, torque_in_local.
        add_force_func: Callable
            The function to call to add the force to the biorbd external forces object.
        num_force_components: int
            The number of force components for the given type
        symbolic_counter: int
            The current symbolic counter to slice the whole external_forces mx.
        biorbd_external_forces: biorbd.ExternalForces
            The biorbd external forces object to add the forces to.

        Returns
        -------
        int
            The updated symbolic counter.
        """
        for force_characteristics in getattr(self.external_force_set, force_type).values():
            force_slicer = slice(symbolic_counter, symbolic_counter + num_force_components)

            point_of_application_mx = self._get_point_of_application(force_characteristics, force_slicer.stop)

            add_force_func(
                biorbd_external_forces,
                force_characteristics["segment"],
                self.external_forces[force_slicer],
                point_of_application_mx,
            )
            symbolic_counter = force_slicer.stop + (
                3 if isinstance(force_characteristics["point_of_application"], np.ndarray) else 0
            )

        return symbolic_counter

    def _get_point_of_application(self, force, stop_index) -> biorbd.NodeSegment | np.ndarray | None:
        """
        Determine the point of application mx slice based on its type. Only sliced if an array is stored

        Parameters
        ----------
        force : dict
            The force dictionary with details on the point of application.
        stop_index : int
            Index position in MX where the point of application components start.

        Returns
        -------
        biorbd.NodeSegment | np.ndarray | None
            Returns a slice of MX, a marker node, or None if no point of application is defined.
        """
        if isinstance(force["point_of_application"], np.ndarray):
            return self.external_forces[slice(stop_index, stop_index + 3)]
        elif isinstance(force["point_of_application"], MX):
            return self.external_forces[slice(stop_index, stop_index + 3)]
        elif isinstance(force["point_of_application"], str):
            value = self.model.marker(self.marker_index(force["point_of_application"]))
            if not isinstance(value, MX):
                value = value.to_mx()
            return value
        else:
            return None

    @staticmethod
    def _get_cx(var: CX) -> CXOptional:
        if isinstance(var, MX):
            return MX
        elif isinstance(var, SX):
            return SX
        else:
            raise TypeError(f"The type of the variable should be either MX or SX, not {type(var)}.")

    def map_rigid_contact_forces_to_global_forces(
        self, rigid_contact_forces: MX | SX, q: MX | SX, parameters: MX | SX
    ) -> MX | SX:
        """
        Takes the rigid contact forces and dispatch is to match the external forces.
        """
        cx = self._get_cx(q)
        external_forces = cx.zeros(self.external_force_set.nb_external_forces_components)

        current_index = 0
        contacts_to_add = 0
        for i_contact in range(self.nb_rigid_contacts):
            # Skip the moments
            contacts_to_add += 3

            # Add the forces to the right place
            available_axes = np.array(self.rigid_contact_axes_index(i_contact))
            contact_force_idx = range(current_index, current_index + available_axes.shape[0])
            for i, idx in enumerate(contact_force_idx):
                external_forces[contacts_to_add + available_axes[i]] += rigid_contact_forces[idx]
            current_index += available_axes.shape[0]
            contacts_to_add += 3

            # Add the point of application to the right place
            external_forces[contacts_to_add : contacts_to_add + 3] = self.rigid_contact_position(i_contact)(
                q, parameters
            )
            contacts_to_add += 3

        return external_forces

    def map_soft_contact_forces_to_global_forces(self, soft_contact_forces: MX | SX) -> MX | SX:
        """
        Takes the soft contact forces and dispatch is to match the external forces.
        """

        external_forces = type(soft_contact_forces).zeros(9 * self.nb_soft_contacts)
        for i_contact in range(self.nb_soft_contacts):
            external_forces[i_contact * 9 + 3 : i_contact * 9 + 9] = soft_contact_forces[
                i_contact * 6 : i_contact * 6 + 6
            ]
        return external_forces

    @cache_function
    def forward_dynamics(self, with_contact: bool = False) -> Function:

        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        tau_biorbd = GeneralizedTorque(self.tau)

        if with_contact:
            if self.external_force_set is None:
                biorbd_return = self.model.ForwardDynamicsConstraintsDirect(q_biorbd, qdot_biorbd, tau_biorbd).to_mx()
            else:
                biorbd_return = self.model.ForwardDynamicsConstraintsDirect(
                    q_biorbd, qdot_biorbd, tau_biorbd, self.biorbd_external_forces_set
                ).to_mx()
            casadi_fun = Function(
                "constrained_forward_dynamics",
                [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
                [biorbd_return],
                ["q", "qdot", "tau", "external_forces", "parameters"],
                ["qddot"],
            )
        else:
            if self.external_force_set is None:
                biorbd_return = self.model.ForwardDynamics(q_biorbd, qdot_biorbd, tau_biorbd).to_mx()
            else:
                biorbd_return = self.model.ForwardDynamics(
                    q_biorbd, qdot_biorbd, tau_biorbd, self.biorbd_external_forces_set
                ).to_mx()
            casadi_fun = Function(
                "forward_dynamics",
                [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
                [biorbd_return],
                ["q", "qdot", "tau", "external_forces", "parameters"],
                ["qddot"],
            )
        return casadi_fun

    @cache_function
    def inverse_dynamics(self, with_contact: bool = False) -> Function:

        if with_contact:
            raise NotImplementedError("Inverse dynamics with contact is not implemented yet")

        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_biorbd = GeneralizedAcceleration(self.qddot)
        if self.external_force_set is None:
            biorbd_return = self.model.InverseDynamics(q_biorbd, qdot_biorbd, qddot_biorbd).to_mx()
        else:
            biorbd_return = self.model.InverseDynamics(
                q_biorbd, qdot_biorbd, qddot_biorbd, self.biorbd_external_forces_set
            ).to_mx()
        casadi_fun = Function(
            "inverse_dynamics",
            [self.q, self.qdot, self.qddot, self.external_forces, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot", "external_forces", "parameters"],
            ["tau"],
        )
        return casadi_fun

    @cache_function
    def contact_forces_from_constrained_forward_dynamics(self) -> Function:

        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        tau_biorbd = GeneralizedTorque(self.tau)
        if self.external_force_set is None:
            biorbd_return = self.model.ContactForcesFromForwardDynamicsConstraintsDirect(
                q_biorbd, qdot_biorbd, tau_biorbd
            ).to_mx()
        else:
            biorbd_return = self.model.ContactForcesFromForwardDynamicsConstraintsDirect(
                q_biorbd, qdot_biorbd, tau_biorbd, self.biorbd_external_forces_set
            ).to_mx()
        casadi_fun = Function(
            "contact_forces_from_constrained_forward_dynamics",
            [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
            [biorbd_return],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["contact_forces"],
        )
        return casadi_fun

    @cache_function
    def rigid_contact_position(self, index: int) -> Function:
        """
        Returns the position of the rigid contact (contact_index) in the global reference frame.
        """
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.rigidContact(q_biorbd, index, True).to_mx()
        casadi_fun = Function(
            "rigid_contact_position",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["Rigid contact position"],
        )
        return casadi_fun

    @cache_function
    def forces_on_each_rigid_contact_point(self) -> Function:
        """
        Returns the 3D force acting on each contact point in the global reference frame computed from the constrained forward dynamics.
        """
        contact_forces = self.contact_forces_from_constrained_forward_dynamics()(
            self.q, self.qdot, self.tau, self.external_forces, self.parameters
        )

        # Rearrange the forces to get all 3 components for each contact point
        forces_on_each_point = None
        current_index = 0
        for i_contact in range(self.nb_rigid_contacts):
            available_axes = np.array(self.rigid_contact_axes_index(i_contact))
            contact_force_idx = range(current_index, current_index + available_axes.shape[0])
            current_force = MX.zeros(3)
            for i, contact_to_add in enumerate(contact_force_idx):
                current_force[available_axes[i]] += contact_forces[contact_to_add]
            current_index += available_axes.shape[0]
            if forces_on_each_point is not None:
                forces_on_each_point = horzcat(forces_on_each_point, current_force)
            else:
                forces_on_each_point = current_force

        casadi_fun = Function(
            "reaction_forces",
            [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
            [forces_on_each_point],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["forces_on_each_point"],
        )
        return casadi_fun

    @cache_function
    def qdot_from_impact(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_pre_impact_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.ComputeConstraintImpulsesDirect(q_biorbd, qdot_pre_impact_biorbd).to_mx()
        casadi_fun = Function(
            "qdot_from_impact",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["qdot post impact"],
        )
        return casadi_fun

    @cache_function
    def muscle_activation_dot(self) -> Function:
        muscle_states = self.model.stateSet()
        for k in range(self.model.nbMuscles()):
            muscle_states[k].setActivation(self.activations[k])
            muscle_states[k].setExcitation(self.muscle[k])
        biorbd_return = self.model.activationDot(muscle_states).to_mx()
        casadi_fun = Function(
            "muscle_activation_dot",
            [self.muscle, self.activations, self.parameters],
            [biorbd_return],
            ["muscle_excitation", "muscle_activation", "parameters"],
            ["muscle_activation_dot"],
        )
        return casadi_fun

    @cache_function
    def muscle_length_jacobian(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        biorbd_return = self.model.musclesLengthJacobian(q_biorbd).to_mx()
        casadi_fun = Function(
            "muscle_length_jacobian",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["muscle_length_jacobian"],
        )
        return casadi_fun

    @cache_function
    def muscle_velocity(self) -> Function:
        J = self.muscle_length_jacobian()(self.q, self.parameters)
        biorbd_return = J @ self.qdot
        casadi_fun = Function(
            "muscle_velocity",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["muscle_velocity"],
        )
        return casadi_fun

    @cache_function
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
            [self.muscle, self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["muscle_activation", "q", "qdot", "parameters"],
            ["muscle_joint_torque"],
        )
        return casadi_fun

    @cache_function
    def markers(self) -> list[MX]:
        biorbd_return = horzcat(*[m.to_mx() for m in self.model.markers(GeneralizedCoordinates(self.q))])
        casadi_fun = Function(
            "markers",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["markers"],
        )
        return casadi_fun

    @property
    def nb_markers(self) -> int:
        return self.model.nbMarkers()

    def marker_index(self, name):
        return biorbd.marker_index(self.model, name)

    def contact_index(self, name):
        return biorbd.contact_index(self.model, name)

    @cache_function
    def marker(self, index: int, reference_segment_index: int = None) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        marker = self.model.marker(q_biorbd, index)
        if reference_segment_index is not None:
            global_homogeneous_matrix = self.model.globalJCS(q_biorbd, reference_segment_index)
            marker_rotated = global_homogeneous_matrix.transpose().to_mx() @ vertcat(marker.to_mx(), 1)
            biorbd_return = marker_rotated[:3]
        else:
            biorbd_return = marker.to_mx()
        casadi_fun = Function(
            "marker",
            [self.q, self.parameters],
            [biorbd_return],
            ["q", "parameters"],
            ["marker"],
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

    def rigid_contact_axes_index(self, contact_index) -> list:
        """
        Returns the axis index of this specific rigid contact.
        Example:
            First contact with axis YZ
            Second contact with axis Z
            rigid_contact_axes_index(0) = (1, 2)
        """
        return list(self.model.rigidContacts()[contact_index].availableAxesIndices())

    def rigid_contact_segment(self, contact_index) -> str:
        """
        Returns the name of the segment on which this specific rigid contact is.
        """
        for segment in self.model.segments():
            if segment.id() == self.model.rigidContact(contact_index).parentId():
                return segment.name().to_string()
        raise RuntimeError(f"The segment for the rigid contact index {contact_index} was not found.")

    def soft_contact_segment(self, contact_index) -> str:
        """
        Returns the name of the segment on which this specific rigid contact is.
        """
        for segment in self.model.segments():
            if segment.id() == self.model.softContact(contact_index).parentId():
                return segment.name().to_string()
        raise RuntimeError(f"The segment for the soft contact index {contact_index} was not found.")

    @cache_function
    def markers_velocities(self, reference_index=None) -> list[MX]:
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
            homogeneous_matrix_transposed = self.homogeneous_matrices_in_global(
                segment_index=reference_index, inverse=True
            )(
                GeneralizedCoordinates(self.q),
            )
            for m in self.model.markersVelocity(GeneralizedCoordinates(self.q), GeneralizedVelocity(self.qdot)):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    biorbd_return.append(m.to_mx())
                else:
                    biorbd_return.append(m.applyRT(homogeneous_matrix_transposed).to_mx())

        casadi_fun = Function(
            "markers_velocities",
            [self.q, self.qdot, self.parameters],
            [horzcat(*biorbd_return)],
            ["q", "qdot", "parameters"],
            ["markers_velocities"],
        )
        return casadi_fun

    @cache_function
    def marker_velocity(self, marker_index: int) -> list[MX]:
        biorbd_return = self.model.markersVelocity(
            GeneralizedCoordinates(self.q),
            GeneralizedVelocity(self.qdot),
            True,
        )[marker_index].to_mx()
        casadi_fun = Function(
            "marker_velocity",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["marker_velocity"],
        )
        return casadi_fun

    @cache_function
    def markers_accelerations(self, reference_index=None) -> list[MX]:
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
            homogeneous_matrix_transposed = self.homogeneous_matrices_in_global(
                segment_index=reference_index,
                inverse=True,
            )(
                GeneralizedCoordinates(self.q),
            )
            for m in self.model.markersAcceleration(
                GeneralizedCoordinates(self.q),
                GeneralizedVelocity(self.qdot),
                GeneralizedAcceleration(self.qddot),
            ):
                if m.applyRT(homogeneous_matrix_transposed) is None:
                    biorbd_return.append(m.to_mx())
                else:
                    biorbd_return.append(m.applyRT(homogeneous_matrix_transposed).to_mx())

        casadi_fun = Function(
            "markers_accelerations",
            [self.q, self.qdot, self.qddot, self.parameters],
            [horzcat(*biorbd_return)],
            ["q", "qdot", "qddot", "parameters"],
            ["markers_accelerations"],
        )
        return casadi_fun

    @cache_function
    def marker_acceleration(self, marker_index: int) -> list[MX]:
        biorbd_return = self.model.markerAcceleration(
            GeneralizedCoordinates(self.q),
            GeneralizedVelocity(self.qdot),
            GeneralizedAcceleration(self.qddot),
            True,
        )[marker_index].to_mx()
        casadi_fun = Function(
            "marker_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot", "parameters"],
            ["marker_acceleration"],
        )
        return casadi_fun

    @cache_function
    def tau_max(self) -> Function:
        self.model.closeActuator()
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        torque_max, torque_min = self.model.torqueMax(q_biorbd, qdot_biorbd)
        casadi_fun = Function(
            "tau_max",
            [self.q, self.qdot, self.parameters],
            [torque_max.to_mx(), torque_min.to_mx()],
            ["q", "qdot", "parameters"],
            ["tau_max", "tau_min"],
        )
        return casadi_fun

    @cache_function
    def rigid_contact_velocity(self, contact_index: Int, contact_axis: IntTuple = None) -> Function:
        contact_axis = [0, 1, 2] if contact_axis is None else contact_axis
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)

        velocity = self.model.rigidContactVelocity(q_biorbd, qdot_biorbd, contact_index, True).to_mx()
        biorbd_return = MX()
        for axis in contact_axis:
            biorbd_return = vertcat(biorbd_return, velocity[axis])
        casadi_fun = Function(
            "rigid_contact_velocity",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["rigid_contact_velocity"],
        )
        return casadi_fun

    @cache_function
    def rigid_contact_acceleration(self, contact_index, contact_axis) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        qddot_biorbd = GeneralizedAcceleration(self.qddot)
        biorbd_return = self.model.rigidContactAcceleration(
            q_biorbd, qdot_biorbd, qddot_biorbd, contact_index, True
        ).to_mx()[contact_axis]
        casadi_fun = Function(
            "rigid_contact_acceleration",
            [self.q, self.qdot, self.qddot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "qddot", "parameters"],
            ["rigid_contact_acceleration"],
        )
        return casadi_fun

    @cache_function
    def markers_jacobian(self) -> Function:
        biorbd_return = [m.to_mx() for m in self.model.markersJacobian(GeneralizedCoordinates(self.q))]
        casadi_fun = Function(
            "markers_jacobian",
            [self.q, self.parameters],
            biorbd_return,
            ["q", "parameters"],
            ["markers_jacobian"],
        )
        return casadi_fun

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple([s.to_string() for s in self.model.markerNames()])

    @cache_function
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
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["soft_contact_forces"],
        )
        return casadi_fun

    @cache_function
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
            "normalize_state_quaternions",
            [self.q],
            [biorbd_return],
            ["q"],
            ["q_normalized"],
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

    @cache_function
    def rigid_contact_forces(self) -> Function:
        force = self.contact_forces_from_constrained_forward_dynamics()(
            self.q, self.qdot, self.tau, self.external_forces, self.parameters
        )
        casadi_fun = Function(
            "contact_forces",
            [self.q, self.qdot, self.tau, self.external_forces, self.parameters],
            [force],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["contact_forces"],
        )
        return casadi_fun

    @cache_function
    def passive_joint_torque(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.passiveJointTorque(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "passive_joint_torque",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["passive_joint_torque"],
        )
        return casadi_fun

    @cache_function
    def ligament_joint_torque(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.ligamentsJointTorque(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "ligament_joint_torque",
            [self.q, self.qdot, self.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["ligament_joint_torque"],
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

    @cache_function
    def lagrangian(self) -> Function:
        q_biorbd = GeneralizedCoordinates(self.q)
        qdot_biorbd = GeneralizedVelocity(self.qdot)
        biorbd_return = self.model.Lagrangian(q_biorbd, qdot_biorbd).to_mx()
        casadi_fun = Function(
            "lagrangian",
            [self.q, self.qdot],
            [biorbd_return],
            ["q", "qdot"],
            ["lagrangian"],
        )
        return casadi_fun

    def partitioned_forward_dynamics(self):
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
