from typing import Callable, Any

from casadi import vertcat, Function, DM, horzcat
import numpy as np

from .configure_new_variable import NewVariableConfiguration
from .dynamics_functions import DynamicsFunctions
from .fatigue.fatigue_dynamics import FatigueList
from .ode_solver import OdeSolver
from ..gui.plot import CustomPlot
from ..misc.enums import (
    PlotType,
    Node,
    ConstraintType,
    RigidBodyDynamics,
    SoftContactDynamics,
    PhaseDynamics,
)
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping, Mapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..limits.constraints import ImplicitConstraintFcn
from ..models.protocols.stochastic_biomodel import StochasticBioModel
from ..optimization.problem_type import SocpType


class ConfigureProblem:
    """
    Dynamics configuration for the most common ocp

    Methods
    -------
    initialize(ocp, nlp)
        Call the dynamics a first time
    custom(ocp, nlp, **extra_params)
        Call the user-defined dynamics configuration function
    torque_driven(ocp, nlp, with_contact=False)
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)
    torque_derivative_driven(ocp, nlp, with_contact=False)
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)
    torque_activations_driven(ocp, nlp, with_contact=False)
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau activations).
        The tau activations are bounded between -1 and 1 and actual tau is computed from torque-position-velocity
        relationship
    muscle_driven(
        ocp, nlp, with_excitations: bool = False, with_residual_torque: bool = False, with_contact: bool = False
    )
        Configure the dynamics for a muscle driven program.
        If with_excitations is set to True, then the muscle muscle activations are computed from the muscle dynamics.
        The tau from muscle is computed using the muscle activations.
        If with_residual_torque is set to True, then tau are used as supplementary force in the
        case muscles are too weak.
    configure_dynamics_function(ocp, nlp, dyn_func, **extra_params)
        Configure the dynamics of the system
    configure_contact_function(ocp, nlp, dyn_func: Callable, **extra_params)
        Configure the contact points
    configure_soft_contact_function
        Configure the soft contact function
    configure_new_variable(
        name: str, name_elements: list, nlp, as_states: bool, as_controls: bool, combine_state_control_plot: bool = False
    )
        Add a new variable to the states/controls pool
    configure_q(nlp, as_states: bool, as_controls: bool)
        Configure the generalized coordinates
    configure_qdot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized velocities
    configure_qddot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized accelerations
    configure_qdddot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized jerks
    configure_tau(nlp, as_states: bool, as_controls: bool)
        Configure the generalized forces
    configure_residual_tau(nlp, as_states: bool, as_controls: bool)
        Configure the residual forces
    configure_taudot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized forces derivative
    configure_muscles(nlp, as_states: bool, as_controls: bool)
        Configure the muscles
    """

    @staticmethod
    def _get_kinematics_based_names(nlp, var_type: str) -> list[str]:
        """
        To modify the names of the variables added to the plots if there is quaternions

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        var_type: str
            A string that refers to the decision variable such as (q, qdot, qddot, tau, etc...)

        Returns
        ----------
        new_name: list[str]
            The list of str to display on figures
        """

        idx = nlp.phase_mapping.to_first.map_idx if nlp.phase_mapping else range(nlp.model.nb_q)

        if nlp.model.nb_quaternions == 0:
            new_names = [nlp.model.name_dof[i] for i in idx]
        else:
            new_names = []
            for i in nlp.phase_mapping.to_first.map_idx:
                if nlp.model.name_dof[i][-4:-1] == "Rot" or nlp.model.name_dof[i][-6:-1] == "Trans":
                    new_names += [nlp.model.name_dof[i]]
                else:
                    if nlp.model.name_dof[i][-5:] != "QuatW":
                        if var_type == "qdot":
                            new_names += [nlp.model.name_dof[i][:-5] + "omega" + nlp.model.name_dof[i][-1]]
                        elif var_type == "qddot":
                            new_names += [nlp.model.name_dof[i][:-5] + "omegadot" + nlp.model.name_dof[i][-1]]
                        elif var_type == "qdddot":
                            new_names += [nlp.model.name_dof[i][:-5] + "omegaddot" + nlp.model.name_dof[i][-1]]
                        elif var_type == "tau" or var_type == "taudot":
                            new_names += [nlp.model.name_dof[i]]

        return new_names

    @staticmethod
    def initialize(ocp, nlp):
        """
        Call the dynamics a first time

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.type(ocp, nlp, **nlp.dynamics_type.extra_parameters)

    @staticmethod
    def custom(ocp,
               nlp,
               dynamics_constants_used_at_each_nodes: dict[list] = {},
               **extra_params):
        """
        Call the user-defined dynamics configuration function

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.configure(ocp, nlp, dynamics_constants_used_at_each_nodes, **extra_params)

    @staticmethod
    def torque_driven(
        ocp,
        nlp,
        with_contact: bool = False,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        soft_contacts_dynamics: SoftContactDynamics = SoftContactDynamics.ODE,
        fatigue: FatigueList = None,
        dynamics_constants_used_at_each_nodes: dict[list] = {},
    ):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficients * qdot)
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        soft_contacts_dynamics: SoftContactDynamics
            which soft contact dynamic should be used
        fatigue: FatigueList
            A list of fatigue elements
        dynamics_constants_used_at_each_nodes: dict[np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """

        _check_contacts_in_biorbd_model(with_contact, nlp.model.nb_contacts, nlp.phase_idx)
        _check_soft_contacts_dynamics(
            rigidbody_dynamics, soft_contacts_dynamics, nlp.model.nb_soft_contacts, nlp.phase_idx
        )
        external_forces = None
        for key in dynamics_constants_used_at_each_nodes.keys():
            if key != "external_forces":
                raise RuntimeError(
                    "The only dynamics_constants_used_at_each_nodes allowed for torque_driven dynamics is external_forces."
                )
            _check_dynamics_constants_format(dynamics_constants_used_at_each_nodes[key], nlp.ns, nlp.phase_idx)
            external_forces = nlp.dynamics_constants[0].mx
            for i in range(1, dynamics_constants_used_at_each_nodes[key].shape[1]):
                external_forces = horzcat(external_forces, nlp.dynamics_constants[i].mx)

        # Declared rigidbody states and controls
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True, fatigue=fatigue)

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
        ):
            ConfigureProblem.configure_qddot(ocp, nlp, False, True, True)
        elif (
            rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
        ):
            ConfigureProblem.configure_qddot(ocp, nlp, True, False, True)
            ConfigureProblem.configure_qdddot(ocp, nlp, False, True)
        else:
            ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)

        # Algebraic constraints of rigidbody dynamics if needed
        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
        ):
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.TAU_EQUALS_INVERSE_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
                with_contact=with_contact,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                with_friction=with_friction,
            )
            if with_contact:
                # qddot is continuous with RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                # so the consistency constraint of the marker acceleration can only be set to zero
                # at the first shooting node
                node = Node.ALL_SHOOTING if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS else Node.ALL
                ConfigureProblem.configure_contact_forces(ocp, nlp, False, True)
                for ii in range(nlp.model.nb_rigid_contacts):
                    for jj in nlp.model.rigid_contact_index(ii):
                        ocp.implicit_constraints.add(
                            ImplicitConstraintFcn.CONTACT_ACCELERATION_EQUALS_ZERO,
                            with_contact=with_contact,
                            contact_index=ii,
                            contact_axis=jj,
                            node=node,
                            constraint_type=ConstraintType.IMPLICIT,
                            phase=nlp.phase_idx,
                        )
        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            # contacts forces are directly handled with this constraint
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.QDDOT_EQUALS_FORWARD_DYNAMICS,
                node=Node.ALL_SHOOTING,
                constraint_type=ConstraintType.IMPLICIT,
                with_contact=with_contact,
                phase=nlp.phase_idx,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                with_friction=with_friction,
            )

        # Declared soft contacts controls
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ConfigureProblem.configure_soft_contact_forces(ocp, nlp, False, True)

        # Configure the actual ODE of the dynamics
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_driven,
                with_contact=with_contact,
                fatigue=fatigue,
                rigidbody_dynamics=rigidbody_dynamics,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                with_friction=with_friction,
                external_forces=external_forces,
            )

        # Configure the contact forces
        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp, nlp, DynamicsFunctions.forces_from_torque_driven, external_forces=external_forces
            )
        # Configure the soft contact forces
        ConfigureProblem.configure_soft_contact_function(ocp, nlp)
        # Algebraic constraints of soft contact forces if needed
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.SOFT_CONTACTS_EQUALS_SOFT_CONTACTS_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
            )

    @staticmethod
    def torque_driven_free_floating_base(
        ocp,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
    ):
        """
        Configure the dynamics for a torque driven program with a free floating base.
        This version of the torque driven dynamics avoids defining a mapping to force the root to generate null forces and torques.
        (states are q_root, q_joints, qdot_root, and qdot_joints, controls are tau_joints)
        Please note that it was not meant to be used with quaternions yet.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficients * qdot)
        """

        nb_q = nlp.model.nb_q
        nb_qdot = nlp.model.nb_qdot
        nb_root = nlp.model.nb_root

        # Declared rigidbody states and controls
        name_q_roots = [str(i) for i in range(nb_root)]
        ConfigureProblem.configure_new_variable(
            "q_roots",
            name_q_roots,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        name_q_joints = [str(i) for i in range(nb_root, nb_q)]
        ConfigureProblem.configure_new_variable(
            "q_joints",
            name_q_joints,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        ConfigureProblem.configure_new_variable(
            "qdot_roots",
            name_q_roots,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=True,
        )

        name_qdot_joints = [str(i) for i in range(nb_root, nb_qdot)]
        ConfigureProblem.configure_new_variable(
            "qdot_joints",
            name_qdot_joints,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=True,
        )

        ConfigureProblem.configure_new_variable(
            "qddot_roots",
            name_q_roots,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=True,
        )

        ConfigureProblem.configure_new_variable(
            "qddot_joints",
            name_qdot_joints,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=True,
        )

        ConfigureProblem.configure_new_variable(
            "tau_joints",
            name_qdot_joints,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_states_dot=False,
        )

        # TODO: add implicit constraints + soft contacts + fatigue

        # Configure the actual ODE of the dynamics
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_driven_free_floating_base,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                with_friction=with_friction,
            )

    @staticmethod
    def stochastic_torque_driven(
        ocp,
        nlp,
        problem_type,
        with_contact: bool = False,
        with_friction: bool = False,
        with_cholesky: bool = False,
        initial_matrix: DM = None,
    ):
        """
        Configure the dynamics for a torque driven stochastic program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficient * qdot)
        """

        if "tau" in nlp.model.motor_noise_mapping:
            n_noised_tau = len(nlp.model.motor_noise_mapping["tau"].to_first.map_idx)
        else:
            n_noised_tau = nlp.model.nb_tau
        n_noise = nlp.model.motor_noise_magnitude.shape[0] + nlp.model.sensory_noise_magnitude.shape[0]
        n_noised_states = 2 * n_noised_tau

        # Algebraic states variables
        ConfigureProblem.configure_stochastic_k(
            ocp, nlp, n_noised_controls=n_noised_tau, n_references=nlp.model.n_references
        )
        ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=nlp.model.n_references)
        n_collocation_points = 1
        if isinstance(problem_type, SocpType.COLLOCATION):
            n_collocation_points += problem_type.polynomial_degree
        ConfigureProblem.configure_stochastic_m(
            ocp, nlp, n_noised_states=n_noised_states, n_collocation_points=n_collocation_points
        )

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            if initial_matrix is None:
                raise RuntimeError(
                    "The initial value for the covariance matrix must be provided for TRAPEZOIDAL_EXPLICIT"
                )
            ConfigureProblem.configure_stochastic_cov_explicit(
                ocp, nlp, n_noised_states=n_noised_states, initial_matrix=initial_matrix
            )
        else:
            if with_cholesky:
                ConfigureProblem.configure_stochastic_cholesky_cov(ocp, nlp, n_noised_states=n_noised_states)
            else:
                ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=n_noised_states)

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_IMPLICIT):
            ConfigureProblem.configure_stochastic_a(ocp, nlp, n_noised_states=n_noised_states)
            ConfigureProblem.configure_stochastic_c(ocp, nlp, n_noised_states=n_noised_states, n_noise=n_noise)

        ConfigureProblem.torque_driven(
            ocp=ocp,
            nlp=nlp,
            with_contact=with_contact,
            with_friction=with_friction,
        )

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            DynamicsFunctions.stochastic_torque_driven,
            with_contact=with_contact,
            with_friction=with_friction,
        )

    @staticmethod
    def stochastic_torque_driven_free_floating_base(
        ocp,
        nlp,
        problem_type,
        with_contact: bool = False,
        with_friction: bool = False,
        with_cholesky: bool = False,
        initial_matrix: DM = None,
    ):
        """
        Configure the dynamics for a stochastic torque driven program with a free floating base.
        (states are q_roots, q_joints, qdot_roots, and qdot_joints, controls are tau_joints)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficient * qdot)
        """
        n_noised_tau = nlp.model.n_noised_controls
        n_noise = nlp.model.motor_noise_magnitude.shape[0] + nlp.model.sensory_noise_magnitude.shape[0]
        n_noised_states = nlp.model.n_noised_states

        # Stochastic variables
        ConfigureProblem.configure_stochastic_k(
            ocp, nlp, n_noised_controls=n_noised_tau, n_references=nlp.model.n_references
        )
        ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=nlp.model.n_references)
        n_collocation_points = 1
        if isinstance(problem_type, SocpType.COLLOCATION):
            n_collocation_points += problem_type.polynomial_degree
        ConfigureProblem.configure_stochastic_m(
            ocp, nlp, n_noised_states=n_noised_states, n_collocation_points=n_collocation_points
        )

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            if initial_matrix is None:
                raise RuntimeError(
                    "The initial value for the covariance matrix must be provided for TRAPEZOIDAL_EXPLICIT"
                )
            ConfigureProblem.configure_stochastic_cov_explicit(
                ocp, nlp, n_noised_states=n_noised_states, initial_matrix=initial_matrix
            )
        else:
            if with_cholesky:
                ConfigureProblem.configure_stochastic_cholesky_cov(ocp, nlp, n_noised_states=n_noised_states)
            else:
                ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=n_noised_states)

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_IMPLICIT):
            ConfigureProblem.configure_stochastic_a(ocp, nlp, n_noised_states=n_noised_states)
            ConfigureProblem.configure_stochastic_c(ocp, nlp, n_noised_states=n_noised_states, n_noise=n_noise)

        ConfigureProblem.torque_driven_free_floating_base(
            ocp=ocp,
            nlp=nlp,
            with_friction=with_friction,
        )

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            DynamicsFunctions.stochastic_torque_driven_free_floating_base,
            with_friction=with_friction,
        )

    @staticmethod
    def torque_derivative_driven(
        ocp,
        nlp,
        with_contact=False,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        soft_contacts_dynamics: SoftContactDynamics = SoftContactDynamics.ODE,
        dynamics_constants_used_at_each_nodes: dict[list] = {},
    ):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        soft_contacts_dynamics: SoftContactDynamics
            which soft contact dynamic should be used
        dynamics_constants_used_at_each_nodes: dict[np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.

        """
        _check_contacts_in_biorbd_model(with_contact, nlp.model.nb_contacts, nlp.phase_idx)

        if rigidbody_dynamics not in (RigidBodyDynamics.DAE_INVERSE_DYNAMICS, RigidBodyDynamics.ODE):
            raise NotImplementedError("TORQUE_DERIVATIVE_DRIVEN cannot be used with this enum RigidBodyDynamics yet")

        _check_soft_contacts_dynamics(
            rigidbody_dynamics, soft_contacts_dynamics, nlp.model.nb_soft_contacts, nlp.phase_idx
        )
        external_forces = None
        for key in dynamics_constants_used_at_each_nodes.keys():
            if key != "external_forces":
                raise RuntimeError(
                    "The only dynamics_constants_used_at_each_nodes allowed for torque_driven dynamics is external_forces."
                )
            _check_dynamics_constants_format(dynamics_constants_used_at_each_nodes[key], nlp.ns, nlp.phase_idx)
            external_forces = nlp.dynamics_constants[0].mx
            for i in range(1, dynamics_constants_used_at_each_nodes[key].shape[1]):
                external_forces = horzcat(external_forces, nlp.dynamics_constants[i].mx)

        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False)
        ConfigureProblem.configure_tau(ocp, nlp, True, False)
        ConfigureProblem.configure_taudot(ocp, nlp, False, True)

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            ConfigureProblem.configure_qddot(ocp, nlp, True, False)
            ConfigureProblem.configure_qdddot(ocp, nlp, False, True)
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.TAU_EQUALS_INVERSE_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
            )
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ConfigureProblem.configure_soft_contact_forces(ocp, nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_derivative_driven,
                with_contact=with_contact,
                rigidbody_dynamics=rigidbody_dynamics,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                external_forces=external_forces,
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp,
                nlp,
                DynamicsFunctions.forces_from_torque_driven,
                external_forces=external_forces,
            )

        ConfigureProblem.configure_soft_contact_function(ocp, nlp)
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.SOFT_CONTACTS_EQUALS_SOFT_CONTACTS_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
            )

    @staticmethod
    def torque_activations_driven(
        ocp,
        nlp,
        with_contact: bool = False,
        with_passive_torque: bool = False,
        with_residual_torque: bool = False,
        with_ligament: bool = False,
        dynamics_constants_used_at_each_nodes: dict[list] = {},
    ):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau activations).
        The tau activations are bounded between -1 and 1 and actual tau is computed from torque-position-velocity
        relationship

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_residual_torque: bool
            If the dynamic with a residual torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        dynamics_constants_used_at_each_nodes: dict[np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """

        _check_contacts_in_biorbd_model(with_contact, nlp.model.nb_contacts, nlp.phase_idx)
        external_forces = None
        for key in dynamics_constants_used_at_each_nodes.keys():
            if key != "external_forces":
                raise RuntimeError(
                    "The only dynamics_constants_used_at_each_nodes allowed for torque_driven dynamics is external_forces."
                )
            _check_dynamics_constants_format(dynamics_constants_used_at_each_nodes[key], nlp.ns, nlp.phase_idx)
            external_forces = nlp.dynamics_constants[0].mx
            for i in range(1, dynamics_constants_used_at_each_nodes[key].shape[1]):
                external_forces = horzcat(external_forces, nlp.dynamics_constants[i].mx)

        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False)
        ConfigureProblem.configure_tau(ocp, nlp, False, True)

        if with_residual_torque:
            ConfigureProblem.configure_residual_tau(ocp, nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_activations_driven,
                with_contact=with_contact,
                with_passive_torque=with_passive_torque,
                with_residual_torque=with_residual_torque,
                with_ligament=with_ligament,
                external_forces=external_forces,
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp, nlp, DynamicsFunctions.forces_from_torque_activation_driven, external_forces=external_forces
            )
        ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def joints_acceleration_driven(ocp, nlp, rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE):
        """
        Configure the dynamics for a joints acceleration driven program
        (states are q and qdot, controls are qddot_joints)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used

        """
        if rigidbody_dynamics != RigidBodyDynamics.ODE:
            raise NotImplementedError("Implicit dynamics not implemented yet.")

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
        # Configure qddot joints
        nb_root = nlp.model.nb_root
        if not nb_root > 0:
            raise RuntimeError("BioModel must have at least one DoF on root.")

        name_qddot_roots = [str(i) for i in range(nb_root)]
        ConfigureProblem.configure_new_variable(
            "qddot_roots",
            name_qddot_roots,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=True,
        )

        name_qddot_joints = [str(i + nb_root) for i in range(nlp.model.nb_qddot - nb_root)]
        ConfigureProblem.configure_new_variable(
            "qddot_joints",
            name_qddot_joints,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_states_dot=True,
        )

        ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.joints_acceleration_driven)

    @staticmethod
    def muscle_driven(
        ocp,
        nlp,
        with_excitations: bool = False,
        fatigue: FatigueList = None,
        with_residual_torque: bool = False,
        with_contact: bool = False,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        dynamics_constants_used_at_each_nodes: dict[list] = {},
    ):
        """
        Configure the dynamics for a muscle driven program.
        If with_excitations is set to True, then the muscle activations are computed from the muscle dynamics.
        The tau from muscle is computed using the muscle activations.
        If with_residual_torque is set to True, then tau are used as supplementary force in the
        case muscles are too weak.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_excitations: bool
            If the dynamic should include the muscle dynamics
        fatigue: FatigueList
            The list of fatigue parameters
        with_residual_torque: bool
            If the dynamic should be added with residual torques
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        dynamics_constants_used_at_each_nodes: dict[np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """
        _check_contacts_in_biorbd_model(with_contact, nlp.model.nb_contacts, nlp.phase_idx)
        external_forces = None
        for key in dynamics_constants_used_at_each_nodes.keys():
            if key != "external_forces":
                raise RuntimeError(
                    "The only dynamics_constants_used_at_each_nodes allowed for torque_driven dynamics is external_forces."
                )
            _check_dynamics_constants_format(dynamics_constants_used_at_each_nodes[key], nlp.ns, nlp.phase_idx)
            external_forces = nlp.dynamics_constants[0].mx
            for i in range(1, dynamics_constants_used_at_each_nodes[key].shape[1]):
                external_forces = horzcat(external_forces, nlp.dynamics_constants[i].mx)
        if fatigue is not None and "tau" in fatigue and not with_residual_torque:
            raise RuntimeError("Residual torques need to be used to apply fatigue on torques")

        if rigidbody_dynamics not in (RigidBodyDynamics.DAE_INVERSE_DYNAMICS, RigidBodyDynamics.ODE):
            raise NotImplementedError("MUSCLE_DRIVEN cannot be used with this enum RigidBodyDynamics yet")

        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
        ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)

        if with_residual_torque:
            ConfigureProblem.configure_tau(ocp, nlp, False, True, fatigue=fatigue)
        ConfigureProblem.configure_muscles(ocp, nlp, with_excitations, True, fatigue=fatigue)

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            ConfigureProblem.configure_qddot(ocp, nlp, False, True)
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.TAU_FROM_MUSCLE_EQUAL_INVERSE_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
            )

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.muscles_driven,
                with_contact=with_contact,
                fatigue=fatigue,
                with_residual_torque=with_residual_torque,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                rigidbody_dynamics=rigidbody_dynamics,
                external_forces=external_forces,
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp, nlp, DynamicsFunctions.forces_from_muscle_driven, external_forces=external_forces
            )
        ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def holonomic_torque_driven(ocp, nlp):
        """
        Tell the program which variables are states and controls.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        name = "q_u"
        names_u = [nlp.model.name_dof[i] for i in nlp.variable_mappings["q"].to_first.map_idx]
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, names_u, ocp, nlp, True, False, False, axes_idx=axes_idx)

        name = "qdot_u"
        names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
        names_udot = [names_qdot[i] for i in nlp.variable_mappings["qdot"].to_first.map_idx]
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, names_udot, ocp, nlp, True, False, False, axes_idx=axes_idx)

        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.holonomic_torque_driven)

    @staticmethod
    def configure_dynamics_function(ocp, nlp, dyn_func, **extra_params):
        """
        Configure the dynamics of the system

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[time, states, controls, param, algebraic_states]
            The function to get the derivative of the states
        """

        DynamicsFunctions.apply_parameters(nlp)

        dynamics_eval = dyn_func(
            nlp.time_mx,
            nlp.states.scaled.mx_reduced,
            nlp.controls.scaled.mx_reduced,
            nlp.parameters.scaled.mx_reduced,
            nlp.algebraic_states.scaled.mx_reduced,
            nlp,
            **extra_params,
        )
        dynamics_dxdt = dynamics_eval.dxdt
        if isinstance(dynamics_dxdt, (list, tuple)):
            dynamics_dxdt = vertcat(*dynamics_dxdt)

        time_span_sym = vertcat(nlp.time_mx, nlp.dt_mx)
        if nlp.dynamics_func is None:
            nlp.dynamics_func = Function(
                "ForwardDyn",
                [
                    time_span_sym,
                    nlp.states.scaled.mx_reduced,
                    nlp.controls.scaled.mx_reduced,
                    nlp.parameters.scaled.mx_reduced,
                    nlp.algebraic_states.scaled.mx_reduced,
                    nlp.dynamics_constants.mx,
                ],
                [dynamics_dxdt],
                ["t_span", "x", "u", "p", "a", "dynamics_constants"],
                ["xdot"],
            )

            # TODO: allow expand for each dynamics independently
            if nlp.dynamics_type.expand_dynamics:
                try:
                    nlp.dynamics_func = nlp.dynamics_func.expand()
                except Exception as me:
                    RuntimeError(
                        f"An error occurred while executing the 'expand()' function for the dynamic function. "
                        f"Please review the following casadi error message for more details.\n"
                        "Several factors could be causing this issue. One of the most likely is the inability to "
                        "use expand=True at all. In that case, try adding expand=False to the dynamics.\n"
                        "Original casadi error message:\n"
                        f"{me}"
                    )

            # Only possible for regular dynamics, not for extra_dynamics
            if dynamics_eval.defects is not None:
                nlp.implicit_dynamics_func = Function(
                    "DynamicsDefects",
                    [
                        time_span_sym,
                        nlp.states.scaled.mx_reduced,
                        nlp.controls.scaled.mx_reduced,
                        nlp.parameters.scaled.mx_reduced,
                        nlp.algebraic_states.scaled.mx_reduced,
                        nlp.states_dot.scaled.mx_reduced,
                    ],
                    [dynamics_eval.defects],
                    ["t_span", "x", "u", "p", "a", "xdot"],
                    ["defects"],
                )
                if nlp.dynamics_type.expand_dynamics:
                    try:
                        nlp.implicit_dynamics_func = nlp.implicit_dynamics_func.expand()
                    except Exception as me:
                        RuntimeError(
                            f"An error occurred while executing the 'expand()' function for the dynamic function. "
                            f"Please review the following casadi error message for more details.\n"
                            "Several factors could be causing this issue. One of the most likely is the inability to "
                            "use expand=True at all. In that case, try adding expand=False to the dynamics.\n"
                            "Original casadi error message:\n"
                            f"{me}"
                        )
        else:
            nlp.extra_dynamics_func.append(
                Function(
                    "ForwardDyn",
                    [
                        time_span_sym,
                        nlp.states.scaled.mx_reduced,
                        nlp.controls.scaled.mx_reduced,
                        nlp.parameters.scaled.mx_reduced,
                        nlp.algebraic_states.scaled.mx_reduced,
                        nlp.dynamics_constants.mx,
                    ],
                    [dynamics_dxdt],
                    ["t_span", "x", "u", "p", "a", "dynamics_constants"],
                    ["xdot"],
                ),
            )

            # TODO: allow expand for each dynamics independently
            if nlp.dynamics_type.expand_dynamics:
                try:
                    nlp.extra_dynamics_func[-1] = nlp.dynamics_funcextra_dynamics_func[-1].expand()
                except Exception as me:
                    RuntimeError(
                        f"An error occurred while executing the 'expand()' function for the dynamic function. "
                        f"Please review the following casadi error message for more details.\n"
                        "Several factors could be causing this issue. One of the most likely is the inability to "
                        "use expand=True at all. In that case, try adding expand=False to the dynamics.\n"
                        "Original casadi error message:\n"
                        f"{me}"
                    )

    @staticmethod
    def configure_contact_function(ocp, nlp, dyn_func: Callable, **extra_params):
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[time, states, controls, param, algebraic_states]
            The function to get the values of contact forces from the dynamics
        """

        time_span_sym = vertcat(nlp.time_mx, nlp.dt_mx)
        nlp.contact_forces_func = Function(
            "contact_forces_func",
            [
                time_span_sym,
                nlp.states.scaled.mx_reduced,
                nlp.controls.scaled.mx_reduced,
                nlp.parameters.scaled.mx_reduced,
                nlp.algebraic_states.scaled.mx_reduced,
            ],
            [
                dyn_func(
                    time_span_sym,
                    nlp.states.scaled.mx_reduced,
                    nlp.controls.scaled.mx_reduced,
                    nlp.parameters.scaled.mx_reduced,
                    nlp.algebraic_states.scaled.mx_reduced,
                    nlp,
                    **extra_params,
                )
            ],
            ["t_span", "x", "u", "p", "a"],
            ["contact_forces"],
        ).expand()

        all_contact_names = []
        for elt in ocp.nlp:
            all_contact_names.extend([name for name in elt.model.contact_names if name not in all_contact_names])

        if "contact_forces" in nlp.plot_mapping:
            contact_names_in_phase = [name for name in nlp.model.contact_names]
            axes_idx = BiMapping(
                to_first=nlp.plot_mapping["contact_forces"].map_idx,
                to_second=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
            )
        else:
            contact_names_in_phase = [name for name in nlp.model.contact_names]
            axes_idx = BiMapping(
                to_first=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
                to_second=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
            )

        nlp.plot["contact_forces"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a: nlp.contact_forces_func(
                [t0, t0 + phases_dt[nlp.phase_idx]], x, u, p, a
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_contact_names,
        )

    @staticmethod
    def configure_soft_contact_function(ocp, nlp):
        """
        Configure the soft contact sphere

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        component_list = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]

        q = nlp.states.mx_reduced[nlp.states["q"].index]
        qdot = nlp.states.mx_reduced[nlp.states["qdot"].index]
        global_soft_contact_force_func = nlp.model.soft_contact_forces(
            nlp.states["q"].mapping.to_second.map(q), nlp.states["qdot"].mapping.to_second.map(qdot)
        )

        # TODO: do not declare unuseful functions!
        nlp.soft_contact_forces_func = Function(
            "soft_contact_forces_func",
            [
                nlp.time_mx,
                nlp.states.scaled.mx_reduced,
                nlp.controls.scaled.mx_reduced,
                nlp.parameters.scaled.mx_reduced,
            ],
            [global_soft_contact_force_func],
            ["t", "x", "u", "p"],
            ["soft_contact_forces"],
        ).expand()

        for i_sc in range(nlp.model.nb_soft_contacts):
            all_soft_contact_names = []
            all_soft_contact_names.extend(
                [
                    f"{nlp.model.soft_contact_names[i_sc]}_{name}"
                    for name in component_list
                    if nlp.model.soft_contact_names[i_sc] not in all_soft_contact_names
                ]
            )

            if "soft_contact_forces" in nlp.plot_mapping:
                soft_contact_names_in_phase = [
                    f"{nlp.model.soft_contact_names[i_sc]}_{name}"
                    for name in component_list
                    if nlp.model.soft_contact_names[i_sc] not in all_soft_contact_names
                ]
                phase_mappings = BiMapping(
                    to_first=nlp.plot_mapping["soft_contact_forces"].map_idx,
                    to_second=[i for i, c in enumerate(all_soft_contact_names) if c in soft_contact_names_in_phase],
                )
            else:
                soft_contact_names_in_phase = [
                    f"{nlp.model.soft_contact_names[i_sc]}_{name}"
                    for name in component_list
                    if nlp.model.soft_contact_names[i_sc] not in all_soft_contact_names
                ]
                phase_mappings = BiMapping(
                    to_first=[i for i, c in enumerate(all_soft_contact_names) if c in soft_contact_names_in_phase],
                    to_second=[i for i, c in enumerate(all_soft_contact_names) if c in soft_contact_names_in_phase],
                )
            nlp.plot[f"soft_contact_forces_{nlp.model.soft_contact_names[i_sc]}"] = CustomPlot(
                lambda t0, phases_dt, node_idx, x, u, p, a: nlp.soft_contact_forces_func(
                    [t0, t0 + phases_dt[nlp.phase_idx]], x, u, p, a
                )[(i_sc * 6) : ((i_sc + 1) * 6), :],
                plot_type=PlotType.INTEGRATED,
                axes_idx=phase_mappings,
                legend=all_soft_contact_names,
            )

    @staticmethod
    def configure_new_variable(
        name: str,
        name_elements: list,
        ocp,
        nlp,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        as_algebraic_states: bool = False,
        fatigue: FatigueList = None,
        combine_name: str = None,
        combine_state_control_plot: bool = False,
        skip_plot: bool = False,
        axes_idx: BiMapping = None,
    ):
        """
        Add a new variable to the states/controls pool

        Parameters
        ----------
        name: str
            The name of the new variable to add
        name_elements: list[str]
            The name of each element of the vector
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the new variable should be added to the state variable set
        as_states_dot: bool
            If the new variable should be added to the state_dot variable set
        as_controls: bool
            If the new variable should be added to the control variable set
        as_algebraic_states: bool
            If the new variable should be added to the algebraic states variable set
        fatigue: FatigueList
            The list of fatigable item
        combine_name: str
            The name of a previously added plot to combine to
        combine_state_control_plot: bool
            If states and controls plot should be combined. Only effective if as_states and as_controls are both True
        skip_plot: bool
            If no plot should be automatically added
        axes_idx: BiMapping
            The axes index to use for the plot
        """
        new_variable_config = NewVariableConfiguration(
            name,
            name_elements,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
            as_algebraic_states,
            fatigue,
            combine_name,
            combine_state_control_plot,
            skip_plot,
            axes_idx,
        )

    @staticmethod
    def configure_integrated_value(
        name: str,
        name_elements: list,
        ocp,
        nlp,
        initial_matrix: DM,
    ):
        """
        Add a new integrated value. This creates an MX (not an optimization variable) that is integrated using the
        integrated_value_functions function provided. This integrated_value can be used in the constraints and objectives
        without having to recompute them over and over again.

        Parameters
        ----------
        name: str
            The name of the new variable to add
        name_elements: list[str]
            The name of each element of the vector
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        initial_matrix: DM
            The initial value of the integrated value
        """

        # TODO: compute values at collocation points
        # but for now only cx_start can be used
        n_cx = nlp.ode_solver.n_cx - 1 if isinstance(nlp.ode_solver, OdeSolver.COLLOCATION) else 3
        if n_cx < 3:
            n_cx = 3

        dummy_mapping = Mapping(list(range(len(name_elements))))
        initial_vector = StochasticBioModel.reshape_to_vector(initial_matrix)
        cx_scaled_next_formatted = [initial_vector for _ in range(n_cx)]
        nlp.integrated_values.append(
            name, cx_scaled_next_formatted, cx_scaled_next_formatted, initial_matrix, dummy_mapping, 0
        )
        for node_index in range(1, nlp.ns + 1):  # cannot use phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
            cx_scaled_next = nlp.integrated_value_functions[name](nlp, node_index)
            cx_scaled_next_formatted = [cx_scaled_next for _ in range(n_cx)]
            nlp.integrated_values.append(
                name,
                cx_scaled_next_formatted,
                cx_scaled_next_formatted,
                cx_scaled_next,
                dummy_mapping,
                node_index,
            )

    @staticmethod
    def configure_q(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized coordinates

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """
        name = "q"
        name_q = nlp.model.name_dof
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_q, ocp, nlp, as_states, as_controls, as_states_dot, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qdot(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized velocities

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """

        name = "qdot"
        name_qdot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_qdot, ocp, nlp, as_states, as_controls, as_states_dot, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qddot(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized accelerations

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        as_states_dot: bool
            If the generalized accelerations should be a state_dot
        """

        name = "qddot"
        name_qddot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_qddot, ocp, nlp, as_states, as_controls, as_states_dot, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qdddot(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized accelerations

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        """

        name = "qdddot"
        name_qdddot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, name_qdddot, ocp, nlp, as_states, as_controls, axes_idx=axes_idx)

    @staticmethod
    def configure_stochastic_k(ocp, nlp, n_noised_controls: int, n_references: int):
        """
        Configure the optimal feedback gain matrix K.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "k"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_k = []
        control_names = [f"control_{i}" for i in range(n_noised_controls)]
        ref_names = [f"feedback_{i}" for i in range(n_references)]
        for name_1 in control_names:
            for name_2 in ref_names:
                name_k += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(
            list(range(len(control_names) * len(ref_names))), list(range(len(control_names) * len(ref_names)))
        )
        ConfigureProblem.configure_new_variable(
            name,
            name_k,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=False,
            as_algebraic_states=True,
        )

    @staticmethod
    def configure_stochastic_c(ocp, nlp, n_noised_states: int, n_noise: int):
        """
        Configure the stochastic variable matrix C representing the injection of motor noise (df/dw).
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "c"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states variables and mapping cannot be use together for now.")

        name_c = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noise)]:
                name_c += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(
            list(range(n_noised_states * n_noise)), list(range(n_noised_states * n_noise))
        )

        ConfigureProblem.configure_new_variable(
            name,
            name_c,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=False,
            as_algebraic_states=True,
            skip_plot=True,
        )

    @staticmethod
    def configure_stochastic_a(ocp, nlp, n_noised_states: int):
        """
        Configure the stochastic variable matrix A representing the propagation of motor noise (df/dx).
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "a"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_a = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_a += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states**2)), list(range(n_noised_states**2)))

        ConfigureProblem.configure_new_variable(
            name,
            name_a,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=False,
            as_algebraic_states=True,
            skip_plot=True,
        )

    @staticmethod
    def configure_stochastic_cov_explicit(ocp, nlp, n_noised_states: int, initial_matrix: DM):
        """
        Configure the covariance matrix P representing the motor noise.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "cov"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_cov = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_cov += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states**2)), list(range(n_noised_states**2)))
        ConfigureProblem.configure_integrated_value(
            name,
            name_cov,
            ocp,
            nlp,
            initial_matrix=initial_matrix,
        )

    @staticmethod
    def configure_stochastic_cov_implicit(ocp, nlp, n_noised_states: int):
        """
        Configure the covariance matrix P representing the motor noise.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "cov"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_cov = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_cov += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states**2)), list(range(n_noised_states**2)))
        ConfigureProblem.configure_new_variable(
            name,
            name_cov,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=False,
            as_algebraic_states=True,
        )

    @staticmethod
    def configure_stochastic_cholesky_cov(ocp, nlp, n_noised_states: int):
        """
        Configure the diagonal matrix needed to reconstruct the covariance matrix using L @ L.T.
        This formulation allows insuring that the covariance matrix is always positive semi-definite.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "cholesky_cov"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_cov = []
        for nb_1, name_1 in enumerate([f"X_{i}" for i in range(n_noised_states)]):
            for name_2 in [f"X_{i}" for i in range(nb_1 + 1)]:
                name_cov += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(len(name_cov))), list(range(len(name_cov))))
        ConfigureProblem.configure_new_variable(
            name,
            name_cov,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=False,
            as_algebraic_states=True,
        )

    @staticmethod
    def configure_stochastic_ref(ocp, nlp, n_references: int):
        """
        Configure the reference kinematics.

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "ref"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_ref = [f"reference_{i}" for i in range(n_references)]
        nlp.variable_mappings[name] = BiMapping(list(range(n_references)), list(range(n_references)))
        ConfigureProblem.configure_new_variable(
            name,
            name_ref,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=False,
            as_algebraic_states=True,
        )

    @staticmethod
    def configure_stochastic_m(ocp, nlp, n_noised_states: int, n_collocation_points: int = 1):
        """
        Configure the helper matrix M (from Gillis 2013 : https://doi.org/10.1109/CDC.2013.6761121).

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "m"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_m = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states * n_collocation_points)]:
                name_m += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(
            list(range(n_noised_states * n_noised_states * n_collocation_points)),
            list(range(n_noised_states * n_noised_states * n_collocation_points)),
        )
        ConfigureProblem.configure_new_variable(
            name,
            name_m,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_states_dot=False,
            as_algebraic_states=True,
        )

    @staticmethod
    def configure_tau(ocp, nlp, as_states: bool, as_controls: bool, fatigue: FatigueList = None):
        """
        Configure the generalized forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized forces should be a state
        as_controls: bool
            If the generalized forces should be a control
        fatigue: FatigueList
            If the dynamics with fatigue should be declared
        """

        name = "tau"
        name_tau = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_tau, ocp, nlp, as_states, as_controls, fatigue=fatigue, axes_idx=axes_idx
        )

    @staticmethod
    def configure_residual_tau(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the residual forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized forces should be a state
        as_controls: bool
            If the generalized forces should be a control
        """

        name = "residual_tau"
        name_residual_tau = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_residual_tau, ocp, nlp, as_states, as_controls, axes_idx=axes_idx
        )

    @staticmethod
    def configure_taudot(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """

        name = "taudot"
        name_taudot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, name_taudot, ocp, nlp, as_states, as_controls, axes_idx=axes_idx)

    @staticmethod
    def configure_contact_forces(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """

        name_contact_forces = [name for name in nlp.model.contact_names]
        ConfigureProblem.configure_new_variable("fext", name_contact_forces, ocp, nlp, as_states, as_controls)

    @staticmethod
    def configure_soft_contact_forces(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """
        name_soft_contact_forces = []
        component_list = ["fx", "fy", "fz"]  # TODO: find a better place to hold this or define it in biorbd ?
        for ii in range(nlp.model.nb_soft_contacts):
            name_soft_contact_forces.extend(
                [
                    f"{nlp.model.soft_contact_name(ii)}_{name}"
                    for name in component_list
                    if nlp.model.soft_contact_name(ii) not in name_soft_contact_forces
                ]
            )
        ConfigureProblem.configure_new_variable("fext", name_soft_contact_forces, ocp, nlp, as_states, as_controls)

    @staticmethod
    def configure_muscles(ocp, nlp, as_states: bool, as_controls: bool, fatigue: FatigueList = None):
        """
        Configure the muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the muscles should be a state
        as_controls: bool
            If the muscles should be a control
        fatigue: FatigueList
            The list of fatigue parameters
        """

        muscle_names = nlp.model.muscle_names
        ConfigureProblem.configure_new_variable(
            "muscles",
            muscle_names,
            ocp,
            nlp,
            as_states,
            as_controls,
            combine_state_control_plot=True,
            fatigue=fatigue,
        )

    @staticmethod
    def _apply_phase_mapping(ocp, nlp, name: str) -> BiMapping | None:
        """
        Apply the phase mapping to the variable

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        name: str
            The name of the variable to map

        Returns
        -------
        The mapping or None if no mapping is defined

        """
        if nlp.phase_mapping:
            if name in nlp.variable_mappings.keys():
                double_mapping_to_first = (
                    nlp.variable_mappings[name].to_first.map(nlp.phase_mapping.to_first.map_idx).T.tolist()[0]
                )
                double_mapping_to_first = [int(double_mapping_to_first[i]) for i in range(len(double_mapping_to_first))]
                double_mapping_to_second = (
                    nlp.variable_mappings[name].to_second.map(nlp.phase_mapping.to_second.map_idx).T.tolist()[0]
                )
                double_mapping_to_second = [
                    int(double_mapping_to_second[i]) for i in range(len(double_mapping_to_second))
                ]
            else:
                double_mapping_to_first = nlp.phase_mapping.to_first.map_idx
                double_mapping_to_second = nlp.phase_mapping.to_second.map_idx
            axes_idx = BiMapping(to_first=double_mapping_to_first, to_second=double_mapping_to_second)
        else:
            axes_idx = None
        return axes_idx


class DynamicsFcn(FcnEnum):
    """
    Selection of valid dynamics functions
    """

    TORQUE_DRIVEN = (ConfigureProblem.torque_driven,)
    TORQUE_DRIVEN_FREE_FLOATING_BASE = (ConfigureProblem.torque_driven_free_floating_base,)
    STOCHASTIC_TORQUE_DRIVEN = (ConfigureProblem.stochastic_torque_driven,)
    STOCHASTIC_TORQUE_DRIVEN_FREE_FLOATING_BASE = (ConfigureProblem.stochastic_torque_driven_free_floating_base,)
    TORQUE_DERIVATIVE_DRIVEN = (ConfigureProblem.torque_derivative_driven,)
    TORQUE_ACTIVATIONS_DRIVEN = (ConfigureProblem.torque_activations_driven,)
    JOINTS_ACCELERATION_DRIVEN = (ConfigureProblem.joints_acceleration_driven,)
    MUSCLE_DRIVEN = (ConfigureProblem.muscle_driven,)
    HOLONOMIC_TORQUE_DRIVEN = (ConfigureProblem.holonomic_torque_driven,)
    CUSTOM = (ConfigureProblem.custom,)


class Dynamics(OptionGeneric):
    """
    A placeholder for the chosen dynamics by the user

    Attributes
    ----------
    dynamic_function: Callable
        The custom dynamic function provided by the user
    configure: Callable
        The configuration function provided by the user that declares the NLP (states and controls),
        usually only necessary when defining custom functions
    expand_dynamics: bool
        If the dynamics function should be expanded
    expand_continuity: bool
        If the continuity function should be expanded. This can be extensive on the RAM usage
    skip_continuity: bool
        If the continuity should be skipped
    state_continuity_weight: float | None
        The weight of the continuity constraint. If None, the continuity is a constraint,
        otherwise it is an objective
    phase_dynamics: PhaseDynamics
        If the dynamics should be shared between the nodes or not
    """

    def __init__(
        self,
        dynamics_type: Callable | DynamicsFcn,
        expand_dynamics: bool = True,
        expand_continuity: bool = False,
        skip_continuity: bool = False,
        state_continuity_weight: float | None = None,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        **extra_parameters: Any,
    ):
        """
        Parameters
        ----------
        dynamics_type: Callable | DynamicsFcn
            The chosen dynamic functions
        params: Any
            Any parameters to pass to the dynamic and configure functions
        expand_dynamics: bool
            If the dynamics function should be expanded
        expand_continuity: bool
            If the continuity function should be expanded. This can be extensive on the RAM usage
        skip_continuity: bool
            If the continuity should be skipped
        state_continuity_weight: float | None
            The weight of the continuity constraint. If None, the continuity is a constraint,
            otherwise it is an objective
        phase_dynamics: PhaseDynamics
            If the dynamics should be shared between the nodes or not
        """

        configure = None
        if not isinstance(dynamics_type, DynamicsFcn):
            configure = dynamics_type
            dynamics_type = DynamicsFcn.CUSTOM
        else:
            if "configure" in extra_parameters:
                configure = extra_parameters["configure"]
                del extra_parameters["configure"]

        dynamic_function = None
        if "dynamic_function" in extra_parameters:
            dynamic_function = extra_parameters["dynamic_function"]
            del extra_parameters["dynamic_function"]

        super(Dynamics, self).__init__(type=dynamics_type, **extra_parameters)
        self.dynamic_function = dynamic_function
        self.configure = configure
        self.expand_dynamics = expand_dynamics
        self.expand_continuity = expand_continuity
        self.skip_continuity = skip_continuity
        self.state_continuity_weight = state_continuity_weight
        self.phase_dynamics = phase_dynamics


class DynamicsList(UniquePerPhaseOptionList):
    """
    A list of Dynamics if more than one is required, typically when more than one phases are declared

    Methods
    -------
    add(dynamics: DynamicsFcn, **extra_parameters)
        Add a new Dynamics to the list
    print(self)
        Print the DynamicsList to the console
    """

    def add(self, dynamics_type: Callable | Dynamics | DynamicsFcn, **extra_parameters: Any):
        """
        Add a new Dynamics to the list

        Parameters
        ----------
        dynamics_type: Callable | Dynamics | DynamicsFcn
            The chosen dynamic functions
        extra_parameters: dict
            Any parameters to pass to Dynamics
        """

        if isinstance(dynamics_type, Dynamics):
            self.copy(dynamics_type)

        else:
            super(DynamicsList, self)._add(dynamics_type=dynamics_type, option_type=Dynamics, **extra_parameters)

    def print(self):
        """
        Print the DynamicsList to the console
        """
        raise NotImplementedError("Printing of DynamicsList is not ready yet")


def _check_dynamics_constants_format(dynamics_constant: np.ndarray, n_shooting: int, phase_idx: int):
    """Check if the dynamics_constant_at_each_node is of the right format"""
    if type(dynamics_constant) is not np.ndarray:
        raise RuntimeError(
            f"Phase {phase_idx} has dynamics_constant_at_each_node of type {type(dynamics_constant)} "
            f"but it should be of type np.ndarray"
        )
    if dynamics_constant is not None and dynamics_constant.shape[2] != n_shooting + 1:
        raise RuntimeError(
            f"Phase {phase_idx} has {n_shooting}+1 shooting points but the dynamics_constant_at_each_node "
            f"has {dynamics_constant.shape[2]} shooting points."
            f"The dynamics_constant_at_each_node should be of format dict[np.ndarray] "
            f"where the list is the number of shooting points of the phase "
        )


def _check_soft_contacts_dynamics(
    rigidbody_dynamics: RigidBodyDynamics,
    soft_contacts_dynamics: SoftContactDynamics,
    nb_soft_contacts,
    phase_idx: int,
):
    if nb_soft_contacts != 0:
        if (
            soft_contacts_dynamics != SoftContactDynamics.CONSTRAINT
            and soft_contacts_dynamics != SoftContactDynamics.ODE
        ):
            raise ValueError(
                f"Phase {phase_idx} has soft contacts but the soft_contacts_dynamics is not "
                f"SoftContactDynamics.CONSTRAINT or SoftContactDynamics.ODE."
            )

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            if soft_contacts_dynamics == SoftContactDynamics.ODE:
                raise ValueError(
                    f"Phase {phase_idx} has soft contacts but the rigidbody_dynamics is "
                    f"RigidBodyDynamics.DAE_INVERSE_DYNAMICS and soft_contacts_dynamics is SoftContactDynamics.ODE."
                    f"Please set soft_contacts_dynamics=SoftContactDynamics.CONSTRAINT"
                )


def _check_contacts_in_biorbd_model(with_contact: bool, nb_contacts: int, phase_idx: int):
    if with_contact and nb_contacts == 0:
        raise ValueError(f"No contact defined in the .bioMod of phase {phase_idx}, set with_contact to False.")
