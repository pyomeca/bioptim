from typing import Callable, Any

import numpy as np
from casadi import vertcat, Function, DM

from .dynamics_functions import DynamicsFunctions
from .fatigue.fatigue_dynamics import FatigueList
from .ode_solvers import OdeSolver, OdeSolverBase
from ..gui.plot import CustomPlot
from ..limits.constraints import ImplicitConstraintFcn
from ..misc.enums import (
    PlotType,
    Node,
    ConstraintType,
    SoftContactDynamics,
    PhaseDynamics,
    ContactType,
    ControlType,
)
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..models.protocols.biomodel import BioModel
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
    torque_driven
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)
    torque_derivative_driven
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)
    torque_activations_driven
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau activations).
        The tau activations are bounded between -1 and 1 and actual tau is computed from torque-position-velocity
        relationship
    muscle_driven
        Configure the dynamics for a muscle driven program.
        If with_excitations is set to True, then the muscle muscle activations are computed from the muscle dynamics.
        The tau from muscle is computed using the muscle activations.
        If with_residual_torque is set to True, then tau are used as supplementary force in the
        case muscles are too weak.
    configure_dynamics_function(ocp, nlp, dyn_func, **extra_params)
        Configure the dynamics of the system
    configure_rigid_contact_function(ocp, nlp, dyn_func: Callable, **extra_params)
        Configure the rigid contact points
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
        # nlp.dynamics_type.type(
        #     ocp,
        #     nlp,
        #     numerical_data_timeseries=nlp.dynamics_type.numerical_data_timeseries,
        #     contact_type=nlp.dynamics_type.contact_type,
        #     **nlp.dynamics_type.extra_parameters,
        # )
        nlp.dynamics_type.configure.initialize(
            ocp,
            nlp,
            numerical_data_timeseries=nlp.dynamics_type.numerical_data_timeseries,
            contact_type=nlp.dynamics_type.contact_type,
            **nlp.dynamics_type.extra_parameters,
        )
        ConfigureProblem.initialize_dynamics(
            ocp,
            nlp,
            numerical_data_timeseries=nlp.dynamics_type.numerical_data_timeseries,
            contact_type=nlp.dynamics_type.contact_type,
            **nlp.dynamics_type.extra_parameters,
        )

    @staticmethod
    def initialize_dynamics(
            ocp,
            nlp,
            numerical_data_timeseries,
            contact_type,
            **extra_parameters,
        ):

        # Collect variables
        q = nlp.get_var_from_states_or_controls("q", nlp.states, nlp.controls)  # ...TODO
        qdot = nlp.get_var_from_states_or_controls("qdot", nlp.states, nlp.controls)
        tau = nlp.get_var_from_states_or_controls("tau", nlp.states, nlp.controls)
        # get all the other variables (None is doe not exists)
        ConfigureProblem.check_variables(q, qdot, tau)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        # TODO: def collect_fext():
        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)

        tau = ConfigureProblem.collect_tau(nlp, tau)

        if fatigue is not None and "tau" in fatigue:
            dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        # TODO: if muscles, ...
        # TODO: if taudot, ...
        # ... append dynamics and defects accordingly

    @staticmethod
    def check_variables(
            q,
            qdot,
            tau,
    ):
        if q is None or qdot is None:
            raise NotImplementedError("All of bioptim's dynamics require q and qdot to be defined")
        # TODO: check combinations, ...

    @staticmethod
    def collect_tau(nlp, tau, ...):
        if tau is not None:
            tau = DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue)
            tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
            tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau
            tau = tau - nlp.model.friction_coefficients @ qdot if with_friction else tau
        return tau


    @staticmethod
    def custom(ocp, nlp, **extra_params):
        """
        Call the user-defined dynamics configuration function

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.configure(ocp, nlp, **extra_params)

    @staticmethod
    def torque_driven(
        ocp,
        nlp,
        contact_type: list[ContactType] | tuple[ContactType] = (),
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
        soft_contacts_dynamics: SoftContactDynamics = SoftContactDynamics.ODE,
        fatigue: FatigueList = None,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficients * qdot)
        soft_contacts_dynamics: SoftContactDynamics
            which soft contact dynamic should be used
        fatigue: FatigueList
            A list of fatigue elements
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """

        _check_contacts_in_biomodel(contact_type, nlp.model, nlp.phase_idx)

        # Declared rigidbody states and controls
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True, fatigue=fatigue)
        ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False)

        # Declared soft contacts controls
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ConfigureProblem.configure_soft_contact_forces(ocp, nlp, as_states=False, as_controls=True)

        # Configure the actual ODE of the dynamics
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_driven,
                contact_type=contact_type,
                fatigue=fatigue,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                with_friction=with_friction,
            )

        # Configure the contact forces
        if ContactType.RIGID_EXPLICIT in contact_type:
            ConfigureProblem.configure_rigid_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

        # Configure the soft contact forces
        if ContactType.SOFT_EXPLICIT in contact_type:
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
        contact_type: list[ContactType] | tuple[ContactType],
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
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
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficients * qdot)
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node.
        """
        if len(contact_type) > 0:
            raise RuntimeError("free floating base dynamics cannot be used with contacts by definition.")

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
        )

        name_q_joints = [str(i) for i in range(nb_root, nb_q)]
        ConfigureProblem.configure_new_variable(
            "q_joints",
            name_q_joints,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
        )

        ConfigureProblem.configure_new_variable(
            "qdot_roots",
            name_q_roots,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
        )

        name_qdot_joints = [str(i) for i in range(nb_root, nb_qdot)]
        ConfigureProblem.configure_new_variable(
            "qdot_joints",
            name_qdot_joints,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
        )

        ConfigureProblem.configure_new_variable(
            "tau_joints",
            name_qdot_joints,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
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
        contact_type: list[ContactType] | tuple[ContactType] = (),
        with_friction: bool = False,
        with_cholesky: bool = False,
        initial_matrix: DM = None,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Configure the dynamics for a torque driven stochastic program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficient * qdot)
        with_cholesky: bool
            If the Cholesky decomposition should be used for the covariance matrix.
        initial_matrix: DM
            The initial value for the covariance matrix
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
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
        ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=n_noised_states)

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
            contact_type=contact_type,
            with_friction=with_friction,
        )

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            DynamicsFunctions.stochastic_torque_driven,
            contact_type=contact_type,
            with_friction=with_friction,
        )

    @staticmethod
    def stochastic_torque_driven_free_floating_base(
        ocp,
        nlp,
        problem_type,
        with_friction: bool = False,
        with_cholesky: bool = False,
        initial_matrix: DM = None,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: list[ContactType] | tuple[ContactType] = (),
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
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficient * qdot)
        with_cholesky: bool
            If the Cholesky decomposition should be used for the covariance matrix.
        initial_matrix: DM
            The initial value for the covariance matrix
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node.
        """
        if len(contact_type) > 0:
            raise RuntimeError("free floating base dynamics cannot be used with contacts by definition.")

        n_noised_tau = nlp.model.n_noised_controls
        n_noise = nlp.model.motor_noise_magnitude.shape[0] + nlp.model.sensory_noise_magnitude.shape[0]
        n_noised_states = nlp.model.n_noised_states

        # Stochastic variables
        ConfigureProblem.configure_stochastic_k(
            ocp, nlp, n_noised_controls=n_noised_tau, n_references=nlp.model.n_references
        )
        ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=nlp.model.n_references)
        ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=n_noised_states)

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
        contact_type: list[ContactType] | tuple[ContactType] = (),
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
        soft_contacts_dynamics: SoftContactDynamics = SoftContactDynamics.ODE,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = - coefficient * qdot)
        soft_contacts_dynamics: SoftContactDynamics
            which soft contact dynamic should be used
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.

        """
        _check_contacts_in_biomodel(contact_type, nlp.model, nlp.phase_idx)

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_taudot(ocp, nlp, as_states=False, as_controls=True)

        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ConfigureProblem.configure_soft_contact_forces(ocp, nlp, as_states=False, as_controls=True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_derivative_driven,
                contact_type=contact_type,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                with_friction=with_friction,
            )

        if ContactType.RIGID_EXPLICIT in contact_type:
            ConfigureProblem.configure_rigid_contact_function(
                ocp,
                nlp,
                DynamicsFunctions.forces_from_torque_driven,
            )

        if ContactType.SOFT_EXPLICIT in contact_type:
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
        contact_type: list[ContactType] | tuple[ContactType] = (),
        with_passive_torque: bool = False,
        with_residual_torque: bool = False,
        with_ligament: bool = False,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
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
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_residual_torque: bool
            If the dynamic with a residual torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """

        _check_contacts_in_biomodel(contact_type, nlp.model, nlp.phase_idx)

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

        if with_residual_torque:
            ConfigureProblem.configure_residual_tau(ocp, nlp, as_states=False, as_controls=True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_activations_driven,
                contact_type=contact_type,
                with_passive_torque=with_passive_torque,
                with_residual_torque=with_residual_torque,
                with_ligament=with_ligament,
            )

        if ContactType.RIGID_EXPLICIT in contact_type:
            ConfigureProblem.configure_rigid_contact_function(
                ocp, nlp, DynamicsFunctions.forces_from_torque_activation_driven
            )

        if ContactType.SOFT_EXPLICIT in contact_type:
            ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def joints_acceleration_driven(
        ocp,
        nlp,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: list[ContactType] | tuple[ContactType] = (),
    ):
        """
        Configure the dynamics for a joints acceleration driven program
        (states are q and qdot, controls are qddot_joints)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """

        if len(contact_type) > 0:
            raise RuntimeError("joints acceleration driven dynamics cannot be used with contacts by definition.")

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
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
        )

        name_qddot_joints = [str(i + nb_root) for i in range(nlp.model.nb_qddot - nb_root)]
        ConfigureProblem.configure_new_variable(
            "qddot_joints",
            name_qddot_joints,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
        )

        ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.joints_acceleration_driven)

    @staticmethod
    def muscle_driven(
        ocp,
        nlp,
        with_excitations: bool = False,
        fatigue: FatigueList = None,
        with_residual_torque: bool = False,
        contact_type: list[ContactType] | tuple[ContactType] = (),
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
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
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficients * qdot)
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """
        _check_contacts_in_biomodel(contact_type, nlp.model, nlp.phase_idx)

        if fatigue is not None and "tau" in fatigue and not with_residual_torque:
            raise RuntimeError("Residual torques need to be used to apply fatigue on torques")

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False)

        if with_residual_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True, fatigue=fatigue)
        ConfigureProblem.configure_muscles(ocp, nlp, with_excitations, as_controls=True, fatigue=fatigue)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.muscles_driven,
                contact_type=contact_type,
                fatigue=fatigue,
                with_residual_torque=with_residual_torque,
                with_passive_torque=with_passive_torque,
                with_ligament=with_ligament,
                with_friction=with_friction,
            )

        if ContactType.RIGID_EXPLICIT in contact_type:
            ConfigureProblem.configure_rigid_contact_function(
                ocp,
                nlp,
                DynamicsFunctions.forces_from_muscle_driven,
            )

        if ContactType.SOFT_EXPLICIT in contact_type:
            ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def holonomic_torque_driven(
        ocp,
        nlp,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: list[ContactType] | tuple[ContactType] = (),
    ):
        """
        Tell the program which variables are states and controls.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node.
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics.
        """

        name = "q_u"
        names_u = [nlp.model.name_dof[i] for i in nlp.model.independent_joint_index]
        ConfigureProblem.configure_new_variable(
            name,
            names_u,
            ocp,
            nlp,
            True,
            False,
            False,
            # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
            # see _set_kinematic_phase_mapping method
            # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
        )

        name = "qdot_u"
        names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
        names_udot = [names_qdot[i] for i in nlp.model.independent_joint_index]
        ConfigureProblem.configure_new_variable(
            name,
            names_udot,
            ocp,
            nlp,
            True,
            False,
            False,
            # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
            # see _set_kinematic_phase_mapping method
            # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
        )

        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

        # extra plots
        ConfigureProblem.configure_qv(ocp, nlp, nlp.model.compute_q_v)
        ConfigureProblem.configure_qdotv(ocp, nlp, nlp.model._compute_qdot_v)
        ConfigureProblem.configure_lagrange_multipliers_function(ocp, nlp, nlp.model.compute_the_lagrangian_multipliers)

        ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.holonomic_torque_driven)

    @staticmethod
    def configure_lagrange_multipliers_function(ocp, nlp, dyn_func: Callable, **extra_params):
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[time, states, controls, param, algebraic_states, numerical_timeseries]
            The function to get the values of contact forces from the dynamics
        """

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.lagrange_multipliers_function = Function(
            "lagrange_multipliers_function",
            [
                time_span_sym,
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.scaled.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [
                dyn_func()(
                    nlp.get_var_from_states_or_controls("q_u", nlp.states.scaled.cx, nlp.controls.scaled.cx),
                    nlp.get_var_from_states_or_controls("qdot_u", nlp.states.scaled.cx, nlp.controls.scaled.cx),
                    DM.zeros(nlp.model.nb_dependent_joints, 1),
                    DynamicsFunctions.get(nlp.controls["tau"], nlp.controls.scaled.cx),
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["lagrange_multipliers"],
        )

        all_multipliers_names = []
        for nlp_i in ocp.nlp:
            if hasattr(nlp_i.model, "has_holonomic_constraints"):  # making sure we have a HolonomicBiorbdModel
                nlp_i_multipliers_names = [nlp_i.model.name_dof[i] for i in nlp_i.model.dependent_joint_index]
                all_multipliers_names.extend(
                    [name for name in nlp_i_multipliers_names if name not in all_multipliers_names]
                )

        all_multipliers_names = [f"lagrange_multiplier_{name}" for name in all_multipliers_names]
        all_multipliers_names_in_phase = [
            f"lagrange_multiplier_{nlp.model.name_dof[i]}" for i in nlp.model.dependent_joint_index
        ]

        axes_idx = BiMapping(
            to_first=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
            to_second=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
        )

        nlp.plot["lagrange_multipliers"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.lagrange_multipliers_function(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_multipliers_names,
        )

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
        dyn_func: Callable[time, states, controls, param, algebraic_states, numerical_timeseries]
            The function to get the derivative of the states
        """

        dynamics_eval = dyn_func(
            nlp.time_cx,
            nlp.states.scaled.cx,
            nlp.controls.scaled.cx,
            nlp.parameters.scaled.cx,
            nlp.algebraic_states.scaled.cx,
            nlp.numerical_timeseries.cx,
            nlp,
            **extra_params,
        )
        dynamics_dxdt = dynamics_eval.dxdt
        if isinstance(dynamics_dxdt, (list, tuple)):
            dynamics_dxdt = vertcat(*dynamics_dxdt)

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        if nlp.dynamics_func is None:
            nlp.dynamics_func = Function(
                "ForwardDyn",
                [
                    time_span_sym,
                    nlp.states.scaled.cx,
                    nlp.controls.scaled.cx,
                    nlp.parameters.scaled.cx,
                    nlp.algebraic_states.scaled.cx,
                    nlp.numerical_timeseries.cx,
                ],
                [dynamics_dxdt],
                ["t_span", "x", "u", "p", "a", "d"],
                ["xdot"],
            )

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

            if dynamics_eval.defects is not None:
                nlp.implicit_dynamics_func = Function(
                    "DynamicsDefects",
                    [
                        time_span_sym,
                        nlp.states.scaled.cx,
                        nlp.controls.scaled.cx,
                        nlp.parameters.scaled.cx,
                        nlp.algebraic_states.scaled.cx,
                        nlp.numerical_timeseries.cx,
                        nlp.states_dot.scaled.cx,
                    ],
                    [dynamics_eval.defects],
                    ["t_span", "x", "u", "p", "a", "d", "xdot"],
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
                        nlp.states.scaled.cx,
                        nlp.controls.scaled.cx,
                        nlp.parameters.scaled.cx,
                        nlp.algebraic_states.scaled.cx,
                        nlp.numerical_timeseries.cx,
                    ],
                    [dynamics_dxdt],
                    ["t_span", "x", "u", "p", "a", "d"],
                    ["xdot"],
                ),
            )

            if nlp.dynamics_type.expand_dynamics:
                try:
                    nlp.extra_dynamics_func[-1] = nlp.extra_dynamics_func[-1].expand()
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
    def configure_rigid_contact_function(ocp, nlp, contact_func: Callable, **extra_params):
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        contact_func: Callable[time, states, controls, param, algebraic_states, numerical_timeseries]
            The function to get the values of contact forces from the dynamics
        """

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.rigid_contact_forces_func = Function(
            "rigid_contact_forces_func",
            [
                time_span_sym,
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.scaled.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [
                contact_func(
                    time_span_sym,
                    nlp.states.scaled.cx,
                    nlp.controls.scaled.cx,
                    nlp.parameters.scaled.cx,
                    nlp.algebraic_states.scaled.cx,
                    nlp.numerical_timeseries.cx,
                    nlp,
                    **extra_params,
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["rigid_contact_forces"],
        ).expand()

        all_contact_names = []
        for elt in ocp.nlp:
            all_contact_names.extend([name for name in elt.model.contact_names if name not in all_contact_names])

        if "rigid_contact_forces" in nlp.plot_mapping:
            contact_names_in_phase = [name for name in nlp.model.contact_names]
            axes_idx = BiMapping(
                to_first=nlp.plot_mapping["rigid_contact_forces"].map_idx,
                to_second=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
            )
        else:
            contact_names_in_phase = [name for name in nlp.model.contact_names]
            axes_idx = BiMapping(
                to_first=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
                to_second=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
            )

        nlp.plot["rigid_contact_forces"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.rigid_contact_forces_func(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
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

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.soft_contact_forces_func = Function(
            "soft_contact_forces_func",
            [
                time_span_sym,
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.scaled.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [nlp.model.soft_contact_forces()(nlp.states["q"].cx, nlp.states["qdot"].cx, nlp.parameters.cx)],
            ["t_span", "x", "u", "p", "a", "d"],
            ["soft_contact_forces"],
        ).expand()

        component_list = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]

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
                lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.soft_contact_forces_func(
                    np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
                )[(i_sc * 6) : ((i_sc + 1) * 6), :],
                plot_type=PlotType.INTEGRATED,
                axes_idx=phase_mappings,
                legend=all_soft_contact_names,
            )


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
        configure: Callable | "AutoConfigure",
        expand_dynamics: bool = True,
        expand_continuity: bool = False,
        skip_continuity: bool = False,
        state_continuity_weight: float | None = None,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver: OdeSolver | OdeSolverBase = OdeSolver.RK4(),
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: list[ContactType] | tuple[ContactType] = (),
        control_type: ControlType = ControlType.CONSTANT,
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
        ode_solver: OdeSolver
            The integrator to use to integrate this dynamics.
        numerical_data_timeseries: dict[str, np.ndarray]
            The numerical timeseries at each node. ex: the experimental external forces data should go here.
        contact_type: list[ContactType] | tuple[ContactType]
            The type of contact to consider in the dynamics
        """

        # configure = None
        # if not isinstance(dynamics_type, DynamicsFcn):
        #     configure = dynamics_type
        #     dynamics_type = DynamicsFcn.CUSTOM
        # else:
        #     if "configure" in extra_parameters:
        #         configure = extra_parameters["configure"]
        #         del extra_parameters["configure"]

        dynamic_function = None
        if "dynamic_function" in extra_parameters:
            dynamic_function = extra_parameters["dynamic_function"]
            del extra_parameters["dynamic_function"]

        if not isinstance(ode_solver, OdeSolverBase):
            raise RuntimeError("ode_solver should be built an instance of OdeSolver")

        super(Dynamics, self).__init__(type=configure, **extra_parameters)
        self.dynamic_function = dynamic_function
        self.configure = configure
        self.expand_dynamics = expand_dynamics
        self.expand_continuity = expand_continuity
        self.skip_continuity = skip_continuity
        self.state_continuity_weight = state_continuity_weight
        self.phase_dynamics = phase_dynamics
        self.ode_solver = ode_solver
        self.numerical_data_timeseries = numerical_data_timeseries
        self.contact_type = contact_type


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

    def add(self, dynamics_type: Callable, **extra_parameters: Any):
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


def _check_numerical_timeseries_format(numerical_timeseries: np.ndarray, n_shooting: int, phase_idx: int):
    """Check if the numerical_data_timeseries is of the right format"""
    if type(numerical_timeseries) is not np.ndarray:
        raise RuntimeError(
            f"Phase {phase_idx} has numerical_data_timeseries of type {type(numerical_timeseries)} "
            f"but it should be of type np.ndarray"
        )
    if numerical_timeseries is not None and numerical_timeseries.shape[2] != n_shooting + 1:
        raise RuntimeError(
            f"Phase {phase_idx} has {n_shooting}+1 shooting points but the numerical_data_timeseries "
            f"has {numerical_timeseries.shape[2]} shooting points."
            f"The numerical_data_timeseries should be of format dict[str, np.ndarray] "
            f"where the list is the number of shooting points of the phase "
        )


def _check_contacts_in_biomodel(contact_type: list[ContactType] | tuple[ContactType], model: BioModel, phase_idx: int):

    # Check rigid contacts
    if (
        ContactType.RIGID_EXPLICIT in contact_type or ContactType.RIGID_IMPLICIT in contact_type
    ) and model.nb_contacts == 0:
        raise ValueError(
            f"No rigid contact defined in the .bioMod of phase {phase_idx}, consider changing the ContactType."
        )

    # Check soft contacts
    if (
        ContactType.SOFT_EXPLICIT in contact_type or ContactType.SOFT_IMPLICIT in contact_type
    ) and model.nb_soft_contacts == 0:
        raise ValueError(
            f"No soft contact defined in the .bioMod of phase {phase_idx}, consider changing the ContactType."
        )

    # Check that contact types are not declared at the same time
    if len(contact_type) > 1:
        raise NotImplementedError("It is not possible to use multiple ContactType at the same time yet.")

