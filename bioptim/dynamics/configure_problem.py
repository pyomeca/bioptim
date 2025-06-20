from typing import Callable, Any

import numpy as np
from casadi import vertcat, Function, DM

from .configure_new_variable import NewVariableConfiguration
from .dynamics_functions import DynamicsFunctions
from .fatigue.fatigue_dynamics import FatigueList
from .ode_solvers import OdeSolver, OdeSolverBase
from ..gui.plot import CustomPlot
from ..misc.enums import (
    PlotType,
    Node,
    ConstraintType,
    PhaseDynamics,
    ContactType,
)
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping, Mapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..models.protocols.biomodel import BioModel
from ..models.protocols.stochastic_biomodel import StochasticBioModel
from ..optimization.problem_type import SocpType
from ..misc.parameters_types import (
    Bool,
    Int,
    FloatOptional,
    Str,
    StrOptional,
    StrList,
    NpArray,
    NpArrayDictOptional,
)
from ..optimization.non_linear_program import NonLinearProgram


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
    def _get_kinematics_based_names(nlp: NonLinearProgram, var_type: Str) -> StrList:
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
    def initialize(ocp, nlp: NonLinearProgram) -> None:
        """
        Call the dynamics a first time

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.type(
            ocp,
            nlp,
            numerical_data_timeseries=nlp.dynamics_type.numerical_data_timeseries,
            **nlp.dynamics_type.extra_parameters,
        )

    @staticmethod
    def custom(ocp, nlp: NonLinearProgram, **extra_params) -> None:
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
        nlp: NonLinearProgram,
        fatigue: FatigueList = None,
        numerical_data_timeseries: NpArrayDictOptional = None,
    ) -> None:
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        fatigue: FatigueList
            A list of fatigue elements
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """

        # Declared rigidbody states and controls
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True, fatigue=fatigue)
        ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False)

        ConfigureProblem.configure_contacts(
            ocp, nlp, nlp.model.contact_types, DynamicsFunctions.forces_from_torque_driven
        )

        # Configure the actual ODE of the dynamics
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_driven,
                fatigue=fatigue,
            )

    @staticmethod
    def torque_driven_free_floating_base(
        ocp,
        nlp: NonLinearProgram,
        numerical_data_timeseries: NpArrayDictOptional = None,
    ) -> None:
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
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node.
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
            )

    @staticmethod
    def stochastic_torque_driven(
        ocp,
        nlp: NonLinearProgram,
        problem_type,
        with_cholesky: Bool = False,
        initial_matrix: DM = None,
        numerical_data_timeseries: NpArrayDictOptional = None,
    ) -> None:
        """
        Configure the dynamics for a torque driven stochastic program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
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
        )

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            DynamicsFunctions.stochastic_torque_driven,
        )

    @staticmethod
    def stochastic_torque_driven_free_floating_base(
        ocp,
        nlp: NonLinearProgram,
        problem_type,
        with_cholesky: Bool = False,
        initial_matrix: DM = None,
        numerical_data_timeseries: NpArrayDictOptional = None,
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
        with_cholesky: bool
            If the Cholesky decomposition should be used for the covariance matrix.
        initial_matrix: DM
            The initial value for the covariance matrix
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node.
        """
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
        )

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            DynamicsFunctions.stochastic_torque_driven_free_floating_base,
        )

    @staticmethod
    def torque_derivative_driven(
        ocp,
        nlp: NonLinearProgram,
        numerical_data_timeseries: NpArrayDictOptional = None,
    ) -> None:
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.

        """
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_taudot(ocp, nlp, as_states=False, as_controls=True)

        ConfigureProblem.configure_contacts(
            ocp, nlp, nlp.model.contact_types, DynamicsFunctions.forces_from_torque_driven
        )

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_derivative_driven,
            )

    @staticmethod
    def torque_activations_driven(
        ocp,
        nlp: NonLinearProgram,
        with_residual_torque: Bool = False,
        numerical_data_timeseries: NpArrayDictOptional = None,
    ) -> None:
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
        with_residual_torque: bool
            If the dynamic with a residual torque should be used
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

        if with_residual_torque:
            ConfigureProblem.configure_residual_tau(ocp, nlp, as_states=False, as_controls=True)

        ConfigureProblem.configure_contacts(
            ocp, nlp, nlp.model.contact_types, DynamicsFunctions.forces_from_torque_activation_driven
        )
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_activations_driven,
                with_residual_torque=with_residual_torque,
            )

    @staticmethod
    def joints_acceleration_driven(
        ocp,
        nlp: NonLinearProgram,
        numerical_data_timeseries: NpArrayDictOptional = None,
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
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        # Configure qddot joints
        nb_root = nlp.model.nb_root
        if not nb_root > 0:
            raise RuntimeError("BioModel must have at least one DoF on root.")

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
        nlp: NonLinearProgram,
        with_excitations: Bool = False,
        fatigue: FatigueList = None,
        with_residual_torque: Bool = False,
        numerical_data_timeseries: NpArrayDictOptional = None,
    ) -> None:
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
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """
        if fatigue is not None and "tau" in fatigue and not with_residual_torque:
            raise RuntimeError("Residual torques need to be used to apply fatigue on torques")

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False)

        if with_residual_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True, fatigue=fatigue)
        ConfigureProblem.configure_muscles(ocp, nlp, with_excitations, as_controls=True, fatigue=fatigue)

        ConfigureProblem.configure_contacts(
            ocp, nlp, nlp.model.contact_types, DynamicsFunctions.forces_from_muscle_driven
        )

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.muscles_driven,
                fatigue=fatigue,
                with_residual_torque=with_residual_torque,
            )

    @staticmethod
    def holonomic_torque_driven(
        ocp,
        nlp: NonLinearProgram,
        numerical_data_timeseries: NpArrayDictOptional = None,
    ) -> None:
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
        """

        name = "q_u"
        names_u = [nlp.model.name_dof[i] for i in nlp.model.independent_joint_index]
        ConfigureProblem.configure_new_variable(
            name,
            names_u,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_algebraic_states=False,
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
            as_states=True,
            as_controls=False,
            as_algebraic_states=False,
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
    def configure_lagrange_multipliers_function(
        ocp, nlp: NpArrayDictOptional, dyn_func: Callable, **extra_params
    ) -> None:
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
    def configure_contacts(ocp, nlp, contact_types, force_from_where):
        if ContactType.RIGID_IMPLICIT in contact_types:
            ConfigureProblem.configure_rigid_contact_forces(
                ocp,
                nlp,
                as_states=False,
                as_algebraic_states=True,
                as_controls=False,
            )
        if ContactType.RIGID_EXPLICIT in contact_types:
            ConfigureProblem.configure_rigid_contact_function(ocp, nlp, force_from_where)
        if ContactType.SOFT_IMPLICIT in contact_types:
            ConfigureProblem.configure_soft_contact_forces(
                ocp, nlp, as_states=False, as_algebraic_states=True, as_controls=False
            )
        if ContactType.SOFT_EXPLICIT in contact_types:
            ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def configure_qv(ocp, nlp: NpArrayDictOptional, dyn_func: Callable, **extra_params) -> None:
        """
        Configure the qv, i.e. the dependent joint coordinates, to be plotted

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
        nlp.q_v_function = Function(
            "qv_function",
            [
                time_span_sym,
                nlp.states.cx,
                nlp.controls.cx,
                nlp.parameters.cx,
                nlp.algebraic_states.cx,
                nlp.numerical_timeseries.cx,
            ],
            [
                dyn_func()(
                    nlp.get_var_from_states_or_controls("q_u", nlp.states.cx, nlp.controls.cx),
                    DM.zeros(nlp.model.nb_dependent_joints, 1),
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["q_v"],
        )

        all_multipliers_names = []
        for nlp_i in ocp.nlp:
            if hasattr(nlp_i.model, "has_holonomic_constraints"):  # making sure we have a HolonomicBiorbdModel
                nlp_i_multipliers_names = [nlp_i.model.name_dof[i] for i in nlp_i.model.dependent_joint_index]
                all_multipliers_names.extend(
                    [name for name in nlp_i_multipliers_names if name not in all_multipliers_names]
                )

        all_multipliers_names_in_phase = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
        axes_idx = BiMapping(
            to_first=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
            to_second=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
        )

        nlp.plot["q_v"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.q_v_function(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_multipliers_names,
        )

    @staticmethod
    def configure_qdotv(ocp, nlp: NonLinearProgram, dyn_func: Callable, **extra_params) -> None:
        """
        Configure the qdot_v, i.e. the dependent joint velocities, to be plotted

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
        nlp.q_v_function = Function(
            "qdot_v_function",
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
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["qdot_v"],
        )

        all_multipliers_names = []
        for nlp_i in ocp.nlp:
            if hasattr(nlp_i.model, "has_holonomic_constraints"):  # making sure we have a HolonomicBiorbdModel
                nlp_i_multipliers_names = [nlp_i.model.name_dof[i] for i in nlp_i.model.dependent_joint_index]
                all_multipliers_names.extend(
                    [name for name in nlp_i_multipliers_names if name not in all_multipliers_names]
                )

        all_multipliers_names_in_phase = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
        axes_idx = BiMapping(
            to_first=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
            to_second=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
        )

        nlp.plot["qdot_v"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.q_v_function(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_multipliers_names,
        )

    @staticmethod
    def configure_dynamics_function(ocp, nlp: NonLinearProgram, dyn_func, **extra_params) -> None:
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

        # Check that the integrator matches the type of internal dynamics constraint
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            if dynamics_eval.defects is None:
                raise ValueError(
                    f"When using OdeSolver {nlp.dynamics_type.ode_solver} you must provide implicit defects (not dxdt)."
                )
        else:
            if dynamics_eval.dxdt is None:
                raise ValueError(
                    f"When using OdeSolver {nlp.dynamics_type.ode_solver} you must provide dxdt (not defects)."
                )

        dynamics_dxdt = dynamics_eval.dxdt
        if isinstance(dynamics_dxdt, (list, tuple)):
            dynamics_dxdt = vertcat(*dynamics_dxdt)

        dynamics_defects = dynamics_eval.defects
        if isinstance(dynamics_defects, (list, tuple)):
            dynamics_defects = vertcat(*dynamics_defects)

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        if dynamics_dxdt is not None:
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

        if dynamics_eval.defects is not None:
            if nlp.dynamics_defects_func is None:
                nlp.dynamics_defects_func = Function(
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
                    [dynamics_defects],
                    ["t_span", "x", "u", "p", "a", "d", "xdot"],
                    ["defects"],
                )
                if nlp.dynamics_type.expand_dynamics:
                    try:
                        nlp.dynamics_defects_func = nlp.dynamics_defects_func.expand()
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
                nlp.extra_dynamics_defects_func.append(
                    Function(
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
                    ),
                )

                if nlp.dynamics_type.expand_dynamics:
                    try:
                        nlp.extra_dynamics_defects_func[-1] = nlp.extra_dynamics_defects_func[-1].expand()
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
    def configure_rigid_contact_function(ocp, nlp: NonLinearProgram, contact_func: Callable, **extra_params) -> None:
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
            all_contact_names.extend([name for name in elt.model.rigid_contact_names if name not in all_contact_names])

        if "rigid_contact_forces" in nlp.plot_mapping:
            contact_names_in_phase = [name for name in nlp.model.rigid_contact_names]
            axes_idx = BiMapping(
                to_first=nlp.plot_mapping["rigid_contact_forces"].map_idx,
                to_second=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
            )
        else:
            contact_names_in_phase = [name for name in nlp.model.rigid_contact_names]
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
    def configure_soft_contact_function(ocp, nlp: NonLinearProgram) -> None:
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
            [nlp.model.soft_contact_forces().expand()(nlp.states["q"].cx, nlp.states["qdot"].cx, nlp.parameters.cx)],
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

    @staticmethod
    def configure_new_variable(
        name: Str,
        name_elements: StrList,
        ocp,
        nlp: NonLinearProgram,
        as_states: Bool,
        as_controls: Bool,
        as_algebraic_states: Bool = False,
        fatigue: FatigueList = None,
        combine_name: StrOptional = None,
        combine_state_control_plot: Bool = False,
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
        NewVariableConfiguration(
            name,
            name_elements,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_algebraic_states,
            fatigue,
            combine_name,
            combine_state_control_plot,
            skip_plot,
            axes_idx,
        )

    @staticmethod
    def configure_integrated_value(
        name: Str,
        name_elements: StrList,
        ocp,
        nlp: NonLinearProgram,
        initial_matrix: DM,
    ) -> None:
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
        n_cx = (
            nlp.dynamics_type.ode_solver.n_cx - 1
            if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 3
        )
        if n_cx < 3:
            n_cx = 3

        dummy_mapping = Mapping(list(range(len(name_elements))))
        initial_vector = StochasticBioModel.reshape_to_vector(initial_matrix)
        cx_scaled_next_formatted = [initial_vector for _ in range(n_cx)]
        nlp.integrated_values.append(
            name=name,
            cx=cx_scaled_next_formatted,
            cx_scaled=cx_scaled_next_formatted,  # Only the first value is used
            mapping=dummy_mapping,
            node_index=0,
        )
        for node_index in range(1, nlp.ns + 1):  # cannot use phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
            cx_scaled_next = [nlp.integrated_value_functions[name](nlp, node_index) for _ in range(n_cx)]
            nlp.integrated_values.append(
                name,
                cx_scaled_next_formatted,
                cx_scaled_next,
                dummy_mapping,
                node_index,
            )

    @staticmethod
    def configure_q(ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool) -> None:
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
        """
        name = "q"
        name_q = nlp.model.name_dof
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_q, ocp, nlp, as_states=as_states, as_controls=as_controls, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qdot(ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool) -> None:
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
        """

        name = "qdot"
        name_qdot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_qdot, ocp, nlp, as_states=as_states, as_controls=as_controls, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qddot(ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool) -> None:
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

        name = "qddot"
        name_qddot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_qddot, ocp, nlp, as_states=as_states, as_controls=as_controls, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qdddot(ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool) -> None:
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
        ConfigureProblem.configure_new_variable(
            name, name_qdddot, ocp, nlp, as_states=as_states, as_controls=as_controls, axes_idx=axes_idx
        )

    @staticmethod
    def configure_stochastic_k(ocp, nlp: NonLinearProgram, n_noised_controls: Int, n_references: Int) -> None:
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
            as_controls=True,
            as_algebraic_states=False,
        )

    @staticmethod
    def configure_stochastic_c(ocp, nlp: NonLinearProgram, n_noised_states: Int, n_noise: Int) -> None:
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
            as_controls=True,
            as_algebraic_states=False,
            skip_plot=True,
        )

    @staticmethod
    def configure_stochastic_a(ocp, nlp: NonLinearProgram, n_noised_states: Int) -> None:
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
            as_controls=True,
            as_algebraic_states=False,
            skip_plot=True,
        )

    @staticmethod
    def configure_stochastic_cov_explicit(ocp, nlp: NonLinearProgram, n_noised_states: Int, initial_matrix: DM) -> None:
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
    def configure_stochastic_cov_implicit(ocp, nlp: NonLinearProgram, n_noised_states: Int) -> None:
        """
        Configure the covariance matrix P representing the motor noise.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "cov"
        if "cov" in nlp.variable_mappings and nlp.variable_mappings["cov"].actually_does_a_mapping:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

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
            as_controls=True,
            as_algebraic_states=False,
        )

    @staticmethod
    def configure_stochastic_cholesky_cov(ocp, nlp: NonLinearProgram, n_noised_states: Int) -> None:
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
            as_controls=True,
            as_algebraic_states=False,
        )

    @staticmethod
    def configure_stochastic_ref(ocp, nlp: NonLinearProgram, n_references: Int) -> None:
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
            as_controls=True,
            as_algebraic_states=False,
        )

    @staticmethod
    def configure_stochastic_m(ocp, nlp: NonLinearProgram, n_noised_states: Int) -> None:
        """
        Configure the helper matrix M (from Gillis 2013 : https://doi.org/10.1109/CDC.2013.6761121).

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "m"

        if "m" in nlp.variable_mappings and nlp.variable_mappings["m"].actually_does_a_mapping:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_m = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_m += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(
            list(range(n_noised_states * n_noised_states)),
            list(range(n_noised_states * n_noised_states)),
        )
        ConfigureProblem.configure_new_variable(
            name,
            name_m,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_algebraic_states=True,
        )

    @staticmethod
    def configure_tau(
        ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool, fatigue: FatigueList = None
    ) -> None:
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
            name, name_tau, ocp, nlp, as_states=as_states, as_controls=as_controls, fatigue=fatigue, axes_idx=axes_idx
        )

    @staticmethod
    def configure_residual_tau(ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool) -> None:
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
            name, name_residual_tau, ocp, nlp, as_states=as_states, as_controls=as_controls, axes_idx=axes_idx
        )

    @staticmethod
    def configure_taudot(ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool) -> None:
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
        ConfigureProblem.configure_new_variable(
            name, name_taudot, ocp, nlp, as_states=as_states, as_controls=as_controls, axes_idx=axes_idx
        )

    @staticmethod
    def configure_translational_forces(
        ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool, n_contacts: Int = 1
    ) -> None:
        """
        Configure contact forces as optimization variables (for now only in global reference frame with an unknown point of application))
        # TODO: Match this with ExternalForceSetTimeSeries (options: 'in_global', 'torque', ...)

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the contact force should be a state
        as_controls: bool
            If the contact force should be a control
        n_contacts: int
            The number of contacts to consider (There will be 3 components for each contact)
        """

        name_contact_forces = [f"Force{i}_{axis}" for i in range(n_contacts) for axis in ("X", "Y", "Z")]
        ConfigureProblem.configure_new_variable(
            "contact_forces",
            name_contact_forces,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
        )
        ConfigureProblem.configure_new_variable(
            "contact_positions",
            name_contact_forces,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
        )

    @staticmethod
    def configure_rigid_contact_forces(
        ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool
    ) -> None:
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force should be a state
        as_controls: bool
            If the generalized force should be a control
        as_algebraic_states: bool
            If the generalized force should be an algebraic state
        """

        name_contact_forces = [name for name in nlp.model.rigid_contact_names]
        ConfigureProblem.configure_new_variable(
            "rigid_contact_forces",
            name_contact_forces,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
        )

    @staticmethod
    def configure_soft_contact_forces(
        ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool
    ) -> None:
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force should be a state
        as_controls: bool
            If the generalized force should be a control
        as_algebraic_states: bool
            If the generalized force should be an algebraic state
        """
        name_soft_contact_forces = [
            f"{name}_{axis}" for name in nlp.model.soft_contact_names for axis in ("MX", "MY", "MZ", "FX", "FY", "FZ")
        ]
        ConfigureProblem.configure_new_variable(
            "soft_contact_forces",
            name_soft_contact_forces,
            ocp,
            nlp,
            as_states=as_states,
            as_algebraic_states=as_algebraic_states,
            as_controls=as_controls,
        )

    @staticmethod
    def configure_muscles(
        ocp, nlp: NonLinearProgram, as_states: Bool, as_controls: Bool, fatigue: FatigueList = None
    ) -> None:
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
            as_states=as_states,
            as_controls=as_controls,
            combine_state_control_plot=True,
            fatigue=fatigue,
        )

    @staticmethod
    def _apply_phase_mapping(ocp, nlp: NonLinearProgram, name: Str) -> BiMapping | None:
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
        expand_dynamics: Bool = True,
        expand_continuity: Bool = False,
        skip_continuity: Bool = False,
        state_continuity_weight: FloatOptional = None,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver: OdeSolver | OdeSolverBase = OdeSolver.RK4(),
        numerical_data_timeseries: NpArrayDictOptional = None,
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

        if not isinstance(ode_solver, OdeSolverBase):
            raise RuntimeError("ode_solver should be built an instance of OdeSolver")

        super(Dynamics, self).__init__(type=dynamics_type, **extra_parameters)
        self.dynamic_function = dynamic_function
        self.configure = configure
        self.expand_dynamics = expand_dynamics
        self.expand_continuity = expand_continuity
        self.skip_continuity = skip_continuity
        self.state_continuity_weight = state_continuity_weight
        self.phase_dynamics = phase_dynamics
        self.ode_solver = ode_solver
        self.numerical_data_timeseries = numerical_data_timeseries


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

    def print(self) -> None:
        """
        Print the DynamicsList to the console
        """
        raise NotImplementedError("Printing of DynamicsList is not ready yet")


def _check_numerical_timeseries_format(numerical_timeseries: NpArray, n_shooting: Int, phase_idx: Int) -> None:
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
