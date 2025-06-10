from typing import Callable, Any

import numpy as np
from casadi import vertcat, Function, DM, horzcat

from .configure_variables import AutoConfigure
from .dynamics_functions import DynamicsFunctions
from .fatigue.fatigue_dynamics import FatigueList
from .ode_solvers import OdeSolver, OdeSolverBase
from ..gui.plot import CustomPlot
from ..misc.enums import (
    PlotType,
    PhaseDynamics,
    ControlType,
)
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..optimization.problem_type import SocpType
from ..misc.parameters_types import (
    Bool,
    Int,
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
        if nlp.dynamics_type.configure_function is not None:
            nlp.dynamics_type.configure_function.initialize(
                ocp,
                nlp,
                **nlp.dynamics_type.extra_parameters,
            )
        else:
            AutoConfigure(
                states=nlp.model.state_type,
                controls=nlp.model.control_type,
                algebraic_states=nlp.model.algebraic_type,
                functions=nlp.model.functions,
            ).initialize(ocp, nlp)

        ConfigureProblem.configure_dynamics_function(ocp, nlp, nlp.model.dynamics, **nlp.dynamics_type.extra_parameters)

        if nlp.model.extra_dynamics is not None:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, nlp.model.extra_dynamics, **nlp.dynamics_type.extra_parameters
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


class DynamicsFcn(FcnEnum):
    """
    Selection of valid dynamics functions
    """

    TORQUE_DERIVATIVE_DRIVEN = (ConfigureProblem.torque_derivative_driven,)
    JOINTS_ACCELERATION_DRIVEN = (ConfigureProblem.joints_acceleration_driven,)
    MUSCLE_DRIVEN = (ConfigureProblem.muscle_driven,)
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
        configure_function: Callable = None,
        dynamic_function: Callable = None,
        expand_dynamics: bool = True,
        expand_continuity: bool = False,
        skip_continuity: bool = False,
        state_continuity_weight: float | None = None,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver: OdeSolver | OdeSolverBase = OdeSolver.RK4(),
        numerical_data_timeseries: dict[str, np.ndarray] = None,
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
        if (configure_function is None and dynamic_function is not None) or (
            configure_function is not None and dynamic_function is None
        ):
            raise RuntimeError(
                "Either both configure_function and dynamics_function should be provided, or none of them."
            )

        super(Dynamics, self).__init__(**extra_parameters)

        self.dynamic_function = dynamic_function
        self.configure_function = configure_function
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

    def add(self, dynamics_type, **extra_parameters: Any):
        """
        Add a new Dynamics to the list

        Parameters
        ----------
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
