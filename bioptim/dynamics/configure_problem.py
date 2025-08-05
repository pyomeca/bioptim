from typing import Any
import numpy as np
from casadi import vertcat, Function

from .configure_variables import AutoConfigure
from .ode_solvers import OdeSolver, OdeSolverBase
from ..misc.enums import (
    PhaseDynamics,
)
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..misc.parameters_types import (
    Int,
    NpArray,
)
from ..optimization.non_linear_program import NonLinearProgram
from ..limits.weight import ObjectiveWeight, ConstraintWeight


class ConfigureProblem:
    """
    Dynamics configuration for the most common ocp
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
        AutoConfigure(
            states=nlp.model.state_configuration,
            controls=nlp.model.control_configuration,
            algebraic_states=nlp.model.algebraic_configuration,
            functions=nlp.model.functions,
        ).initialize(ocp, nlp)

        ConfigureProblem.configure_dynamics_function(ocp, nlp, nlp.model.dynamics, **nlp.dynamics_type.extra_parameters)

        if nlp.model.extra_dynamics is not None:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, nlp.model.extra_dynamics, **nlp.dynamics_type.extra_parameters
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


class DynamicsOptions(OptionGeneric):
    """
    A placeholder for the chosen dynamics by the user

    Attributes
    ----------
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
        expand_dynamics: bool = True,
        expand_continuity: bool = False,
        skip_continuity: bool = False,
        state_continuity_weight: (
            float | int | ConstraintWeight | ObjectiveWeight
        ) = ConstraintWeight(),  # Default is a constraint
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver: OdeSolver | OdeSolverBase = OdeSolver.RK4(),
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        **extra_parameters: Any,
    ):
        """
        Parameters
        ----------
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
            # TODO: numerical_data_timeseries should be moved in the model instead of the dynamics options.
        """
        if "fatigue" in extra_parameters:
            raise ValueError(
                "Fatigue is not an argument of the dynamics anymore, it should be sent to the model instead."
            )

        super().__init__(**extra_parameters)

        if not isinstance(expand_dynamics, bool):
            raise RuntimeError("expand_dynamics must be a boolean.")
        if not isinstance(expand_continuity, bool):
            raise RuntimeError("expand_continuity must be a boolean.")
        if not isinstance(skip_continuity, bool):
            raise RuntimeError("skip_continuity must be a boolean.")
        if not isinstance(state_continuity_weight, (float, int, ConstraintWeight, ObjectiveWeight)):
            raise RuntimeError(
                "state_continuity_weight must be an int, a float, an ObjectiveWeight, or a ConstraintWeight."
            )
        if not isinstance(phase_dynamics, PhaseDynamics):
            raise RuntimeError("phase_dynamics must be of type PhaseDynamics.")
        if not isinstance(ode_solver, (OdeSolver, OdeSolverBase)):
            raise RuntimeError("ode_solver should be built an instance of OdeSolver")
        if numerical_data_timeseries is not None:
            if not isinstance(numerical_data_timeseries, dict):
                raise RuntimeError("numerical_data_timeseries must be a dictionary.")
            for key, value in numerical_data_timeseries.items():
                if not isinstance(value, np.ndarray):
                    raise RuntimeError(
                        f"numerical_data_timeseries[{key}] must be a numpy array, but got {type(value)}."
                    )

        self.expand_dynamics = expand_dynamics
        self.expand_continuity = expand_continuity
        self.skip_continuity = skip_continuity
        self.state_continuity_weight = state_continuity_weight
        self.phase_dynamics = phase_dynamics
        self.ode_solver = ode_solver
        self.numerical_data_timeseries = numerical_data_timeseries


class DynamicsOptionsList(UniquePerPhaseOptionList):
    """
    A list of DynamicsOptions if more than one is required, typically when more than one phases are declared

    Methods
    -------
    add(dynamics: DynamicsOptions, **extra_parameters)
        Add a new DynamicsOptions to the list
    print(self)
        Print the DynamicsOptionsList to the console
    """

    def add(self, dynamics=None, **extra_parameters: Any):
        """
        Add a new DynamicsOptions to the list

        Parameters
        ----------
        dynamics: DynamicsOptions | None
            The dynamics to add to the list. If None, a DynamicsOptions will be created using the extra_parameters.
        extra_parameters: dict
            Any parameters to pass to DynamicsOptions
        """
        if dynamics is None:
            self.add(DynamicsOptions(**extra_parameters))
        elif isinstance(dynamics, DynamicsOptions):
            self.copy(dynamics)
        else:
            raise ValueError("The dynamics must be of type DynamicsOptions.")

    def print(self) -> None:
        """
        Print the DynamicsOptionsList to the console
        """
        raise NotImplementedError("Printing of DynamicsOptionsList is not ready yet")


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
