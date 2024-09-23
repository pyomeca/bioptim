from math import inf
from typing import Callable, Any

import biorbd_casadi as biorbd
import casadi
import numpy as np
from casadi import MX, SX, sum1, horzcat
from matplotlib import pyplot as plt

from .non_linear_program import NonLinearProgram as NLP
from .optimization_vector import OptimizationVectorHelper
from ..dynamics.configure_problem import DynamicsList, Dynamics, ConfigureProblem
from ..dynamics.ode_solver import OdeSolver, OdeSolverBase
from ..gui.check_conditioning import check_conditioning
from ..gui.graph import OcpToConsole, OcpToGraph
from ..gui.ipopt_output_plot import SaveIterationsInfo
from ..gui.plot import CustomPlot, PlotOcp
from ..interfaces import Solver
from ..interfaces.abstract_options import GenericSolver
from ..limits.constraints import (
    ConstraintFunction,
    ConstraintFcn,
    ConstraintList,
    Constraint,
    ParameterConstraintList,
    ParameterConstraint,
)
from ..limits.multinode_constraint import MultinodeConstraintList
from ..limits.multinode_objective import MultinodeObjectiveList
from ..limits.objective_functions import (
    ObjectiveFcn,
    ObjectiveList,
    Objective,
    ParameterObjectiveList,
    ParameterObjective,
)
from ..limits.objective_functions import ObjectiveFunction
from ..limits.path_conditions import BoundsList, Bounds
from ..limits.path_conditions import InitialGuess, InitialGuessList
from ..limits.penalty import PenaltyOption
from ..limits.penalty_helpers import PenaltyHelpers
from ..limits.phase_transition import PhaseTransitionList, PhaseTransitionFcn
from ..limits.phase_transtion_factory import PhaseTransitionFactory
from ..misc.__version__ import __version__
from ..misc.enums import (
    ControlType,
    SolverType,
    Shooting,
    PlotType,
    CostType,
    SolutionIntegrator,
    InterpolationType,
    PenaltyType,
    Node,
)
from ..misc.mapping import BiMappingList, Mapping, BiMapping, NodeMappingList
from ..misc.options import OptionDict
from ..models.biorbd.variational_biorbd_model import VariationalBiorbdModel
from ..models.protocols.biomodel import BioModel
from ..optimization.optimization_variable import OptimizationVariableList
from ..optimization.parameters import ParameterList, Parameter, ParameterContainer
from ..optimization.solution.solution import Solution
from ..optimization.solution.solution_data import SolutionMerge
from ..optimization.variable_scaling import VariableScalingList, VariableScaling


class OptimalControlProgram:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed interface to modify and solve the program

    Attributes
    ----------
    cx: [MX, SX]
        The base type for the symbolic casadi variables
    g: list
        Constraints that are not phase dependent (mostly parameters and continuity constraints)
    g_internal: list[list[Constraint]]
        All the constraints internally defined by the OCP at each of the node of the phase
    g_implicit: list[list[Constraint]]
        All the implicit constraints defined by the OCP at each of the node of the phase
    J: list
        Objective values that are not phase dependent (mostly parameters)
    nlp: NLP
        All the phases of the ocp
    n_phases: int | list | tuple
        The number of phases of the ocp
    n_threads: int
        The number of thread to use if using multithreading
    phase_time: list[float]
        The time vector as sent by the user
    phase_transitions: list[PhaseTransition]
        The list of transition constraint between phases
    ocp_solver: SolverInterface
        A reference to the ocp solver
    version: dict
        The version of all the underlying software. This is important when loading a previous ocp

    Methods
    -------
    update_objectives(self, new_objective_function: Objective | ObjectiveList)
        The main user interface to add or modify objective functions in the ocp
    update_objectives_target(self, target, phase=None, list_index=None)
        Fast accessor to update the target of a specific objective function. To update target of global objective
        (usually defined by parameters), one can pass 'phase=-1
    update_constraints(self, new_constraint: Constraint | ConstraintList)
        The main user interface to add or modify constraint in the ocp
    update_parameters(self, new_parameters: Parameter | ParameterList)
        The main user interface to add or modify parameters in the ocp
    update_bounds(self, x_bounds: Bounds | BoundsList, u_bounds: Bounds | BoundsList)
        The main user interface to add bounds in the ocp
    update_initial_guess(
        self,
        x_init: InitialGuess | InitialGuessList,
        u_init: InitialGuess | InitialGuessList,
        param_init: InitialGuess | InitialGuessList,
    )
        The main user interface to add initial guesses in the ocp
    add_plot(self, fig_name: str, update_function: Callable, phase: int = -1, **parameters: Any)
        The main user interface to add a new plot to the ocp
    prepare_plots(self, automatically_organize: bool, show_bounds: bool,
            shooting_type: Shooting) -> PlotOCP
        Create all the plots associated with the OCP
    solve(self, solver: Solver) -> Solution
        Call the solver to actually solve the ocp
    _define_time(self, phase_time: float | tuple, objective_functions: ObjectiveList, constraints: ConstraintList)
        Declare the phase_time vector in v. If objective_functions or constraints defined a time optimization,
        a sanity check is perform and the values of initial guess and bounds for these particular phases
    _modify_penalty(self, new_penalty: PenaltyOption | Parameter)
        The internal function to modify a penalty.
    _set_nlp_is_stochastic(self)
        Set the nlp as stochastic if any of the phases is stochastic
    _set_internal_algebraic_states(self)
        Set the internal algebraic_states variables (a_init, a_bounds, a_scaling) if any of the phases
    _check_quaternions_hasattr(self, bio_model)
        Check if the bio_model has quaternions and set the flag accordingly
    """

    def __init__(
        self,
        bio_model: list | tuple | BioModel,
        dynamics: Dynamics | DynamicsList,
        n_shooting: int | list | tuple,
        phase_time: int | float | list | tuple,
        x_bounds: BoundsList = None,
        u_bounds: BoundsList = None,
        x_init: InitialGuessList | None = None,
        u_init: InitialGuessList | None = None,
        objective_functions: Objective | ObjectiveList = None,
        constraints: Constraint | ConstraintList = None,
        parameters: ParameterList = None,
        parameter_bounds: BoundsList = None,
        parameter_init: InitialGuessList = None,
        parameter_objectives: ParameterObjectiveList = None,
        parameter_constraints: ParameterConstraintList = None,
        ode_solver: list | OdeSolverBase | OdeSolver = None,
        control_type: ControlType | list = ControlType.CONSTANT,
        variable_mappings: BiMappingList = None,
        time_phase_mapping: BiMapping = None,
        node_mappings: NodeMappingList = None,
        plot_mappings: Mapping = None,
        phase_transitions: PhaseTransitionList = None,
        multinode_constraints: MultinodeConstraintList = None,
        multinode_objectives: MultinodeObjectiveList = None,
        x_scaling: VariableScalingList = None,
        xdot_scaling: VariableScalingList = None,
        u_scaling: VariableScalingList = None,
        n_threads: int = 1,
        use_sx: bool = False,
        integrated_value_functions: dict[str, Callable] = None,
    ):
        """
        Parameters
        ----------
        bio_model: list | tuple | BioModel
            The bio_model to use for the optimization
        dynamics: Dynamics | DynamicsList
            The dynamics of the phases
        n_shooting: int | list[int]
            The number of shooting point of the phases
        phase_time: int | float | list | tuple
            The phase time of the phases
        x_init: InitialGuess | InitialGuessList
            The initial guesses for the states
        u_init: InitialGuess | InitialGuessList
            The initial guesses for the controls
        x_bounds: Bounds | BoundsList
            The bounds for the states
        u_bounds: Bounds | BoundsList
            The bounds for the controls
        x_scaling: VariableScalingList
            The scaling for the states at each phase, if only one is sent, then the scaling is copied over the phases
        xdot_scaling: VariableScalingList
            The scaling for the states derivative, if only one is sent, then the scaling is copied over the phases
        u_scaling: VariableScalingList
            The scaling for the controls, if only one is sent, then the scaling is copied over the phases
        objective_functions: Objective | ObjectiveList
            All the objective function of the program
        constraints: Constraint | ConstraintList
            All the constraints of the program
        parameters: ParameterList
            All the parameters to optimize of the program
        parameter_bounds: BoundsList
            The bounds for the parameters, default values are -inf to inf
        parameter_init: InitialGuessList
            The initial guess for the parameters, default value is 0
        parameter_objectives: ParameterObjectiveList
            All the parameter objectives to optimize of the program
        parameter_constraints: ParameterConstraintList
            All the parameter constraints of the program
        ode_solver: OdeSolverBase
            The solver for the ordinary differential equations
        control_type: ControlType
            The type of controls for each phase
        variable_mappings: BiMappingList
            The mapping to apply on variables
        time_phase_mapping: BiMapping
            The mapping of the time of the phases, so some phase share the same time variable (when optimized, that is
            a constraint or an objective on the time is declared)
        node_mappings: NodeMappingList
            The mapping to apply between the variables associated with the nodes
        plot_mappings: Mapping
            The mapping to apply on the plots
        phase_transitions: PhaseTransitionList
            The transition types between the phases
        n_threads: int
            The number of thread to use while solving (multi-threading if > 1)
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        """

        self._check_bioptim_version()

        bio_model = self._initialize_model(bio_model)

        # Placed here because of MHE
        a_init, a_bounds, a_scaling = self._set_internal_algebraic_states()

        self._check_and_set_threads(n_threads)
        self._check_and_set_shooting_points(n_shooting)
        self._check_and_set_phase_time(phase_time)

        (
            x_bounds,
            x_init,
            x_scaling,
            u_bounds,
            u_init,
            u_scaling,
            a_bounds,
            a_init,
            a_scaling,
            xdot_scaling,
        ) = self._prepare_all_decision_variables(
            x_bounds,
            x_init,
            x_scaling,
            u_bounds,
            u_init,
            u_scaling,
            xdot_scaling,
            a_bounds,
            a_init,
            a_scaling,
        )

        (
            constraints,
            objective_functions,
            parameter_constraints,
            parameter_objectives,
            multinode_constraints,
            multinode_objectives,
            phase_transitions,
            parameter_bounds,
            parameter_init,
        ) = self._check_arguments_and_build_nlp(
            dynamics,
            objective_functions,
            constraints,
            parameters,
            phase_transitions,
            multinode_constraints,
            multinode_objectives,
            parameter_bounds,
            parameter_init,
            parameter_constraints,
            parameter_objectives,
            ode_solver,
            use_sx,
            bio_model,
            plot_mappings,
            time_phase_mapping,
            control_type,
            variable_mappings,
            integrated_value_functions,
        )
        self._is_warm_starting = False

        # Do not copy singleton since x_scaling was already dealt with before
        NLP.add(self, "x_scaling", x_scaling, True)
        NLP.add(self, "xdot_scaling", xdot_scaling, True)
        NLP.add(self, "u_scaling", u_scaling, True)
        NLP.add(self, "a_scaling", a_scaling, True)

        self._set_nlp_is_stochastic()

        self._prepare_node_mapping(node_mappings)
        self._prepare_dynamics()
        self._prepare_bounds_and_init(
            x_bounds, u_bounds, parameter_bounds, a_bounds, x_init, u_init, parameter_init, a_init
        )

        self._declare_multi_node_penalties(multinode_constraints, multinode_objectives, constraints, phase_transitions)

        self._finalize_penalties(
            constraints,
            parameter_constraints,
            objective_functions,
            parameter_objectives,
            phase_transitions,
        )

    def _check_bioptim_version(self):
        self.version = {"casadi": casadi.__version__, "biorbd": biorbd.__version__, "bioptim": __version__}
        return

    def _initialize_model(self, bio_model):
        """
        Initialize the bioptim model and check if the quaternions are used, if yes then setting them.
        Setting the number of phases.
        """
        if not isinstance(bio_model, (list, tuple)):
            bio_model = [bio_model]
        bio_model = self._check_quaternions_hasattr(bio_model)
        self.n_phases = len(bio_model)
        return bio_model

    def _check_and_set_threads(self, n_threads):
        if not isinstance(n_threads, int) or isinstance(n_threads, bool) or n_threads < 1:
            raise RuntimeError("n_threads should be a positive integer greater or equal than 1")
        self.n_threads = n_threads

    def _check_and_set_shooting_points(self, n_shooting):
        if not isinstance(n_shooting, int) or n_shooting < 2:
            if isinstance(n_shooting, (tuple, list)):
                if sum([True for i in n_shooting if not isinstance(i, int) and not isinstance(i, bool)]) != 0:
                    raise RuntimeError("n_shooting should be a positive integer (or a list of) greater or equal than 2")
            else:
                raise RuntimeError("n_shooting should be a positive integer (or a list of) greater or equal than 2")
        self.n_shooting = n_shooting

    def _check_and_set_phase_time(self, phase_time):
        if not isinstance(phase_time, (int, float)):
            if isinstance(phase_time, (tuple, list)):
                if sum([True for i in phase_time if not isinstance(i, (int, float))]) != 0:
                    raise RuntimeError("phase_time should be a number or a list of number")
            else:
                raise RuntimeError("phase_time should be a number or a list of number")
        self.phase_time = phase_time

    def _check_and_prepare_decision_variables(
        self,
        var_name: str,
        bounds: BoundsList,
        init: InitialGuessList,
        scaling: VariableScalingList,
    ):
        """
        This function checks if the decision variables are of the right type for initial guess and bounds.
        It also prepares the scaling for the decision variables.
        And set them in a dictionary for the phase.
        """

        if bounds is None:
            bounds = BoundsList()
        elif not isinstance(bounds, BoundsList):
            raise RuntimeError(f"{var_name}_bounds should be built from a BoundsList")

        if init is None:
            init = InitialGuessList()
        elif not isinstance(init, InitialGuessList):
            raise RuntimeError(f"{var_name}_init should be built from a InitialGuessList")

        bounds = self._prepare_option_dict_for_phase(f"{var_name}_bounds", bounds, BoundsList)
        init = self._prepare_option_dict_for_phase(f"{var_name}_init", init, InitialGuessList)
        scaling = self._prepare_option_dict_for_phase(f"{var_name}_scaling", scaling, VariableScalingList)

        return bounds, init, scaling

    def _prepare_all_decision_variables(
        self,
        x_bounds,
        x_init,
        x_scaling,
        u_bounds,
        u_init,
        u_scaling,
        xdot_scaling,
        a_bounds,
        a_init,
        a_scaling,
    ):
        """
        This function checks if the decision variables are of the right type for initial guess and bounds.
        It also prepares the scaling for the decision variables.
        """

        # states
        x_bounds, x_init, x_scaling = self._check_and_prepare_decision_variables("x", x_bounds, x_init, x_scaling)
        # controls
        u_bounds, u_init, u_scaling = self._check_and_prepare_decision_variables("u", u_bounds, u_init, u_scaling)
        # algebraic states
        a_bounds, a_init, a_scaling = self._check_and_prepare_decision_variables("a", a_bounds, a_init, a_scaling)

        xdot_scaling = self._prepare_option_dict_for_phase("xdot_scaling", xdot_scaling, VariableScalingList)

        return x_bounds, x_init, x_scaling, u_bounds, u_init, u_scaling, a_bounds, a_init, a_scaling, xdot_scaling

    def _check_arguments_and_build_nlp(
        self,
        dynamics,
        objective_functions,
        constraints,
        parameters,
        phase_transitions,
        multinode_constraints,
        multinode_objectives,
        parameter_bounds,
        parameter_init,
        parameter_constraints,
        parameter_objectives,
        ode_solver,
        use_sx,
        bio_model,
        plot_mappings,
        time_phase_mapping,
        control_type,
        variable_mappings,
        integrated_value_functions,
    ):
        if objective_functions is None:
            objective_functions = ObjectiveList()
        elif isinstance(objective_functions, Objective):
            objective_functions_tp = ObjectiveList()
            objective_functions_tp.add(objective_functions)
            objective_functions = objective_functions_tp
        elif not isinstance(objective_functions, ObjectiveList):
            raise RuntimeError("objective_functions should be built from an Objective or ObjectiveList")

        self.implicit_constraints = ConstraintList()

        if constraints is None:
            constraints = ConstraintList()
        elif isinstance(constraints, Constraint):
            constraints_tp = ConstraintList()
            constraints_tp.add(constraints)
            constraints = constraints_tp
        elif not isinstance(constraints, ConstraintList):
            raise RuntimeError("constraints should be built from an Constraint or ConstraintList")

        if parameters is None:
            parameters = ParameterList(use_sx=use_sx)
        elif not isinstance(parameters, ParameterList):
            raise RuntimeError("parameters should be built from an ParameterList")

        if phase_transitions is None:
            phase_transitions = PhaseTransitionList()
        elif not isinstance(phase_transitions, PhaseTransitionList):
            raise RuntimeError("phase_transitions should be built from an PhaseTransitionList")

        if multinode_constraints is None:
            multinode_constraints = MultinodeConstraintList()
        elif not isinstance(multinode_constraints, MultinodeConstraintList):
            raise RuntimeError("multinode_constraints should be built from an MultinodeConstraintList")

        if multinode_objectives is None:
            multinode_objectives = MultinodeObjectiveList()
        elif not isinstance(multinode_objectives, MultinodeObjectiveList):
            raise RuntimeError("multinode_objectives should be built from an MultinodeObjectiveList")

        if parameter_bounds is None:
            parameter_bounds = BoundsList()
        elif not isinstance(parameter_bounds, BoundsList):
            raise ValueError("parameter_bounds must be of type BoundsList")

        if parameter_init is None:
            parameter_init = InitialGuessList()
        elif not isinstance(parameter_init, InitialGuessList):
            raise ValueError("parameter_init must be of type InitialGuessList")

        if parameter_objectives is None:
            parameter_objectives = ParameterObjectiveList()
        elif isinstance(parameter_objectives, ParameterObjective):
            parameter_objectives_tp = ParameterObjectiveList()
            parameter_objectives_tp.add(parameter_objectives)
            parameter_objectives = parameter_objectives_tp
        elif not isinstance(parameter_objectives, ParameterObjectiveList):
            raise RuntimeError("objective_functions should be built from an Objective or ObjectiveList")

        if parameter_constraints is None:
            parameter_constraints = ParameterConstraintList()
        elif isinstance(constraints, ParameterConstraint):
            parameter_constraints_tp = ParameterConstraintList()
            parameter_constraints_tp.add(parameter_constraints)
            parameter_constraints = parameter_constraints_tp
        elif not isinstance(parameter_constraints, ParameterConstraintList):
            raise RuntimeError("constraints should be built from an Constraint or ConstraintList")

        if ode_solver is None:
            ode_solver = self._set_default_ode_solver()
        elif not isinstance(ode_solver, OdeSolverBase):
            raise RuntimeError("ode_solver should be built an instance of OdeSolver")

        if not isinstance(use_sx, bool):
            raise RuntimeError("use_sx should be a bool")

        if isinstance(dynamics, Dynamics):
            tp = dynamics
            dynamics = DynamicsList()
            dynamics.add(tp)
        if not isinstance(dynamics, DynamicsList):
            raise ValueError("dynamics must be of type DynamicsList or Dynamics")

        # Type of CasADi graph
        self.cx = SX if use_sx else MX

        # Declare optimization variables
        self.program_changed = True
        self.J = []
        self.J_internal = []
        self.g = []
        self.g_internal = []
        self.g_implicit = []

        # nlp is the core of a phase
        self.nlp = [NLP(dynamics[i].phase_dynamics, use_sx) for i in range(self.n_phases)]
        NLP.add(self, "model", bio_model, False)
        NLP.add(self, "phase_idx", [i for i in range(self.n_phases)], False)

        # Define some aliases
        NLP.add(self, "ns", self.n_shooting, False)
        for nlp in self.nlp:
            if nlp.ns < 1:
                raise RuntimeError("Number of shooting points must be at least 1")

        NLP.add(self, "n_threads", self.n_threads, True)
        self.ocp_solver = None

        plot_mappings = plot_mappings if plot_mappings is not None else {}
        reshaped_plot_mappings = []
        for i in range(self.n_phases):
            reshaped_plot_mappings.append({})
            for key in plot_mappings:
                reshaped_plot_mappings[i][key] = plot_mappings[key][i]
        NLP.add(self, "plot_mapping", reshaped_plot_mappings, False, name="plot_mapping")

        phase_mapping, dof_names = self._set_kinematic_phase_mapping()
        NLP.add(self, "phase_mapping", phase_mapping, True)
        NLP.add(self, "dof_names", dof_names, True)

        # Prepare the parameter mappings
        if time_phase_mapping is None:
            time_phase_mapping = BiMapping(
                to_second=[i for i in range(self.n_phases)], to_first=[i for i in range(self.n_phases)]
            )
        self.time_phase_mapping = time_phase_mapping

        # Add any time related parameters to the parameters list before declaring it
        self._define_time(self.phase_time, objective_functions, constraints)

        # Declare and fill the parameters
        self._declare_parameters(parameters)

        # Declare the numerical timeseries used at each nodes as symbolic variables
        self._define_numerical_timeseries(dynamics)

        # Prepare path constraints and dynamics of the program
        NLP.add(self, "dynamics_type", dynamics, False)
        NLP.add(self, "ode_solver", ode_solver, True)
        NLP.add(self, "control_type", control_type, True)

        # Prepare the variable mappings
        if variable_mappings is None:
            variable_mappings = BiMappingList()

        variable_mappings = variable_mappings.variable_mapping_fill_phases(self.n_phases)
        NLP.add(self, "variable_mappings", variable_mappings, True)

        NLP.add(self, "integrated_value_functions", integrated_value_functions, True)

        # If we want to plot what is printed by IPOPT in the console
        self.plot_ipopt_outputs = False
        # If we want the conditioning of the problem to be plotted live
        self.plot_check_conditioning = False
        self.save_ipopt_iterations_info = None

        return (
            constraints,
            objective_functions,
            parameter_constraints,
            parameter_objectives,
            multinode_constraints,
            multinode_objectives,
            phase_transitions,
            parameter_bounds,
            parameter_init,
        )

    def _prepare_node_mapping(self, node_mappings):
        # Prepare the node mappings
        if node_mappings is None:
            node_mappings = NodeMappingList()
        (
            use_states_from_phase_idx,
            use_states_dot_from_phase_idx,
            use_controls_from_phase_idx,
        ) = node_mappings.get_variable_from_phase_idx(self)

        self._check_variable_mapping_consistency_with_node_mapping(
            use_states_from_phase_idx, use_controls_from_phase_idx
        )

    def _prepare_dynamics(self):
        # Prepare the dynamics
        for i in range(self.n_phases):
            self.nlp[i].initialize(self.cx)
            self.nlp[i].parameters = self.parameters  # This should be remove when phase parameters will be implemented
            self.nlp[i].parameters_except_time = self.parameters_except_time
            self.nlp[i].numerical_data_timeseries = self.nlp[i].dynamics_type.numerical_data_timeseries
            ConfigureProblem.initialize(self, self.nlp[i])
            self.nlp[i].ode_solver.prepare_dynamic_integrator(self, self.nlp[i])
            if (isinstance(self.nlp[i].model, VariationalBiorbdModel)) and self.nlp[i].algebraic_states.shape > 0:
                raise NotImplementedError(
                    "Algebraic states were not tested with variational integrators. If you come across this error, "
                    "please notify the developers by opening open an issue on GitHub pinging Ipuch and EveCharbie"
                )

    def _prepare_bounds_and_init(
        self, x_bounds, u_bounds, parameter_bounds, a_bounds, x_init, u_init, parameter_init, a_init
    ):
        self.parameter_bounds = BoundsList()
        self.parameter_init = InitialGuessList()

        self.update_bounds(x_bounds, u_bounds, parameter_bounds, a_bounds)
        self.update_initial_guess(x_init, u_init, parameter_init, a_init)
        # Define the actual NLP problem
        OptimizationVectorHelper.declare_ocp_shooting_points(self)

    def _declare_multi_node_penalties(
        self,
        multinode_constraints: ConstraintList,
        multinode_objectives: ObjectiveList,
        constraints: ConstraintList,
        phase_transition: PhaseTransitionList,
    ):
        """
        This function declares the multi node penalties (constraints and objectives) to the penalty pool.

        Note
        ----
        This function is overriden in StochasticOptimalControlProgram
        """
        multinode_constraints.add_or_replace_to_penalty_pool(self)
        multinode_objectives.add_or_replace_to_penalty_pool(self)

    def _finalize_penalties(
        self,
        constraints,
        parameter_constraints,
        objective_functions,
        parameter_objectives,
        phase_transitions,
    ):
        # Define continuity constraints
        # Prepare phase transitions (Reminder, it is important that parameters are declared before,
        # otherwise they will erase the phase_transitions)
        self.phase_transitions = PhaseTransitionFactory(ocp=self).prepare_phase_transitions(phase_transitions)

        # Skipping creates an OCP without built-in continuity constraints, make sure you declared constraints elsewhere
        self._declare_continuity()

        # Prepare constraints
        self.update_constraints(self.implicit_constraints)
        self.update_constraints(constraints)
        self.update_parameter_constraints(parameter_constraints)

        # Prepare objectives
        self.update_objectives(objective_functions)
        self.update_parameter_objectives(parameter_objectives)
        return

    def finalize_plot_phase_mappings(self):
        """
        Finalize the plot phase mappings (if not already done)

        Parameters
        ----------
        n_phases: int
            The number of phases
        """

        for nlp in self.nlp:
            if not nlp.plot:
                return

            for key in nlp.plot:
                if isinstance(nlp.plot[key], tuple):
                    nlp.plot[key] = nlp.plot[key][0]

                # This is the point where we can safely define node_idx of the plot
                if nlp.plot[key].node_idx is None:
                    nlp.plot[key].node_idx = range(nlp.n_states_nodes)

                # If the number of subplots is not known, compute it
                if nlp.plot[key].phase_mappings is None:
                    node_index = nlp.plot[key].node_idx[0]
                    nlp.states.node_index = node_index
                    nlp.states_dot.node_index = node_index
                    nlp.controls.node_index = node_index
                    nlp.algebraic_states.node_index = node_index

                    # If multi-node penalties = None, stays zero
                    size_x = nlp.states.shape
                    size_u = nlp.controls.shape
                    size_p = nlp.parameters.shape
                    size_a = nlp.algebraic_states.shape
                    size_d = nlp.numerical_timeseries.shape
                    if "penalty" in nlp.plot[key].parameters:
                        penalty = nlp.plot[key].parameters["penalty"]

                        # As stated in penalty_option, the last controller is always supposed to be the right one
                        casadi_function = (
                            penalty.function[0] if penalty.function[0] is not None else penalty.function[-1]
                        )
                        if casadi_function is not None:
                            size_x = casadi_function.size_in("x")[0]
                            size_u = casadi_function.size_in("u")[0]
                            size_p = casadi_function.size_in("p")[0]
                            size_a = casadi_function.size_in("a")[0]
                            size_d = casadi_function.size_in("d")[0]

                    size = (
                        nlp.plot[key]
                        .function(
                            0,  # t0
                            np.zeros(self.n_phases),  # phases_dt
                            node_index,  # node_idx
                            np.zeros((size_x, 1)),  # states
                            np.zeros((size_u, 1)),  # controls
                            np.zeros((size_p, 1)),  # parameters
                            np.zeros((size_a, 1)),  # algebraic_states
                            np.zeros((size_d, 1)),  # numerical_timeseries
                            **nlp.plot[key].parameters,  # parameters
                        )
                        .shape[0]
                    )
                    nlp.plot[key].phase_mappings = BiMapping(to_first=range(size), to_second=range(size))

    @property
    def variables_vector(self):
        return OptimizationVectorHelper.vector(self)

    @property
    def bounds_vectors(self):
        return OptimizationVectorHelper.bounds_vectors(self)

    @property
    def init_vector(self):
        return OptimizationVectorHelper.init_vector(self)

    @classmethod
    def from_loaded_data(cls, data):
        """
        Loads an OCP from a dictionary ("ocp_initializer")

        Parameters
        ----------
        data: dict
            A dictionary containing the data to load

        Returns
        -------
        OptimalControlProgram
        """
        for i, model in enumerate(data["bio_model"]):
            model_class = model[0]
            model_initializer = model[1]
            data["bio_model"][i] = model_class(**model_initializer)

        return cls(**data)

    def _check_variable_mapping_consistency_with_node_mapping(
        self, use_states_from_phase_idx, use_controls_from_phase_idx
    ):
        # TODO this feature is broken since the merge with bi_node, fix it
        if (
            list(set(use_states_from_phase_idx)) != use_states_from_phase_idx
            or list(set(use_controls_from_phase_idx)) != use_controls_from_phase_idx
        ):
            raise NotImplementedError("Mapping over phases is broken")

        for i in range(self.n_phases):
            for j in [idx for idx, x in enumerate(use_states_from_phase_idx) if x == i]:
                for key in self.nlp[i].variable_mappings.keys():
                    if key in self.nlp[j].variable_mappings.keys():
                        if (
                            self.nlp[i].variable_mappings[key].to_first.map_idx
                            != self.nlp[j].variable_mappings[key].to_first.map_idx
                            or self.nlp[i].variable_mappings[key].to_second.map_idx
                            != self.nlp[j].variable_mappings[key].to_second.map_idx
                        ):
                            raise RuntimeError(
                                f"The variable mappings must be the same for the mapped phases."
                                f"Mapping on {key} is different between phases {i} and {j}."
                            )
        for i in range(self.n_phases):
            for j in [idx for idx, x in enumerate(use_controls_from_phase_idx) if x == i]:
                for key in self.nlp[i].variable_mappings.keys():
                    if key in self.nlp[j].variable_mappings.keys():
                        if (
                            self.nlp[i].variable_mappings[key].to_first.map_idx
                            != self.nlp[j].variable_mappings[key].to_first.map_idx
                            or self.nlp[i].variable_mappings[key].to_second.map_idx
                            != self.nlp[j].variable_mappings[key].to_second.map_idx
                        ):
                            raise RuntimeError(
                                f"The variable mappings must be the same for the mapped phases."
                                f"Mapping on {key} is different between phases {i} and {j}."
                            )
        return

    def _set_kinematic_phase_mapping(self):
        """
        To add phase_mapping for different kinematic number of states in the ocp. It maps the degrees of freedom
        across phases, so they appear on the same graph.
        """
        dof_names_all_phases = []
        phase_mappings = []  # [[] for _ in range(len(self.nlp))]
        dof_names = []  # [[] for _ in range(len(self.nlp))]
        for i, nlp in enumerate(self.nlp):
            current_dof_mapping = []
            for legend in nlp.model.name_dof:
                if legend in dof_names_all_phases:
                    current_dof_mapping += [dof_names_all_phases.index(legend)]
                else:
                    dof_names_all_phases += [legend]
                    current_dof_mapping += [len(dof_names_all_phases) - 1]
            phase_mappings.append(
                BiMapping(to_first=current_dof_mapping, to_second=list(range(len(current_dof_mapping))))
            )
            dof_names.append([dof_names_all_phases[i] for i in phase_mappings[i].to_first.map_idx])
        return phase_mappings, dof_names

    @staticmethod
    def _check_quaternions_hasattr(biomodels: list[BioModel]) -> list[BioModel]:
        """
        This functions checks if the biomodels have quaternions and if not we set an attribute to nb_quaternion to 0

        Note: this need to be checked as this information is of importance for ODE solvers

        Parameters
        ----------
        biomodels: list[BioModel]
            The list of biomodels to check

        Returns
        -------
        biomodels: list[BioModel]
            The list of biomodels with the attribute nb_quaternion set to 0 if no quaternion is present
        """

        for i, model in enumerate(biomodels):
            if not hasattr(model, "nb_quaternions"):
                setattr(model, "nb_quaternions", 0)

        return biomodels

    def _prepare_option_dict_for_phase(self, name: str, option_dict: OptionDict, option_dict_type: type) -> Any:
        if option_dict is None:
            option_dict = option_dict_type()

        if not isinstance(option_dict, option_dict_type):
            raise RuntimeError(f"{name} should be built from a {option_dict_type.__name__} or a tuple of which")

        option_dict: Any
        if len(option_dict) == 1 and self.n_phases > 1:
            scaling_phase_0 = option_dict[0]
            for i in range(1, self.n_phases):
                option_dict.add("None", [], phase=i)  # Force the creation of the structure internally
                for key in scaling_phase_0.keys():
                    option_dict.add(key, scaling_phase_0[key], phase=i)
        return option_dict

    def _declare_continuity(self) -> None:
        """
        Declare the continuity function for the state variables. By default, the continuity function
        is a constraint, but it declared as an objective if dynamics_type.state_continuity_weight is not None
        """

        for nlp in self.nlp:  # Inner-phase
            self._declare_inner_phase_continuity(nlp)

        for pt in self.phase_transitions:  # Inter-phase
            self._declare_phase_transition_continuity(pt)

    def _declare_inner_phase_continuity(self, nlp: NLP) -> None:
        """Declare the continuity function for the state variables in a phase"""
        if nlp.dynamics_type.skip_continuity:
            return

        if nlp.dynamics_type.state_continuity_weight is None:
            # Continuity as constraints
            penalty = Constraint(
                ConstraintFcn.STATE_CONTINUITY, node=Node.ALL_SHOOTING, penalty_type=PenaltyType.INTERNAL
            )
            penalty.add_or_replace_to_penalty_pool(self, nlp)
            if nlp.ode_solver.is_direct_collocation and nlp.ode_solver.duplicate_starting_point:
                penalty = Constraint(
                    ConstraintFcn.FIRST_COLLOCATION_HELPER_EQUALS_STATE,
                    node=Node.ALL_SHOOTING,
                    penalty_type=PenaltyType.INTERNAL,
                )
                penalty.add_or_replace_to_penalty_pool(self, nlp)

        else:
            # Continuity as objectives
            penalty = Objective(
                ObjectiveFcn.Mayer.STATE_CONTINUITY,
                weight=nlp.dynamics_type.state_continuity_weight,
                quadratic=True,
                node=Node.ALL_SHOOTING,
                penalty_type=PenaltyType.INTERNAL,
            )
            penalty.add_or_replace_to_penalty_pool(self, nlp)

    def _declare_phase_transition_continuity(self, pt):
        """Declare the continuity function for the variables between phases, mainly for the state variables"""
        # Phase transition as constraints
        if pt.type == PhaseTransitionFcn.DISCONTINUOUS:
            return
        # Dynamics must be respected between phases
        pt.name = f"PHASE_TRANSITION ({pt.type.name}) {pt.nodes_phase[0] % self.n_phases}->{pt.nodes_phase[1] % self.n_phases}"
        pt.list_index = -1
        pt.add_or_replace_to_penalty_pool(self, self.nlp[pt.nodes_phase[0]])

    def update_objectives(self, new_objective_function: Objective | ObjectiveList):
        """
        The main user interface to add or modify objective functions in the ocp

        Parameters
        ----------
        new_objective_function: Objective | ObjectiveList
            The objective to add to the ocp
        """

        if isinstance(new_objective_function, Objective):
            self._modify_penalty(new_objective_function)

        elif isinstance(new_objective_function, ObjectiveList):
            for objective_in_phase in new_objective_function:
                for objective in objective_in_phase:
                    self._modify_penalty(objective)

        else:
            raise RuntimeError("new_objective_function must be a Objective or an ObjectiveList")

    def update_parameter_objectives(self, new_objective_function: ParameterObjective | ParameterObjectiveList):
        """
        The main user interface to add or modify a parameter objective functions in the ocp

        Parameters
        ----------
        new_objective_function: ParameterObjective | ParameterObjectiveList
            The parameter objective to add to the ocp
        """

        if isinstance(new_objective_function, ParameterObjective):
            self._modify_parameter_penalty(new_objective_function)

        elif isinstance(new_objective_function, ParameterObjectiveList):
            for objective_in_phase in new_objective_function:
                for objective in objective_in_phase:
                    self._modify_parameter_penalty(objective)

        else:
            raise RuntimeError("new_objective_function must be a ParameterObjective or an ParameterObjectiveList")

    def update_objectives_target(self, target, phase=None, list_index=None):
        """
        Fast accessor to update the target of a specific objective function. To update target of global objective
        (usually defined by parameters), one can pass 'phase=-1'

        Parameters
        ----------
        target: np.ndarray
            The new target of the objective function. The last dimension must be the number of frames
        phase: int
            The phase the objective is in. None is interpreted as zero if the program has one phase. The value -1
            changes the values of ocp.J
        list_index: int
            The objective index
        """

        if phase is None and len(self.nlp) == 1:
            phase = 0

        if list_index is None:
            raise ValueError("'phase' must be defined")

        ObjectiveFunction.update_target(self.nlp[phase] if phase >= 0 else self, list_index, target)

    def update_constraints(self, new_constraints: Constraint | ConstraintList):
        """
        The main user interface to add or modify constraint in the ocp

        Parameters
        ----------
        new_constraints: Constraint | ConstraintList
            The constraint to add to the ocp
        """

        if isinstance(new_constraints, Constraint):
            self._modify_penalty(new_constraints)

        elif isinstance(new_constraints, ConstraintList):
            for constraints_in_phase in new_constraints:
                for constraint in constraints_in_phase:
                    self._modify_penalty(constraint)
        else:
            raise RuntimeError("new_constraint must be a Constraint or a ConstraintList")

    def update_parameter_constraints(self, new_constraint: ParameterConstraint | ParameterConstraintList):
        """
        The main user interface to add or modify a parameter constraint in the ocp

        Parameters
        ----------
        new_constraint: ParameterConstraint | ParameterConstraintList
            The parameter constraint to add to the ocp
        """

        if isinstance(new_constraint, ParameterConstraint):
            # This should work, but was not fully tested
            self._modify_parameter_penalty(new_constraint)
        elif isinstance(new_constraint, ParameterConstraintList):
            for constraint_in_phase in new_constraint:
                for constraint in constraint_in_phase:
                    self._modify_parameter_penalty(constraint)
        else:
            raise RuntimeError("new_constraint must be a ParameterConstraint or a ParameterConstraintList")

    def _declare_parameters(self, parameters: ParameterList):
        """
        The main user interface to add or modify parameters in the ocp

        Parameters
        ----------
        parameters: ParameterList
            The parameters to add to the ocp
        """

        if not isinstance(parameters, ParameterList):
            raise RuntimeError("new_parameter must be a Parameter or a ParameterList")

        self.parameters = ParameterContainer(use_sx=(True if self.cx == SX else False))
        self.parameters.initialize(parameters)
        # The version without time will not be updated later when time is declared
        self.parameters_except_time = ParameterContainer(use_sx=(True if self.cx == SX else False))
        self.parameters_except_time.initialize(parameters)

    def update_bounds(
        self,
        x_bounds: BoundsList = None,
        u_bounds: BoundsList = None,
        parameter_bounds: BoundsList = None,
        a_bounds: BoundsList = None,
    ):
        """
        The main user interface to add bounds in the ocp

        Parameters
        ----------
        x_bounds: BoundsList
            The state bounds to add
        u_bounds: BoundsList
            The control bounds to add
        parameter_bounds: BoundsList
            The parameters bounds to add
        a_bounds: BoundsList
            The algebraic_states variable bounds to add
        """
        for i in range(self.n_phases):
            if x_bounds is not None:
                if not isinstance(x_bounds, BoundsList):
                    raise RuntimeError("x_bounds should be built from a BoundsList")
                origin_phase = 0 if len(x_bounds) == 1 else i
                for key in x_bounds[origin_phase].keys():
                    if key not in self.nlp[i].states.keys() + ["None"]:
                        raise ValueError(
                            f"{key} is not a state variable, please check for typos in the declaration of x_bounds"
                        )
                    self.nlp[i].x_bounds.add(key, x_bounds[origin_phase][key], phase=0)

            if u_bounds is not None:
                if not isinstance(u_bounds, BoundsList):
                    raise RuntimeError("u_bounds should be built from a BoundsList")
                for key in u_bounds.keys():
                    if key not in self.nlp[i].controls.keys() + ["None"]:
                        raise ValueError(
                            f"{key} is not a control variable, please check for typos in the declaration of u_bounds"
                        )
                    origin_phase = 0 if len(u_bounds) == 1 else i
                    self.nlp[i].u_bounds.add(key, u_bounds[origin_phase][key], phase=0)

            if a_bounds is not None:
                if not isinstance(a_bounds, BoundsList):
                    raise RuntimeError("a_bounds should be built from a BoundsList")
                for key in a_bounds.keys():
                    if key not in self.nlp[i].algebraic_states.keys() + ["None"]:
                        raise ValueError(
                            f"{key} is not an algebraic variable, please check for typos in the declaration of a_bounds"
                        )
                    origin_phase = 0 if len(a_bounds) == 1 else i
                    self.nlp[i].a_bounds.add(key, a_bounds[origin_phase][key], phase=0)

        if parameter_bounds is not None:
            if not isinstance(parameter_bounds, BoundsList):
                raise RuntimeError("parameter_bounds should be built from a BoundsList")
            for key in parameter_bounds.keys():
                if key not in self.parameters.keys() + ["None"]:
                    raise ValueError(
                        f"{key} is not a parameter variable, please check for typos in the declaration of parameter_bounds"
                    )
                self.parameter_bounds.add(key, parameter_bounds[key], phase=0)

        for nlp in self.nlp:
            for key in nlp.states.keys():
                if f"{key}_states" in nlp.plot and key in nlp.x_bounds.keys():
                    nlp.plot[f"{key}_states"].bounds = nlp.x_bounds[key]
            for key in nlp.controls.keys():
                if f"{key}_controls" in nlp.plot and key in nlp.u_bounds.keys():
                    nlp.plot[f"{key}_controls"].bounds = nlp.u_bounds[key]
            for key in nlp.algebraic_states.keys():
                if f"{key}_algebraic" in nlp.plot and key in nlp.a_bounds.keys():
                    nlp.plot[f"{key}_algebraic"].bounds = nlp.a_bounds[key]

    def update_initial_guess(
        self,
        x_init: InitialGuessList = None,
        u_init: InitialGuessList = None,
        parameter_init: InitialGuessList = None,
        a_init: InitialGuessList = None,
    ):
        """
        The main user interface to add initial guesses in the ocp

        Parameters
        ----------
        x_init: BoundsList
            The state initial guess to add
        u_init: BoundsList
            The control initial guess to add
        parameter_init: BoundsList
            The parameters initial guess to add
        a_init: BoundsList
            The algebraic_states variable initial guess to add
        """

        for i in range(self.n_phases):
            if x_init:
                if not isinstance(x_init, InitialGuessList):
                    raise RuntimeError("x_init should be built from a InitialGuessList")
                origin_phase = 0 if len(x_init) == 1 else i
                for key in x_init[origin_phase].keys():
                    if key not in self.nlp[i].states.keys() + ["None"]:
                        raise ValueError(
                            f"{key} is not a state variable, please check for typos in the declaration of x_init"
                        )
                    if (
                        not self.nlp[i].ode_solver.is_direct_collocation
                        and x_init[origin_phase].type == InterpolationType.ALL_POINTS
                    ):
                        raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")
                    self.nlp[i].x_init.add(key, x_init[origin_phase][key], phase=0)

            if u_init:
                if not isinstance(u_init, InitialGuessList):
                    raise RuntimeError("u_init should be built from a InitialGuessList")
                origin_phase = 0 if len(u_init) == 1 else i
                for key in u_init.keys():
                    if key not in self.nlp[i].controls.keys() + ["None"]:
                        raise ValueError(
                            f"{key} is not a control variable, please check for typos in the declaration of u_init"
                        )
                    if (
                        not self.nlp[i].ode_solver.is_direct_collocation
                        and x_init[origin_phase].type == InterpolationType.ALL_POINTS
                    ):
                        raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")
                    self.nlp[i].u_init.add(key, u_init[origin_phase][key], phase=0)

            if a_init:
                if not isinstance(a_init, InitialGuessList):
                    raise RuntimeError("a_init should be built from a InitialGuessList")
                origin_phase = 0 if len(a_init) == 1 else i
                for key in a_init[origin_phase].keys():
                    if key not in self.nlp[i].algebraic_states.keys() + ["None"]:
                        raise ValueError(
                            f"{key} is not an algebraic variable, please check for typos in the declaration of a_init"
                        )
                    self.nlp[i].a_init.add(key, a_init[origin_phase][key], phase=0)

        if parameter_init is not None:
            if not isinstance(parameter_init, InitialGuessList):
                raise RuntimeError("parameter_init should be built from a InitialGuessList")
            for key in parameter_init.keys():
                if key not in self.parameters.keys() + ["None"]:
                    raise ValueError(
                        f"{key} is not a parameter variable, please check for typos in the declaration of parameter_init"
                    )
                self.parameter_init.add(key, parameter_init[key], phase=0)

    def add_plot(self, fig_name: str, update_function: Callable, phase: int = -1, **parameters: Any):
        """
        The main user interface to add a new plot to the ocp

        Parameters
        ----------
        fig_name: str
            The name of the figure, it the name already exists, it is merged
        update_function: Callable
            The update function callable using f(states, controls, parameters, **parameters)
        phase: int
            The phase to add the plot to. -1 is the last
        parameters: dict
            Any parameters to pass to the update_function
        """

        if "combine_to" in parameters:
            raise RuntimeError(
                "'combine_to' cannot be specified in add_plot, please use same 'fig_name' to combine plots"
            )

        # --- Solve the program --- #
        if len(self.nlp) == 1:
            phase = 0
        else:
            if phase < 0:
                raise RuntimeError("phase_idx must be specified for multiphase OCP")
        nlp = self.nlp[phase]
        custom_plot = CustomPlot(update_function, **parameters)

        plot_name = "no_name"
        if fig_name in nlp.plot:
            # Make sure we add a unique name in the dict
            custom_plot.combine_to = fig_name

            if fig_name:
                cmp = 0
                while True:
                    plot_name = f"{fig_name}_phase{phase}_{cmp}"
                    if plot_name not in nlp.plot:
                        break
                    cmp += 1
        else:
            plot_name = fig_name

        nlp.plot[plot_name] = custom_plot

    def add_plot_penalty(self, cost_type: CostType = None):
        """
        To add penlaty (objectivs and constraints) plots

        Parameters
        ----------
        cost_type: str
            The name of the penalty to be plotted (objectives, constraints)
        """

        def penalty_color():
            """
            Penalty plot with different name have a different color on the graph
            """
            name_unique_objective = []
            for nlp in self.nlp:
                if cost_type == CostType.OBJECTIVES:
                    penalties = nlp.J
                    penalties_internal = nlp.J_internal
                    penalties_implicit = []
                else:  # Constraints
                    penalties = nlp.g
                    penalties_internal = nlp.g_internal
                    penalties_implicit = nlp.g_implicit

                for penalty in penalties:
                    if not penalty:
                        continue
                    name_unique_objective.append(penalty.name)
                for penalty_internal in penalties_internal:
                    if not penalty_internal:
                        continue
                    name_unique_objective.append(penalty_internal.name)
                for penalty_implicit in penalties_implicit:
                    if not penalty_implicit:
                        continue
                    name_unique_objective.append(penalty_implicit.name)
            color = {}
            for i, name in enumerate(name_unique_objective):
                color[name] = plt.cm.viridis(i / len(name_unique_objective))
            return color

        def compute_penalty_values(t0, phases_dt, node_idx, x, u, p, a, d, penalty):
            """
            Compute the penalty value for the given time, state, control, parameters, penalty and time step

            Parameters
            ----------
            t0: float
                Time at the beginning of the penalty
            phases_dt: list[float]
                List of the time step of each phase
            node_idx: int
                Time index
            x: ndarray
                State vector with intermediate states
            u: ndarray
                Control vector with starting control (and sometimes final control)
            p: ndarray
                Parameters vector
            a: ndarray
                Algebraic states variables vector
            d: ndarray
                numerical timeseries
            penalty: Penalty
                The penalty object containing details on how to compute it

            Returns
            -------
            Values computed for the given time, state, control, parameters, penalty and time step
            """

            weight = PenaltyHelpers.weight(penalty)
            target = PenaltyHelpers.target(penalty, penalty.node_idx.index(node_idx))

            val = penalty.weighted_function_non_threaded[node_idx](t0, phases_dt, x, u, p, a, d, weight, target)
            return sum1(horzcat(val))

        def add_penalty(_penalties):
            for penalty in _penalties:
                if not penalty:
                    continue

                plot_params = {
                    "fig_name": cost_type.name,
                    "update_function": compute_penalty_values,
                    "phase": i_phase,
                    "penalty": penalty,
                    "color": color[penalty.name],
                    "label": penalty.name,
                    "compute_derivative": penalty.derivative or penalty.explicit_derivative or penalty.integrate,
                    "integration_rule": penalty.integration_rule,
                    "plot_type": PlotType.POINT,
                    "node_idx": penalty.node_idx,
                }

                self.add_plot(**plot_params)

            return

        if cost_type is None:
            cost_type = CostType.ALL

        color = penalty_color()
        for i_phase, nlp in enumerate(self.nlp):
            if cost_type == CostType.OBJECTIVES:
                penalties = nlp.J
                penalties_internal = nlp.J_internal
                penalties_implicit = []
            elif cost_type == CostType.CONSTRAINTS:
                penalties = nlp.g
                penalties_internal = nlp.g_internal
                penalties_implicit = nlp.g_implicit
            elif cost_type == CostType.ALL:
                self.add_plot_penalty(CostType.OBJECTIVES)
                self.add_plot_penalty(CostType.CONSTRAINTS)
                return
            else:
                raise RuntimeError(f"cost_type parameter {cost_type} is not valid.")

            add_penalty(penalties)
            add_penalty(penalties_internal)
            add_penalty(penalties_implicit)
        return

    def add_plot_ipopt_outputs(self):
        self.plot_ipopt_outputs = True

    def add_plot_check_conditioning(self):
        self.plot_check_conditioning = True

    def save_intermediary_ipopt_iterations(self, path_to_results, result_file_name, nb_iter_save):
        self.save_ipopt_iterations_info = SaveIterationsInfo(path_to_results, result_file_name, nb_iter_save)

    def prepare_plots(
        self,
        automatically_organize: bool = True,
        show_bounds: bool = False,
        shooting_type: Shooting = Shooting.MULTIPLE,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
    ) -> PlotOcp:
        """
        Create all the plots associated with the OCP

        Parameters
        ----------
        automatically_organize: bool
            If the graphs should be parsed on the screen
        show_bounds: bool
            If the ylim should fit the bounds
        shooting_type: Shooting
            What type of integration
        integrator: SolutionIntegrator
            Use the ode defined by OCP or use a separate integrator provided by scipy

        Returns
        -------
        The PlotOcp class
        """

        return PlotOcp(
            self,
            automatically_organize=automatically_organize,
            show_bounds=show_bounds,
            shooting_type=shooting_type,
            integrator=integrator,
            dummy_phase_times=OptimizationVectorHelper.extract_step_times(self, casadi.DM(np.ones(self.n_phases))),
        )

    def check_conditioning(self):
        """
        Visualisation of jacobian and hessian contraints and hessian objective for each phase at initial time
        """
        check_conditioning(self)

    def solve(
        self, solver: GenericSolver = None, warm_start: Solution = None, expand_during_shake_tree=False
    ) -> Solution:
        """
        Call the solver to actually solve the ocp

        Parameters
        ----------
        solver: GenericSolver
            The solver which will be used to solve the ocp
        warm_start: Solution
            The solution to pass to the warm start method
        expand_during_shake_tree: bool
            If the tree should be expanded during the shake phase

        Returns
        -------
        The optimized solution structure
        """

        if solver is None:
            solver = Solver.IPOPT()

        if self.ocp_solver is None:
            if solver.type == SolverType.IPOPT:
                from ..interfaces.ipopt_interface import IpoptInterface

                self.ocp_solver = IpoptInterface(self)

            elif solver.type == SolverType.SQP:
                from ..interfaces.sqp_interface import SQPInterface

                self.ocp_solver = SQPInterface(self)

            elif solver.type == SolverType.ACADOS:
                from ..interfaces.acados_interface import AcadosInterface

                self.ocp_solver = AcadosInterface(self, solver)

            elif solver.type == SolverType.NONE:
                raise RuntimeError("Invalid solver")

        if warm_start is not None:
            self.set_warm_start(sol=warm_start)

        if self._is_warm_starting:
            if solver.type == SolverType.IPOPT:
                solver.set_warm_start_options(1e-10)

        self.ocp_solver.opts = solver

        self.ocp_solver.solve(expand_during_shake_tree=expand_during_shake_tree)
        self._is_warm_starting = False

        return Solution.from_dict(self, self.ocp_solver.get_optimized_value())

    def set_warm_start(self, sol: Solution):
        """
        Modify x and u initial guess based on a solution.

        Parameters
        ----------
        sol: Solution
            The solution to initiate the OCP from
        """

        state = sol.decision_states(to_merge=SolutionMerge.NODES)
        ctrl = sol.decision_controls(to_merge=SolutionMerge.NODES)
        param = sol.parameters

        u_init_guess = InitialGuessList()
        x_init_guess = InitialGuessList()
        param_init_guess = InitialGuessList()

        for i in range(self.n_phases):
            x_interp = (
                InterpolationType.EACH_FRAME
                if self.nlp[i].ode_solver.is_direct_shooting
                else InterpolationType.ALL_POINTS
            )
            if self.n_phases == 1:
                for key in state:
                    x_init_guess.add(key, state[key], interpolation=x_interp, phase=0)
                for key in ctrl:
                    u_init_guess.add(key, ctrl[key], interpolation=InterpolationType.EACH_FRAME, phase=0)

            else:
                for key in state[i]:
                    x_init_guess.add(key, state[i][key], interpolation=x_interp, phase=i)
                for key in ctrl[i]:
                    u_init_guess.add(key, ctrl[i][key], interpolation=InterpolationType.EACH_FRAME, phase=i)

        for key in param:
            param_init_guess.add(key, param[key], name=key)

        self.update_initial_guess(x_init=x_init_guess, u_init=u_init_guess, parameter_init=param_init_guess)

        if self.ocp_solver:
            self.ocp_solver.set_lagrange_multiplier(sol)

        self._is_warm_starting = True

    def print(
        self,
        to_console: bool = True,
        to_graph: bool = True,
    ):
        if to_console:
            display_console = OcpToConsole(self)
            display_console.print()

        if to_graph:
            display_graph = OcpToGraph(self)
            display_graph.print()

    def _define_time(
        self, phase_time: int | float | list | tuple, objective_functions: ObjectiveList, constraints: ConstraintList
    ):
        """
        Declare the phase_time vector in v. If objective_functions or constraints defined a time optimization,
        a sanity check is perform and the values of initial guess and bounds for these particular phases

        Parameters
        ----------
        phase_time: int | float | list | tuple
            The time of all the phases
        objective_functions: ObjectiveList
            All the objective functions. It is used to scan if any time optimization was defined
        constraints: ConstraintList
            All the constraint functions. It is used to scan if any free time was defined
        """

        def define_parameters_phase_time(
            ocp: OptimalControlProgram,
            penalty_functions: ObjectiveList | ConstraintList,
            _has_penalty: list = None,
        ) -> list:
            """
            Sanity check to ensure that only one time optimization is defined per phase. It also creates the time vector
            for initial guesses and bounds

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            penalty_functions: ObjectiveList | ConstraintList
                The list to parse to ensure no double free times are declared
            _has_penalty: list[bool]
                If a penalty was previously found. This should be None on the first call to ensure proper initialization

            Returns
            --------
            The state of has_penalty
            """

            if _has_penalty is None:
                _has_penalty = [False] * ocp.n_phases

            for i, penalty_functions_phase in enumerate(penalty_functions):
                key = f"dt_phase_{i}"
                if key not in dt_bounds.keys():
                    # This means there is a mapping on this value
                    continue

                for pen_fun in penalty_functions_phase:
                    if not pen_fun:
                        continue
                    if pen_fun.type in (ObjectiveFcn.Mayer.MINIMIZE_TIME, ConstraintFcn.TIME_CONSTRAINT):
                        if _has_penalty[i]:
                            raise RuntimeError("Time constraint/objective cannot be declared more than once per phase")
                        _has_penalty[i] = True

                        if pen_fun.type.get_type() == ConstraintFunction:
                            _min = pen_fun.min_bound if pen_fun.min_bound else 0
                            _max = pen_fun.max_bound if pen_fun.max_bound else inf
                        else:
                            _min = (
                                pen_fun.extra_parameters["min_bound"] if "min_bound" in pen_fun.extra_parameters else 0
                            )
                            _max = (
                                pen_fun.extra_parameters["max_bound"]
                                if "max_bound" in pen_fun.extra_parameters
                                else inf
                            )
                        dt_bounds[key]["min"] = _min / self.nlp[i].ns
                        dt_bounds[key]["max"] = _max / self.nlp[i].ns

            return _has_penalty

        self.phase_time = phase_time if isinstance(phase_time, (tuple, list)) else [phase_time]

        self.dt_parameter = ParameterList(use_sx=(True if self.cx == SX else False))
        for i_phase in range(self.n_phases):
            if i_phase != self.time_phase_mapping.to_second.map_idx[i_phase]:
                self.dt_parameter.add_a_copied_element(self.time_phase_mapping.to_second.map_idx[i_phase])
            else:
                self.dt_parameter.add(
                    name=f"dt_phase{i_phase}",
                    function=lambda model, values: None,
                    size=1,
                    mapping=BiMapping([1], [1]),
                    scaling=VariableScaling("dt", np.ones((1,))),
                    allow_reserved_name=True,
                )

        dt_bounds = {}
        dt_initial_guess = {}
        dt_cx = []
        # dt_mx = []
        for i_phase in range(self.n_phases):
            if i_phase == self.time_phase_mapping.to_second.map_idx[i_phase]:
                dt = self.phase_time[i_phase] / self.nlp[i_phase].ns
                dt_bounds[f"dt_phase_{i_phase}"] = {"min": dt, "max": dt}
                dt_initial_guess[f"dt_phase_{i_phase}"] = dt

            dt_cx.append(self.dt_parameter[self.time_phase_mapping.to_second.map_idx[i_phase]].cx)
            # dt_mx.append(self.dt_parameter[self.time_phase_mapping.to_second.map_idx[i_phase]].mx)

        has_penalty = define_parameters_phase_time(self, objective_functions)
        define_parameters_phase_time(self, constraints, has_penalty)

        # Add to the nlp
        NLP.add(self, "time_index", self.time_phase_mapping.to_second.map_idx, True)
        NLP.add(self, "time_cx", self.cx.sym("time", 1, 1), True)
        # NLP.add(self, "time_mx", MX.sym("time", 1, 1), True)
        NLP.add(self, "dt", dt_cx, False)
        # NLP.add(self, "dt_mx", dt_mx, False)
        NLP.add(self, "tf", [nlp.dt * max(nlp.ns, 1) for nlp in self.nlp], False)
        # NLP.add(self, "tf_mx", [nlp.dt_mx * max(nlp.ns, 1) for nlp in self.nlp], False)

        self.dt_parameter_bounds = Bounds(
            "dt_bounds",
            min_bound=[v["min"] for v in dt_bounds.values()],
            max_bound=[v["max"] for v in dt_bounds.values()],
            interpolation=InterpolationType.CONSTANT,
        )
        self.dt_parameter_initial_guess = InitialGuess(
            "dt_initial_guess", initial_guess=[v for v in dt_initial_guess.values()]
        )

    def _define_numerical_timeseries(self, dynamics):
        """
        Declare the numerical_timeseries symbolic variables.

        Parameters
        ----------
        dynamics:
            The dynamics for each phase.
        """

        numerical_timeseries = []
        for i_phase, nlp in enumerate(self.nlp):
            numerical_timeseries += [OptimizationVariableList(self.cx, dynamics[i_phase].phase_dynamics)]
            if dynamics[i_phase].numerical_data_timeseries is not None:
                for key in dynamics[i_phase].numerical_data_timeseries.keys():
                    variable_shape = dynamics[i_phase].numerical_data_timeseries[key].shape
                    for i_component in range(variable_shape[1] if len(variable_shape) > 1 else 1):
                        cx = self.cx.sym(
                            f"{key}_phase{i_phase}_{i_component}_cx",
                            variable_shape[0],
                        )

                        numerical_timeseries[-1].append(
                            name=f"{key}_{i_component}",
                            cx=[cx, cx, cx],
                            bimapping=BiMapping(
                                Mapping(list(range(variable_shape[0]))), Mapping(list(range(variable_shape[0])))
                            ),
                        )

        # Add to the nlp
        NLP.add(self, "numerical_timeseries", numerical_timeseries, True)

    def _modify_penalty(self, new_penalty: PenaltyOption | Parameter):
        """
        The internal function to modify a penalty.

        Parameters
        ----------
        new_penalty: PenaltyOption | Parameter
            Any valid option to add to the program
        """

        if not new_penalty:
            return
        phase_idx = new_penalty.phase
        new_penalty.add_or_replace_to_penalty_pool(self, self.nlp[phase_idx])

        self.program_changed = True

    def _modify_parameter_penalty(self, new_penalty: PenaltyOption | Parameter):
        """
        The internal function to modify a parameter penalty.

        Parameters
        ----------
        new_penalty: PenaltyOption | Parameter
            Any valid option to add to the program
        """

        if not new_penalty:
            return

        new_penalty.add_or_replace_to_penalty_pool(self, self.nlp[new_penalty.phase])
        self.program_changed = True

    def node_time(self, phase_idx: int, node_idx: int):
        """
        Gives the time of the node node_idx of from the phase phase_idx

        Parameters
        ----------
        phase_idx: int
          Index of the phase
        node_idx: int
          Index of the node

        Returns
        -------
        The node time node_idx from the phase phase_idx
        """
        if phase_idx > self.n_phases - 1:
            raise ValueError(f"phase_index out of range [0:{self.n_phases}]")
        if node_idx < 0 or node_idx > self.nlp[phase_idx].ns:
            raise ValueError(f"node_index out of range [0:{self.nlp[phase_idx].ns}]")
        previous_phase_time = sum([nlp.tf for nlp in self.nlp[:phase_idx]])

        return previous_phase_time + self.nlp[phase_idx].tf * node_idx / self.nlp[phase_idx].ns

    def _set_default_ode_solver(self):
        """
        Set the default ode solver to RK4
        """
        return OdeSolver.RK4()

    def _set_internal_algebraic_states(self):
        """
        Set the algebraic states to their internal values
        """
        self._a_init = InitialGuessList()
        self._a_bounds = BoundsList()
        self._a_scaling = VariableScalingList()
        return self._a_init, self._a_bounds, self._a_scaling

    def _set_nlp_is_stochastic(self):
        """
        Set the is_stochastic variable to False
        because it's not relevant for traditional OCP,
        only relevant for StochasticOptimalControlProgram

        Note
        ----
        This method is thus overriden in StochasticOptimalControlProgram
        """
        NLP.add(self, "is_stochastic", False, True)
