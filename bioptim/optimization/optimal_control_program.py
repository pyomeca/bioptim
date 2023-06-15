from typing import Callable, Any
import os
import sys
import pickle
from copy import deepcopy
from math import inf

import numpy as np
import biorbd_casadi as biorbd
import casadi
from casadi import MX, SX, Function, sum1, horzcat
from matplotlib import pyplot as plt

from .non_linear_program import NonLinearProgram as NLP
from .optimization_vector import OptimizationVector
from ..dynamics.configure_problem import DynamicsList, Dynamics
from ..dynamics.ode_solver import OdeSolver, OdeSolverBase
from ..dynamics.configure_problem import ConfigureProblem
from ..gui.plot import CustomPlot, PlotOcp
from ..gui.graph import OcpToConsole, OcpToGraph
from ..interfaces.biomodel import BioModel
from ..interfaces.solver_options import Solver
from ..limits.constraints import (
    ConstraintFunction,
    ConstraintFcn,
    ConstraintList,
    Constraint,
)
from ..limits.phase_transition import PhaseTransitionList, PhaseTransitionFcn
from ..limits.multinode_constraint import MultinodeConstraintList
from ..limits.multinode_objective import MultinodeObjectiveList
from ..limits.objective_functions import ObjectiveFcn, ObjectiveList, Objective
from ..limits.path_conditions import BoundsList, Bounds
from ..limits.path_conditions import InitialGuess, InitialGuessList
from ..limits.penalty import PenaltyOption
from ..limits.objective_functions import ObjectiveFunction
from ..misc.__version__ import __version__
from ..misc.enums import (
    ControlType,
    SolverType,
    Shooting,
    PlotType,
    CostType,
    SolutionIntegrator,
    IntegralApproximation,
    InterpolationType,
    PenaltyType,
    Node,
)
from ..misc.mapping import BiMappingList, Mapping, BiMapping, NodeMappingList
from ..misc.options import OptionDict
from ..misc.utils import check_version
from ..optimization.parameters import ParameterList, Parameter
from ..optimization.solution import Solution
from ..optimization.variable_scaling import VariableScalingList
from ..gui.check_conditioning import check_conditioning


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
    isdef_x_init: bool
        If the initial condition of the states are set
    isdef_x_bounds: bool
        If the bounds of the states are set
    isdef_u_init: bool
        If the initial condition of the controls are set
    isdef_u_bounds: bool
        If the bounds of the controls are set
    nlp: NLP
        All the phases of the ocp
    n_phases: int | list | tuple
        The number of phases of the ocp
    n_threads: int
        The number of thread to use if using multithreading
    original_phase_time: list[float]
        The time vector as sent by the user
    original_values: dict
        A copy of the ocp as it is after defining everything
    phase_transitions: list[PhaseTransition]
        The list of transition constraint between phases
    ocp_solver: SolverInterface
        A reference to the ocp solver
    v: OptimizationVector
        The variable optimization holder
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
    solve(self, solver: Solver, show_online_optim: bool, solver_options: dict) -> Solution
        Call the solver to actually solve the ocp
    save(self, sol: Solution, file_path: str, stand_alone: bool = False)
        Save the ocp and solution structure to the hard drive. It automatically create the required
        folder if it does not exists. Please note that biorbd is required to load back this structure.
    @staticmethod
    load(file_path: str) -> list
        Reload a previous optimization (*.bo) saved using save
    _define_time(self, phase_time: float | tuple, objective_functions: ObjectiveList, constraints: ConstraintList)
        Declare the phase_time vector in v. If objective_functions or constraints defined a time optimization,
        a sanity check is perform and the values of initial guess and bounds for these particular phases
    __modify_penalty(self, new_penalty: PenaltyOption | Parameter)
        The internal function to modify a penalty. It is also stored in the original_values, meaning that if one
        overrides an objective only the latter is preserved when saved
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
        parameters: Parameter | ParameterList = None,
        external_forces: list[list[Any], ...] | tuple[list[Any], ...] = None,
        ode_solver: list | OdeSolverBase | OdeSolver = None,
        control_type: ControlType | list = ControlType.CONSTANT,
        variable_mappings: BiMappingList = None,
        parameter_mappings: BiMappingList = None,
        node_mappings: NodeMappingList = None,
        plot_mappings: Mapping = None,
        phase_transitions: PhaseTransitionList = None,
        multinode_constraints: MultinodeConstraintList = None,
        multinode_objectives: MultinodeObjectiveList = None,
        x_scaling: VariableScalingList = None,
        xdot_scaling: VariableScalingList = None,
        u_scaling: VariableScalingList = None,
        state_continuity_weight: float = None,  # TODO: docstring
        n_threads: int = 1,
        use_sx: bool = False,
        skip_continuity: bool = False,
        assume_phase_dynamics: bool = False,
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
        parameters: Parameter | ParameterList
            All the parameters to optimize of the program
        external_forces: list[list, ...] | tuple[list, ...]
            The external forces acting on the center of mass of the segments specified in the bioMod
        ode_solver: OdeSolverBase
            The solver for the ordinary differential equations
        control_type: ControlType
            The type of controls for each phase
        variable_mappings: BiMappingList
            The mapping to apply on variables
        parameter_mappings: BiMappingList
            The mapping to apply on parameters
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
        skip_continuity: bool
            This is mainly for internal purposes when creating an OCP not destined to be solved
        assume_phase_dynamics: bool
            If the dynamics of for each shooting node in phases are assumed to be the same
        """

        if not isinstance(bio_model, (list, tuple)):
            bio_model = [bio_model]

        bio_model = self._check_quaternions_hasattr(bio_model)

        self.version = {"casadi": casadi.__version__, "biorbd": biorbd.__version__, "bioptim": __version__}
        self.n_phases = len(bio_model)

        if isinstance(dynamics, Dynamics):
            dynamics_type_tp = DynamicsList()
            dynamics_type_tp.add(dynamics)
            dynamics = dynamics_type_tp
        elif not isinstance(dynamics, DynamicsList):
            raise RuntimeError("dynamics should be a Dynamics or a DynamicsList")

        self.original_values = {
            "bio_model": [m.serialize() for m in bio_model],
            "dynamics": dynamics,
            "n_shooting": n_shooting,
            "phase_time": phase_time,
            "x_init": x_init,
            "u_init": u_init,
            "x_bounds": x_bounds,
            "u_bounds": u_bounds,
            "x_scaling": x_scaling,
            "xdot_scaling": xdot_scaling,
            "u_scaling": u_scaling,
            "objective_functions": ObjectiveList(),
            "constraints": ConstraintList(),
            "parameters": ParameterList(),
            "external_forces": external_forces,
            "ode_solver": ode_solver,
            "control_type": control_type,
            "variable_mappings": variable_mappings,
            "parameter_mappings": parameter_mappings,
            "node_mappings": node_mappings,
            "plot_mappings": plot_mappings,
            "phase_transitions": phase_transitions,
            "multinode_constraints": multinode_constraints,
            "multinode_objectives": multinode_objectives,
            "state_continuity_weight": state_continuity_weight,
            "n_threads": n_threads,
            "use_sx": use_sx,
            "assume_phase_dynamics": assume_phase_dynamics,
        }

        # Check integrity of arguments
        if not isinstance(n_threads, int) or isinstance(n_threads, bool) or n_threads < 1:
            raise RuntimeError("n_threads should be a positive integer greater or equal than 1")

        ns = n_shooting
        if not isinstance(ns, int) or ns < 2:
            if isinstance(ns, (tuple, list)):
                if sum([True for i in ns if not isinstance(i, int) and not isinstance(i, bool)]) != 0:
                    raise RuntimeError("n_shooting should be a positive integer (or a list of) greater or equal than 2")
            else:
                raise RuntimeError("n_shooting should be a positive integer (or a list of) greater or equal than 2")

        if not isinstance(phase_time, (int, float)):
            if isinstance(phase_time, (tuple, list)):
                if sum([True for i in phase_time if not isinstance(i, (int, float))]) != 0:
                    raise RuntimeError("phase_time should be a number or a list of number")
            else:
                raise RuntimeError("phase_time should be a number or a list of number")

        if x_bounds is None:
            x_bounds = BoundsList()
        elif not isinstance(x_bounds, BoundsList):
            raise RuntimeError("x_bounds should be built from a BoundsList")

        if u_bounds is None:
            u_bounds = BoundsList()
        elif not isinstance(u_bounds, BoundsList):
            raise RuntimeError("u_bounds should be built from a BoundsList")

        if x_init is None:
            x_init = InitialGuessList()
        if not isinstance(x_init, InitialGuessList):
            raise RuntimeError("x_init should be built from a InitialGuessList")

        if u_init is None:
            u_init = InitialGuessList()
        if not isinstance(u_init, InitialGuessList):
            raise RuntimeError("u_init should be built from a InitialGuessList")

        x_bounds = self._prepare_option_dict_for_phase("x_bounds", x_bounds, BoundsList)
        u_bounds = self._prepare_option_dict_for_phase("u_bounds", u_bounds, BoundsList)

        x_init = self._prepare_option_dict_for_phase("x_init", x_init, InitialGuessList)
        u_init = self._prepare_option_dict_for_phase("u_init", u_init, InitialGuessList)

        x_scaling = self._prepare_option_dict_for_phase("x_scaling", x_scaling, VariableScalingList)
        xdot_scaling = self._prepare_option_dict_for_phase("xdot_scaling", xdot_scaling, VariableScalingList)
        u_scaling = self._prepare_option_dict_for_phase("u_scaling", u_scaling, VariableScalingList)

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
            parameters = ParameterList()
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

        if ode_solver is None:
            ode_solver = OdeSolver.RK4()
        elif not isinstance(ode_solver, OdeSolverBase):
            raise RuntimeError("ode_solver should be built an instance of OdeSolver")

        if not isinstance(use_sx, bool):
            raise RuntimeError("use_sx should be a bool")

        # Type of CasADi graph
        self.cx = SX if use_sx else MX

        # If the dynamics should be declared individually for each node of the phase or not
        self.assume_phase_dynamics = assume_phase_dynamics

        # Declare optimization variables
        self.program_changed = True
        self.J = []
        self.J_internal = []
        self.g = []
        self.g_internal = []
        self.g_implicit = []

        self.v = OptimizationVector(self)

        # nlp is the core of a phase
        self.nlp = [NLP(self.assume_phase_dynamics) for _ in range(self.n_phases)]
        NLP.add(self, "model", bio_model, False)
        NLP.add(self, "phase_idx", [i for i in range(self.n_phases)], False)

        # Define some aliases
        NLP.add(self, "ns", n_shooting, False)
        for nlp in self.nlp:
            if nlp.ns < 1:
                raise RuntimeError("Number of shooting points must be at least 1")

        self.n_threads = n_threads
        NLP.add(self, "n_threads", n_threads, True)
        self.ocp_solver = None
        self.is_warm_starting = False

        # External forces
        if external_forces is not None:
            NLP.add(self, "external_forces", external_forces, False)

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

        # Prepare the parameters to optimize
        self.phase_transitions = []
        if len(parameters) > 0:
            self.update_parameters(parameters)

        # Prepare the parameter mappings
        if parameter_mappings is None:
            parameter_mappings = BiMappingList()
        if "time" not in parameter_mappings.keys():
            parameter_mappings.add(
                "time", to_second=[i for i in range(self.n_phases)], to_first=[i for i in range(self.n_phases)]
            )
        self.parameter_mappings = parameter_mappings

        # Declare the time to optimize
        self._define_time(phase_time, objective_functions, constraints)

        # Prepare path constraints and dynamics of the program
        NLP.add(self, "dynamics_type", dynamics, False)
        NLP.add(self, "ode_solver", ode_solver, True)
        NLP.add(self, "control_type", control_type, True)

        # Prepare the variable mappings
        if variable_mappings is None:
            variable_mappings = BiMappingList()

        variable_mappings = variable_mappings.variable_mapping_fill_phases(self.n_phases)
        NLP.add(self, "variable_mappings", variable_mappings, True)

        # Do not copy singleton since x_scaling was already dealt with before
        NLP.add(self, "x_scaling", x_scaling, True)
        NLP.add(self, "xdot_scaling", xdot_scaling, True)
        NLP.add(self, "u_scaling", u_scaling, True)

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

        # Prepare the dynamics
        for i in range(self.n_phases):
            self.nlp[i].initialize(self.cx)
            ConfigureProblem.initialize(self, self.nlp[i])
            self.nlp[i].ode_solver.prepare_dynamic_integrator(self, self.nlp[i])

        self.isdef_x_init = False
        self.isdef_u_init = False
        self.isdef_x_bounds = False
        self.isdef_u_bounds = False

        self.update_bounds(x_bounds, u_bounds)
        self.update_initial_guess(x_init, u_init)
        # Define the actual NLP problem
        self.v.define_ocp_shooting_points()

        # Define continuity constraints
        # Prepare phase transitions (Reminder, it is important that parameters are declared before,
        # otherwise they will erase the phase_transitions)
        self.phase_transitions = phase_transitions.prepare_phase_transitions(self, state_continuity_weight)
        multinode_constraints.add_or_replace_to_penalty_pool(self)
        multinode_objectives.add_or_replace_to_penalty_pool(self)

        # Skipping creates a valid but unsolvable OCP class
        if not skip_continuity:
            self._declare_continuity(state_continuity_weight)

        # Prepare constraints
        self.update_constraints(self.implicit_constraints)
        self.update_constraints(constraints)

        # Prepare objectives
        self.update_objectives(objective_functions)

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

    def _declare_continuity(self, state_continuity_weight: float = None) -> None:
        """
        Declare the continuity function for the state variables. By default, the continuity function
        is a constraint, but it declared as an objective if  state_continuity_weight is not None

        Parameters
        ----------
        state_continuity_weight:
            The weight on continuity objective. If it is not None, then the continuity are objective
            instead of constraints
        """

        for nlp in self.nlp:  # Inner-phase
            if state_continuity_weight is None:
                # Continuity as constraints
                if self.assume_phase_dynamics:
                    penalty = Constraint(
                        ConstraintFcn.CONTINUITY, node=Node.ALL_SHOOTING, penalty_type=PenaltyType.INTERNAL
                    )
                    penalty.add_or_replace_to_penalty_pool(self, nlp)
                else:
                    for shooting_node in range(nlp.ns):
                        penalty = Constraint(
                            ConstraintFcn.CONTINUITY, node=shooting_node, penalty_type=PenaltyType.INTERNAL
                        )
                        penalty.add_or_replace_to_penalty_pool(self, nlp)
            else:
                # Continuity as objectives
                if self.assume_phase_dynamics:
                    penalty = Objective(
                        ObjectiveFcn.Mayer.CONTINUITY,
                        weight=state_continuity_weight,
                        quadratic=True,
                        node=Node.ALL_SHOOTING,
                        penalty_type=PenaltyType.INTERNAL,
                    )
                    penalty.add_or_replace_to_penalty_pool(self, nlp)
                else:
                    for shooting_point in range(nlp.ns):
                        penalty = Objective(
                            ObjectiveFcn.Mayer.CONTINUITY,
                            weight=state_continuity_weight,
                            quadratic=True,
                            node=shooting_point,
                            penalty_type=PenaltyType.INTERNAL,
                        )
                        penalty.add_or_replace_to_penalty_pool(self, nlp)

        for pt in self.phase_transitions:
            # Phase transition as constraints
            if pt.type == PhaseTransitionFcn.DISCONTINUOUS:
                continue
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
            self.__modify_penalty(new_objective_function)

        elif isinstance(new_objective_function, ObjectiveList):
            for objective_in_phase in new_objective_function:
                for objective in objective_in_phase:
                    self.__modify_penalty(objective)

        else:
            raise RuntimeError("new_objective_function must be a Objective or an ObjectiveList")

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

    def update_constraints(self, new_constraint: Constraint | ConstraintList):
        """
        The main user interface to add or modify constraint in the ocp

        Parameters
        ----------
        new_constraint: Constraint | ConstraintList
            The constraint to add to the ocp
        """

        if isinstance(new_constraint, Constraint):
            self.__modify_penalty(new_constraint)

        elif isinstance(new_constraint, ConstraintList):
            for constraints_in_phase in new_constraint:
                for constraint in constraints_in_phase:
                    self.__modify_penalty(constraint)
        else:
            raise RuntimeError("new_constraint must be a Constraint or a ConstraintList")

    def update_parameters(self, new_parameters: Parameter | ParameterList):
        """
        The main user interface to add or modify parameters in the ocp

        Parameters
        ----------
        new_parameters: Parameter | ParameterList
            The parameters to add to the ocp
        """

        if isinstance(new_parameters, Parameter):
            self.__modify_penalty(new_parameters)

        elif isinstance(new_parameters, ParameterList):
            for parameter in new_parameters:
                self.__modify_penalty(parameter)
        else:
            raise RuntimeError("new_parameter must be a Parameter or a ParameterList")

    def update_bounds(self, x_bounds: Bounds | BoundsList = BoundsList(), u_bounds: Bounds | BoundsList = BoundsList()):
        """
        The main user interface to add bounds in the ocp

        Parameters
        ----------
        x_bounds: Bounds | BoundsList
            The state bounds to add
        u_bounds: Bounds | BoundsList
            The control bounds to add
        """
        for i in range(self.n_phases):
            if x_bounds:
                if not isinstance(x_bounds, BoundsList):
                    raise RuntimeError("x_bounds should be built from a BoundsList")
                origin_phase = 0 if len(x_bounds) == 1 else i
                for key in x_bounds[origin_phase].keys():
                    self.nlp[i].x_bounds.add(key, x_bounds[origin_phase][key], phase=0)
                self.isdef_x_bounds = True

            if u_bounds:
                if not isinstance(u_bounds, BoundsList):
                    raise RuntimeError("u_bounds should be built from a BoundsList")
                for key in u_bounds.keys():
                    origin_phase = 0 if len(u_bounds) == 1 else i
                    self.nlp[i].u_bounds.add(key, u_bounds[origin_phase][key], phase=0)
                self.isdef_u_bounds = True

        if self.isdef_x_bounds and self.isdef_u_bounds:
            self.v.define_ocp_bounds()

        for nlp in self.nlp:
            for key in nlp.states:
                if f"{key}_states" in nlp.plot and key in nlp.x_bounds:
                    nlp.plot[f"{key}_states"].bounds = nlp.x_bounds[key]
            for key in nlp.controls:
                if f"{key}_controls" in nlp.plot and key in nlp.u_bounds:
                    nlp.plot[f"{key}_controls"].bounds = nlp.u_bounds[key]

    def update_initial_guess(
        self,
        x_init: InitialGuessList = None,
        u_init: InitialGuessList = None,
        param_init: InitialGuessList = None,
    ):
        """
        The main user interface to add initial guesses in the ocp

        Parameters
        ----------
        x_init: Bounds | BoundsList
            The state initial guess to add
        u_init: Bounds | BoundsList
            The control initial guess to add
        param_init: Bounds | BoundsList
            The parameters initial guess to add
        """

        for i in range(self.n_phases):
            if x_init:
                if not isinstance(x_init, InitialGuessList):
                    raise RuntimeError("x_init should be built from a InitialGuessList")
                origin_phase = 0 if len(x_init) == 1 else i
                for key in x_init[origin_phase].keys():
                    self.nlp[i].x_init.add(key, x_init[origin_phase][key], phase=0)
                self.isdef_x_init = True

            if u_init:
                if not isinstance(u_init, InitialGuessList):
                    raise RuntimeError("u_init should be built from a InitialGuessList")
                for key in u_init.keys():
                    origin_phase = 0 if len(u_init) == 1 else i
                    self.nlp[i].u_init.add(key, u_init[origin_phase][key], phase=0)
                self.isdef_u_init = True

        if param_init is None:
            param_init = InitialGuessList()

        for key in param_init.keys():
            if not key:
                raise ValueError("update_initial_guess must specify a name for the parameters")
            try:
                idx = self.v.parameters_in_list.index(key)
                self.v.parameters_in_list[idx].initial_guess.init = param_init[key].init
            except ValueError:
                raise ValueError("update_initial_guess cannot declare new parameters")

        if self.isdef_x_init and self.isdef_u_init:
            self.v.define_ocp_initial_guess()

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

        def compute_penalty_values(t, x, u, p, penalty, dt: int | Callable):
            """
            Compute the penalty value for the given time, state, control, parameters, penalty and time step

            Parameters
            ----------
            t: int
                Time index
            x: ndarray
                State vector with intermediate states
            u: ndarray
                Control vector with starting control (and sometimes final control)
            p: ndarray
                Parameters vector
            penalty: Penalty
                The penalty object containing details on how to compute it
            dt: float, Callable
                Time step for the whole interval

            Returns
            -------
            Values computed for the given time, state, control, parameters, penalty and time step
            """
            if len(x.shape) < 2:
                x = x.reshape((-1, 1))

            if len(u.shape) < 2:
                u = u.reshape((-1, 1))

            # if time is parameter of the ocp, we need to evaluate with current parameters
            if isinstance(dt, Function):
                dt = dt(p)
            # The division is to account for the steps in the integration. The else is for Mayer term
            dt = dt / (x.shape[1] - 1) if x.shape[1] > 1 else dt
            if not isinstance(penalty.dt, (float, int)):
                if dt.shape[0] > 1:
                    dt = dt[penalty.phase]

            _target = (
                np.hstack([p[..., penalty.node_idx.index(t)] for p in penalty.target])
                if penalty.target is not None and isinstance(t, int)
                else []
            )

            x_scaling = np.concatenate(
                [
                    np.repeat(self.nlp[penalty.phase].x_scaling[key].scaling[:, np.newaxis], x.shape[1], axis=1)
                    for key in self.nlp[penalty.phase].states
                ]
            )
            x /= x_scaling

            if u.size != 0:
                u_scaling = np.concatenate(
                    [
                        np.repeat(self.nlp[penalty.phase].u_scaling[key].scaling[:, np.newaxis], u.shape[1], axis=1)
                        for key in self.nlp[penalty.phase].controls
                    ]
                )
                u /= u_scaling

            out = []
            if penalty.transition or penalty.multinode_penalty:
                out.append(
                    penalty.weighted_function_non_threaded[t](
                        x.reshape((-1, 1)), u.reshape((-1, 1)), p, penalty.weight, _target, dt
                    )
                )

            elif penalty.derivative or penalty.explicit_derivative:
                if not np.all(
                    x == 0
                ):  # This is a hack to initialize the plots because it x is (N,2) and we need (N, M) in collocation
                    state_value = x[:, :] if penalty.name == "CONTINUITY" else x[:, [0, -1]]
                else:
                    state_value = np.zeros(
                        (x.shape[0], int(penalty.weighted_function_non_threaded[t].nnz_in(0) / x.shape[0]))
                    )

                out.append(penalty.weighted_function_non_threaded[t](state_value, u, p, penalty.weight, _target, dt))
            elif (
                penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL
                or penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
            ):
                out = [
                    penalty.weighted_function_non_threaded[t](x[:, [i, i + 1]], u[:, i], p, penalty.weight, _target, dt)
                    for i in range(x.shape[1] - 1)
                ]
            else:
                out.append(penalty.weighted_function_non_threaded[t](x, u, p, penalty.weight, _target, dt))
            return sum1(horzcat(*out))

        def add_penalty(_penalties):
            for penalty in _penalties:
                if not penalty:
                    continue

                dt = penalty.dt
                if "time" in nlp.parameters:
                    if isinstance(penalty.type, ObjectiveFcn.Mayer):
                        dt = 1
                    elif isinstance(penalty.type, ObjectiveFcn.Lagrange):
                        if not isinstance(penalty.dt, (float, int)):
                            if i_phase in self.time_param_phases_idx:
                                dt = Function(
                                    "time",
                                    [nlp.parameters.cx_start[i_phase]],
                                    [nlp.parameters.cx_start[i_phase] / nlp.ns],
                                )

                plot_params = {
                    "fig_name": cost_type.name,
                    "update_function": compute_penalty_values,
                    "phase": i_phase,
                    "penalty": penalty,
                    "dt": dt,
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
        )

    def check_conditioning(self):
        """
        Visualisation of jacobian and hessian contraints and hessian objective for each phase at initial time
        """
        check_conditioning(self)

    def solve(
        self,
        solver: Solver | Solver.Generic = None,
        warm_start: Solution = None,
    ) -> Solution:
        """
        Call the solver to actually solve the ocp

        Parameters
        ----------
        solver: Generic
            The solver which will be used to solve the ocp
        warm_start: Solution
            The solution to pass to the warm start method

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

        if self.is_warm_starting:
            if solver.type == SolverType.IPOPT:
                solver.set_warm_start_options(1e-10)

        self.ocp_solver.opts = solver

        self.ocp_solver.solve()
        self.is_warm_starting = False

        return Solution(self, self.ocp_solver.get_optimized_value())

    def set_warm_start(self, sol: Solution):
        """
        Modify x and u initial guess based on a solution.

        Parameters
        ----------
        sol: Solution
            The solution to initiate the OCP from
        """

        state, ctrl, param = sol.states, sol.controls, sol.parameters
        u_init_guess = InitialGuessList()
        x_init_guess = InitialGuessList()
        param_init_guess = InitialGuessList()
        for i in range(self.n_phases):
            if self.n_phases == 1:
                for key in state:
                    x_init_guess.add(key, state[key], interpolation=InterpolationType.EACH_FRAME, phase=0)
                for key in ctrl:
                    if self.nlp[i].control_type == ControlType.LINEAR_CONTINUOUS:
                        u_init_guess.add(key, ctrl[key], interpolation=InterpolationType.EACH_FRAME, phase=0)
                    else:
                        u_init_guess.add(key, ctrl[key][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=0)
            else:
                for key in state[i]:
                    x_init_guess.add(key, state[i][key], interpolation=InterpolationType.EACH_FRAME, phase=i)
                for key in ctrl[i]:
                    if self.nlp[i].control_type == ControlType.LINEAR_CONTINUOUS:
                        u_init_guess.add(key, ctrl[i][key], interpolation=InterpolationType.EACH_FRAME, phase=i)
                    else:
                        u_init_guess.add(key, ctrl[i][key][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=i)

        for key in param:
            if key != "all":
                param_init_guess.add(param[key], name=key)
        self.update_initial_guess(x_init=x_init_guess, u_init=u_init_guess, param_init=param_init_guess)

        if self.ocp_solver:
            self.ocp_solver.set_lagrange_multiplier(sol)

        self.is_warm_starting = True

    def save(self, sol: Solution, file_path: str, stand_alone: bool = False):
        """
        Save the ocp and solution structure to the hard drive. It automatically creates the required
        folder if it does not exist. Please note that biorbd is required to load back this structure.

        IMPORTANT NOTICE: Please note that this is dependent on the bioptim version used to create the .bo file
        and retrocompatibility is NOT enforced. This means that an optimized solution from a previous version will
        probably NOT load on a newer bioptim version. To save the solution in a way which is independent of the
        version of bioptim, one may use the stand_alone flag to True.

        Parameters
        ----------
        sol: Solution
            The solution structure to save
        file_path: str
            The path to solve the structure. It creates a .bo (BiOptim file)
        stand_alone: bool
            If set to True, the variable dictionaries (states, controls and parameters) are saved instead of the full
            Solution class itself. This allows to load the saved file into a setting where bioptim is not installed
            using the pickle package, but prevents from using the class methods Solution offers after loading the file
        """

        _, ext = os.path.splitext(file_path)
        if ext == "":
            file_path = file_path + ".bo"
        elif ext != ".bo":
            raise RuntimeError(f"Incorrect extension({ext}), it should be (.bo) or (.bob) if you use save_get_data.")

        if stand_alone:
            # TODO check if this file is loaded when load is used, and raise an error
            data_to_save = sol.states, sol.controls, sol.parameters
        else:
            sol_copy = sol.copy()
            sol_copy.ocp = None  # Ocp is not pickable
            data_to_save = {"ocp_initializer": self.original_values, "sol": sol_copy, "versions": self.version}

        # Create folder if necessary
        directory, _ = os.path.split(file_path)
        if directory != "" and not os.path.isdir(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as file:
            pickle.dump(data_to_save, file)

    @staticmethod
    def load(file_path: str) -> list:
        """
        Reload a previous optimization (*.bo) saved using save

        Parameters
        ----------
        file_path: str
            The path to the *.bo file

        Returns
        -------
        The ocp and sol structure. If it was saved, the iterations are also loaded
        """

        with open(file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except BaseException as error_message:
                raise ValueError(
                    f"The file '{file_path}' cannot be loaded, maybe the version of bioptim (version {__version__})\n"
                    f"is not the same as the one that created the file (version unknown). For more information\n"
                    "please refer to the original error message below\n\n"
                    f"{type(error_message).__name__}: {error_message}"
                )
            ocp = OptimalControlProgram.from_loaded_data(data["ocp_initializer"])
            for key in data["versions"].keys():
                key_module = "biorbd_casadi" if key == "biorbd" else key
                try:
                    check_version(sys.modules[key_module], data["versions"][key], ocp.version[key], exclude_max=False)
                except ImportError:
                    raise ImportError(
                        f"Version of {key} from file ({data['versions'][key]}) is not the same as the "
                        f"installed version ({ocp.version[key]})"
                    )
            sol = data["sol"]
            sol.ocp = Solution.SimplifiedOCP(ocp)
            out = [ocp, sol]
        return out

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
        self,
        phase_time: int | float | list | tuple,
        objective_functions: ObjectiveList,
        constraints: ConstraintList,
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
            _initial_time_guess: list,
            _phase_time: list,
            _time_min: list,
            _time_max: list,
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
            _initial_time_guess: list
                The list of all initial guesses for the free time optimization
            _phase_time: list
                Replaces the values where free time is found for MX or SX
            _time_min: list
                Minimal bounds for the time parameter
            _time_max: list
                Maximal bounds for the time parameter
            _has_penalty: list[bool]
                If a penalty was previously found. This should be None on the first call to ensure proper initialization

            Returns
            -------
            The state of has_penalty
            """

            if _has_penalty is None:
                _has_penalty = [False] * ocp.n_phases

            for i, penalty_functions_phase in enumerate(penalty_functions):
                for pen_fun in penalty_functions_phase:
                    if not pen_fun:
                        continue
                    if (
                        pen_fun.type == ObjectiveFcn.Mayer.MINIMIZE_TIME
                        or pen_fun.type == ObjectiveFcn.Lagrange.MINIMIZE_TIME
                        or pen_fun.type == ConstraintFcn.TIME_CONSTRAINT
                    ):
                        if _has_penalty[i]:
                            raise RuntimeError("Time constraint/objective cannot declare more than once")
                        _has_penalty[i] = True

                        if i in ocp.parameter_mappings["time"].to_first.map_idx:
                            _initial_time_guess.append(_phase_time[i])
                            _phase_time[i] = ocp.cx.sym(f"time_phase_{i}", 1, 1)
                            if pen_fun.type.get_type() == ConstraintFunction:
                                _time_min.append(pen_fun.min_bound if pen_fun.min_bound else 0)
                                _time_max.append(pen_fun.max_bound if pen_fun.max_bound else inf)
                            else:
                                _time_min.append(pen_fun.params["min_bound"] if "min_bound" in pen_fun.params else 0)
                                _time_max.append(pen_fun.params["max_bound"] if "max_bound" in pen_fun.params else inf)
                        else:
                            _phase_time[i] = _phase_time[ocp.parameter_mappings["time"].to_second.map_idx[i]]
            return _has_penalty

        NLP.add(self, "t_initial_guess", phase_time, False)
        self.original_phase_time = phase_time
        if isinstance(phase_time, (int, float)):
            phase_time = [phase_time]
        phase_time = list(phase_time)
        initial_time_guess, time_min, time_max = [], [], []
        has_penalty = define_parameters_phase_time(
            self, objective_functions, initial_time_guess, phase_time, time_min, time_max
        )
        define_parameters_phase_time(
            self, constraints, initial_time_guess, phase_time, time_min, time_max, _has_penalty=has_penalty
        )

        # Add to the nlp
        NLP.add(self, "tf", phase_time, False)
        NLP.add(self, "t0", [0] + [nlp.tf for i, nlp in enumerate(self.nlp) if i != len(self.nlp) - 1], False)
        NLP.add(self, "dt", [self.nlp[i].tf / max(self.nlp[i].ns, 1) for i in range(self.n_phases)], False)

        # Add to the v vector
        i = 0
        time_param_phases_idx = []
        for nlp in self.nlp:
            if isinstance(nlp.tf, self.cx):
                if nlp.phase_idx in self.parameter_mappings["time"].to_first.map_idx:
                    time_bounds = Bounds("time", time_min[i], time_max[i], interpolation=InterpolationType.CONSTANT)
                    time_init = InitialGuess("time", initial_time_guess[i])
                    time_param = Parameter(
                        cx=nlp.tf, function=None, size=1, bounds=time_bounds, initial_guess=time_init, name="time"
                    )
                    self.v.add_parameter(time_param)
                    time_param_phases_idx += [i]
                    i += 1

        self.time_param_phases_idx = time_param_phases_idx

    def __modify_penalty(self, new_penalty: PenaltyOption | Parameter):
        """
        The internal function to modify a penalty. It is also stored in the original_values, meaning that if one
        overrides an objective only the latter is preserved when saved

        Parameters
        ----------
        new_penalty: PenaltyOption | Parameter
            Any valid option to add to the program
        """

        if not new_penalty:
            return
        phase_idx = new_penalty.phase

        # Copy to self.original_values so it can be save/load
        pen = new_penalty.type.get_type()
        self.original_values[pen.penalty_nature()].add(deepcopy(new_penalty))
        new_penalty.add_or_replace_to_penalty_pool(self, self.nlp[phase_idx])

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
        if phase_idx < 0 or phase_idx > self.n_phases - 1:
            return ValueError(f"phase_index out of range [0:{self.n_phases}]")
        if node_idx < 0 or node_idx > self.nlp[phase_idx].ns:
            return ValueError(f"node_index out of range [0:{self.nlp[phase_idx].ns}]")
        previous_phase_time = sum([nlp.tf for nlp in self.nlp[:phase_idx]])
        return previous_phase_time + self.nlp[phase_idx].node_time(node_idx)
