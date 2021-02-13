from typing import Union, Callable, Any
import os
import pickle
from copy import deepcopy
from math import inf

import biorbd
import casadi
from casadi import MX, vertcat, SX

from bioptim.optimization.non_linear_program import NonLinearProgram as NLP
from bioptim.misc.__version__ import __version__
from bioptim.misc.data import Data
from bioptim.dynamics.integrator import Integrator
from bioptim.misc.enums import ControlType, OdeSolver, Solver
from bioptim.misc.mapping import BidirectionalMapping, Mapping
from bioptim.optimization.parameters import Parameters, ParameterList, Parameter
from bioptim.misc.utils import check_version
from bioptim.dynamics.problem import Problem
from bioptim.dynamics.dynamics_type import DynamicsList, Dynamics
from bioptim.gui.plot import CustomPlot
from bioptim.interfaces.biorbd_interface import BiorbdInterface
from bioptim.limits.constraints import ConstraintFunction, ConstraintFcn, ConstraintList, Constraint, ContinuityFunctions
from bioptim.limits.phase_transition import PhaseTransitionFunctions, PhaseTransitionList
from bioptim.limits.objective_functions import ObjectiveFcn, ObjectiveList, Objective
from bioptim.limits.path_conditions import BoundsList, Bounds
from bioptim.limits.path_conditions import InitialGuess, InitialGuessList
from bioptim.limits.path_conditions import InterpolationType
from bioptim.limits.penalty import PenaltyOption

check_version(biorbd, "1.4.0", "1.5.0")


class OptimalControlProgram:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed interface to modify and solve the program

    Attributes
    ----------
    version: dict
        The version of all the underlying software. This is important when loading a previous ocp
    n_phases: Union[int, list, tuple]
        The number of phases of the ocp
    original_values: A copy of the ocp as it is after defining everything
    J: MX
        Objective values that are not phase dependent (mostly parameters)
    g: MX
        Constraints that are not phase dependent (mostly parameters and continuity constraints)
    V: MX
        All the casadi variables of the ocp in a single vector
    V_bounds: Bounds
        All the bounds for V
    V_init: InitialGuess
        All the initial guesses for V
    param_to_optimize: dict
        All the parameters to optimize
    nlp: NLP
        All the phases of the ocp
    CX: [MX, SX]
        The base type for the symbolic casadi variables
    original_phase_time: list[float]
        The time vector as sent by the user
    n_threads: int
        The number of thread to use if using multithreading
    solver_type: Solver
        The designated solver to solve the ocp
    solver: SolverInterface
        A reference to the ocp solver
    phase_transitions: list[PhaseTransition]
        The list of transition constraint between phases
    isdef_x_init: bool
        If the initial condition of the states are set
    isdef_u_init: bool
        If the initial condition of the controls are set
    isdef_x_bounds: bool
        If the bounds of the states are set
    isdef_u_bounds: bool
        If the bounds of the controls are set

    Methods
    -------

    """

    def __init__(
        self,
        biorbd_model: Union[str, biorbd.Model, list, tuple],
        dynamics: Union[Dynamics, DynamicsList],
        n_shooting: Union[int, list, tuple],
        phase_time: Union[int, float, list, tuple],
        x_init: Union[InitialGuess, InitialGuessList] = InitialGuessList(),
        u_init: Union[InitialGuess, InitialGuessList] = InitialGuessList(),
        x_bounds: Union[Bounds, BoundsList] = BoundsList(),
        u_bounds: Union[Bounds, BoundsList] = BoundsList(),
        objective_functions: Union[Objective, ObjectiveList] = ObjectiveList(),
        constraints: Union[Constraint, ConstraintList] = ConstraintList(),
        parameters: Union[Parameter, ParameterList] = ParameterList(),
        external_forces: Union[list, tuple] = (),
        ode_solver: OdeSolver = OdeSolver.RK4,
        n_integration_steps: int = 5,
        irk_polynomial_interpolation_degree: int = 4,
        control_type: Union[ControlType, list] = ControlType.CONSTANT,
        all_generalized_mapping: Union[BidirectionalMapping, list, tuple] = None,
        q_mapping: Union[BidirectionalMapping, list, tuple] = None,
        qdot_mapping: Union[BidirectionalMapping, list, tuple] = None,
        tau_mapping: Union[BidirectionalMapping, list, tuple] = None,
        plot_mappings: Mapping = None,
        phase_transitions: PhaseTransitionList = PhaseTransitionList(),
        n_threads: int = 1,
        use_sx: bool = False,
    ):
        """
        Parameters
        ----------
        biorbd_model: Union[str, biorbd.Model, list, tuple]
            The biorbd model. If biorbd_model is an str, a new model is loaded. Otherwise, the references are used
        dynamics: Union[Dynamics, DynamicsList]
            The dynamics of the phases
        n_shooting: Union[int, list[int]]
            The number of shooting point of the phases
        phase_time: Union[int, float, list, tuple]
            The phase time of the phases
        x_init: Union[InitialGuess, InitialGuessList]
            The initial guesses for the states
        u_init: Union[InitialGuess, InitialGuessList]
            The initial guesses for the controls
        x_bounds: Union[Bounds, BoundsList]
            The bounds for the states
        u_bounds: Union[Bounds, BoundsList]
            The bounds for the controls
        objective_functions: Union[Objective, ObjectiveList]
            All the objective function of the program
        constraints: Union[Constraint, ConstraintList]
            All the constraints of the program
        parameters: Union[Parameter, ParameterList]
            All the parameters to optimize of the program
        external_forces: Union[list, tuple]
            The external forces acting on the center of mass of the segments specified in the bioMod
        ode_solver: OdeSolver
            The solver for the ordinary differential equations
        n_integration_steps: int
            The number of steps for the RK
        irk_polynomial_interpolation_degree: int
            The degrees for the IRK
        control_type: ControlType
            The type of controls for each phase
        all_generalized_mapping: BidirectionalMapping
            The mapping to apply on q, qdot and tau at the same time
        q_mapping: BidirectionalMapping
            The mapping to apply on q
        qdot_mapping: BidirectionalMapping
            The mapping to apply on qdot
        tau_mapping: BidirectionalMapping
            The mapping to apply on tau
        plot_mappings: Mapping
            The mapping to apply on the plots
        phase_transitions: PhaseTransitionList
            The transition types between the phases
        n_threads: int
            The number of thread to use while solving (multi-threading if > 1)
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        """

        if isinstance(biorbd_model, str):
            biorbd_model = [biorbd.Model(biorbd_model)]
        elif isinstance(biorbd_model, biorbd.biorbd.Model):
            biorbd_model = [biorbd_model]
        elif isinstance(biorbd_model, (list, tuple)):
            biorbd_model = [biorbd.Model(m) if isinstance(m, str) else m for m in biorbd_model]
        else:
            raise RuntimeError("biorbd_model must either be a string or an instance of biorbd.Model()")
        self.version = {"casadi": casadi.__version__, "biorbd": biorbd.__version__, "bioptim": __version__}
        self.n_phases = len(biorbd_model)

        biorbd_model_path = [m.path().relativePath().to_string() for m in biorbd_model]
        self.original_values = {
            "biorbd_model": biorbd_model_path,
            "dynamics": dynamics,
            "n_shooting": n_shooting,
            "phase_time": phase_time,
            "x_init": x_init,
            "u_init": u_init,
            "x_bounds": x_bounds,
            "u_bounds": u_bounds,
            "objective_functions": ObjectiveList(),
            "constraints": ConstraintList(),
            "parameters": ParameterList(),
            "external_forces": external_forces,
            "ode_solver": ode_solver,
            "n_integration_steps": n_integration_steps,
            "irk_polynomial_interpolation_degree": irk_polynomial_interpolation_degree,
            "control_type": control_type,
            "all_generalized_mapping": all_generalized_mapping,
            "q_mapping": q_mapping,
            "qdot_mapping": qdot_mapping,
            "tau_mapping": tau_mapping,
            "plot_mappings": plot_mappings,
            "phase_transitions": phase_transitions,
            "n_threads": n_threads,
            "use_sx": use_sx,
        }

        # Check integrity of arguments
        if not isinstance(n_threads, int) or isinstance(n_threads, bool) or n_threads < 1:
            raise RuntimeError("n_threads should be a positive integer greater or equal than 1")

        if isinstance(dynamics, Dynamics):
            dynamics_type_tp = DynamicsList()
            dynamics_type_tp.add(dynamics)
            dynamics = dynamics_type_tp
        elif not isinstance(dynamics, DynamicsList):
            raise RuntimeError("dynamics should be a Dynamics or a DynamicsList")

        ns = n_shooting
        if not isinstance(ns, int) or ns < 2:
            if isinstance(ns, (tuple, list)):
                if sum([True for i in ns if not isinstance(i, int) and not isinstance(i, bool)]) != 0:
                    raise RuntimeError("n_shooting should be a positive integer (or a list of) greater or equal than 2")
            else:
                raise RuntimeError("n_shooting should be a positive integer (or a list of) greater or equal than 2")
        n_step = n_integration_steps
        if not isinstance(n_step, int) or isinstance(n_step, bool) or n_step < 1:
            raise RuntimeError("n_integration_steps should be a positive integer greater or equal than 1")

        if not isinstance(phase_time, (int, float)):
            if isinstance(phase_time, (tuple, list)):
                if sum([True for i in phase_time if not isinstance(i, (int, float))]) != 0:
                    raise RuntimeError("phase_time should be a number or a list of number")
            else:
                raise RuntimeError("phase_time should be a number or a list of number")

        if isinstance(x_bounds, Bounds):
            x_bounds_tp = BoundsList()
            x_bounds_tp.add(bounds=x_bounds)
            x_bounds = x_bounds_tp
        elif not isinstance(x_bounds, BoundsList):
            raise RuntimeError("x_bounds should be built from a Bounds or a BoundsList")

        if isinstance(u_bounds, Bounds):
            u_bounds_tp = BoundsList()
            u_bounds_tp.add(bounds=u_bounds)
            u_bounds = u_bounds_tp
        elif not isinstance(u_bounds, BoundsList):
            raise RuntimeError("u_bounds should be built from a Bounds or a BoundsList")

        if isinstance(x_init, InitialGuess):
            x_init_tp = InitialGuessList()
            x_init_tp.add(x_init)
            x_init = x_init_tp
        elif not isinstance(x_init, InitialGuessList):
            raise RuntimeError("x_init should be built from a InitialGuess or InitialGuessList")

        if isinstance(u_init, InitialGuess):
            u_init_tp = InitialGuessList()
            u_init_tp.add(u_init)
            u_init = u_init_tp
        elif not isinstance(u_init, InitialGuessList):
            raise RuntimeError("u_init should be built from a InitialGuess or InitialGuessList")

        if isinstance(objective_functions, Objective):
            objective_functions_tp = ObjectiveList()
            objective_functions_tp.add(objective_functions)
            objective_functions = objective_functions_tp
        elif not isinstance(objective_functions, ObjectiveList):
            raise RuntimeError("objective_functions should be built from an Objective or ObjectiveList")

        if isinstance(constraints, Constraint):
            constraints_tp = ConstraintList()
            constraints_tp.add(constraints)
            constraints = constraints_tp
        elif not isinstance(constraints, ConstraintList):
            raise RuntimeError("constraints should be built from an Constraint or ConstraintList")

        if not isinstance(parameters, ParameterList):
            raise RuntimeError("parameters should be built from an ParameterList")

        if not isinstance(phase_transitions, PhaseTransitionList):
            raise RuntimeError("phase_transitions should be built from an PhaseTransitionList")

        if not isinstance(ode_solver, OdeSolver):
            raise RuntimeError("ode_solver should be built an instance of OdeSolver")

        if not isinstance(use_sx, bool):
            raise RuntimeError("use_sx should be a bool")

        # Declare optimization variables
        self.J = []
        self.g = []
        self.V = []
        self.V_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        self.V_init = InitialGuess(interpolation=InterpolationType.CONSTANT)
        self.param_to_optimize = {}

        # nlp is the core of a phase
        self.nlp = [NLP() for _ in range(self.n_phases)]
        NLP.add(self, "model", biorbd_model, False)
        NLP.add(self, "phase_idx", [i for i in range(self.n_phases)], False)

        # Type of CasADi graph
        if use_sx:
            self.CX = SX
        else:
            self.CX = MX

        # Define some aliases
        NLP.add(self, "ns", n_shooting, False)
        for nlp in self.nlp:
            if nlp.ns < 1:
                raise RuntimeError("Number of shooting points must be at least 1")

        self.n_threads = n_threads
        NLP.add(self, "n_threads", n_threads, True)
        self.solver_type = Solver.NONE
        self.solver = None

        # External forces
        if external_forces != ():
            external_forces = BiorbdInterface.convert_array_to_external_forces(external_forces)
            NLP.add(self, "external_forces", external_forces, False)

        # Compute problem size
        if all_generalized_mapping is not None:
            if q_mapping is not None or qdot_mapping is not None or tau_mapping is not None:
                raise RuntimeError("all_generalized_mapping and a specified mapping cannot be used alongside")
            q_mapping = qdot_mapping = tau_mapping = all_generalized_mapping
        NLP.add(self, "q", q_mapping, q_mapping is None, BidirectionalMapping, name="mapping")
        NLP.add(self, "qdot", qdot_mapping, qdot_mapping is None, BidirectionalMapping, name="mapping")
        NLP.add(self, "tau", tau_mapping, tau_mapping is None, BidirectionalMapping, name="mapping")
        plot_mappings = plot_mappings if plot_mappings is not None else {}
        reshaped_plot_mappings = []
        for i in range(self.n_phases):
            reshaped_plot_mappings.append({})
            for key in plot_mappings:
                reshaped_plot_mappings[i][key] = plot_mappings[key][i]
        NLP.add(self, "plot", reshaped_plot_mappings, False, name="mapping")

        # Prepare the parameters to optimize
        self.phase_transitions = []
        if len(parameters) > 0:
            self.update_parameters(parameters)

        # Declare the time to optimize
        self.__define_time(phase_time, objective_functions, constraints)

        # Prepare path constraints and dynamics of the program
        NLP.add(self, "dynamics_type", dynamics, False)
        NLP.add(self, "ode_solver", ode_solver, True)
        NLP.add(self, "control_type", control_type, True)
        NLP.add(self, "n_integration_steps", n_integration_steps, True)
        NLP.add(self, "irk_polynomial_interpolation_degree", irk_polynomial_interpolation_degree, True)

        # Prepare the dynamics
        for i in range(self.n_phases):
            self.nlp[i].initialize(self.CX)
            Problem.initialize(self, self.nlp[i])
            if self.nlp[0].nx != self.nlp[i].nx or self.nlp[0].nu != self.nlp[i].nu:
                raise RuntimeError("Dynamics with different nx or nu is not supported yet")
            Integrator.prepare_dynamic_integrator(self, self.nlp[i])

        # Define the actual NLP problem
        for i in range(self.n_phases):
            self.__define_multiple_shooting_nodes_per_phase(self.nlp[i], i)

        # Define continuity constraints
        # Prepare phase transitions (Reminder, it is important that parameters are declared before,
        # otherwise they will erase the phase_transitions)
        self.phase_transitions = PhaseTransitionFunctions.prepare_phase_transitions(self, phase_transitions)

        # Inner- and inter-phase continuity
        ContinuityFunctions.continuity(self)

        self.isdef_x_init = False
        self.isdef_u_init = False
        self.isdef_x_bounds = False
        self.isdef_u_bounds = False

        self.update_bounds(x_bounds, u_bounds)
        self.update_initial_guess(x_init, u_init)

        # Prepare constraints
        self.update_constraints(constraints)

        # Prepare objectives
        self.update_objectives(objective_functions)

    def __define_multiple_shooting_nodes_per_phase(self, nlp: NLP, idx_phase: int):
        """
        Declare all the casadi variables with the right size to be used during a specific phase

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the current phase
        idx_phase
            The index associate with that phase
        """

        V = []
        X = []
        U = []

        if nlp.control_type != ControlType.CONSTANT and nlp.control_type != ControlType.LINEAR_CONTINUOUS:
            raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

        offset = 0
        for k in range(nlp.ns + 1):
            X_ = nlp.CX.sym("X_" + str(idx_phase) + "_" + str(k), nlp.nx)
            X.append(X_)
            offset += nlp.nx
            V = vertcat(V, X_)

            if nlp.control_type != ControlType.CONSTANT or (nlp.control_type == ControlType.CONSTANT and k != nlp.ns):
                U_ = nlp.CX.sym("U_" + str(idx_phase) + "_" + str(k), nlp.nu, 1)
                U.append(U_)
                offset += nlp.nu
                V = vertcat(V, U_)

        nlp.X = X
        nlp.U = U
        self.V = vertcat(self.V, V)

    def __define_initial_guess(self):
        """
        Declare and parse the initial guesses for all the variables (V vector)
        """

        for i in range(self.n_phases):
            self.nlp[i].x_init.check_and_adjust_dimensions(self.nlp[i].nx, self.nlp[i].ns)
            if self.nlp[i].control_type == ControlType.CONSTANT:
                self.nlp[i].u_init.check_and_adjust_dimensions(self.nlp[i].nu, self.nlp[i].ns - 1)
            elif self.nlp[i].control_type == ControlType.LINEAR_CONTINUOUS:
                self.nlp[i].u_init.check_and_adjust_dimensions(self.nlp[i].nu, self.nlp[i].ns)
            else:
                raise NotImplementedError(f"Plotting {self.nlp[i].control_type} is not implemented yet")

        self.V_init = InitialGuess(interpolation=InterpolationType.CONSTANT)

        for key in self.param_to_optimize.keys():
            self.V_init.concatenate(self.param_to_optimize[key].initial_guess)

        for idx_phase, nlp in enumerate(self.nlp):
            if nlp.control_type == ControlType.CONSTANT:
                nV = nlp.nx * (nlp.ns + 1) + nlp.nu * nlp.ns
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                nV = (nlp.nx + nlp.nu) * (nlp.ns + 1)
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            V_init = InitialGuess([0] * nV, interpolation=InterpolationType.CONSTANT)

            offset = 0
            for k in range(nlp.ns + 1):
                V_init.init[offset : offset + nlp.nx, 0] = nlp.x_init.init.evaluate_at(shooting_point=k)
                offset += nlp.nx

                if nlp.control_type != ControlType.CONSTANT or (
                    nlp.control_type == ControlType.CONSTANT and k != nlp.ns
                ):
                    V_init.init[offset : offset + nlp.nu, 0] = nlp.u_init.init.evaluate_at(shooting_point=k)
                    offset += nlp.nu

            V_init.check_and_adjust_dimensions(nV, 1)
            self.V_init.concatenate(V_init)

    def __define_bounds(self):
        """
        Declare and parse the bounds for all the variables (V vector)
        """

        for i in range(self.n_phases):
            self.nlp[i].x_bounds.check_and_adjust_dimensions(self.nlp[i].nx, self.nlp[i].ns)
            if self.nlp[i].control_type == ControlType.CONSTANT:
                self.nlp[i].u_bounds.check_and_adjust_dimensions(self.nlp[i].nu, self.nlp[i].ns - 1)
            elif self.nlp[i].control_type == ControlType.LINEAR_CONTINUOUS:
                self.nlp[i].u_bounds.check_and_adjust_dimensions(self.nlp[i].nu, self.nlp[i].ns)
            else:
                raise NotImplementedError(f"Plotting {self.nlp[i].control_type} is not implemented yet")

        self.V_bounds = Bounds(interpolation=InterpolationType.CONSTANT)

        for key in self.param_to_optimize.keys():
            self.V_bounds.concatenate(self.param_to_optimize[key].bounds)

        for idx_phase, nlp in enumerate(self.nlp):
            if nlp.control_type == ControlType.CONSTANT:
                nV = nlp.nx * (nlp.ns + 1) + nlp.nu * nlp.ns
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                nV = (nlp.nx + nlp.nu) * (nlp.ns + 1)
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
            V_bounds = Bounds([0] * nV, [0] * nV, interpolation=InterpolationType.CONSTANT)

            offset = 0
            for k in range(nlp.ns + 1):
                V_bounds.min[offset : offset + nlp.nx, 0] = nlp.x_bounds.min.evaluate_at(shooting_point=k)
                V_bounds.max[offset : offset + nlp.nx, 0] = nlp.x_bounds.max.evaluate_at(shooting_point=k)
                offset += nlp.nx

                if nlp.control_type != ControlType.CONSTANT or (
                    nlp.control_type == ControlType.CONSTANT and k != nlp.ns
                ):
                    V_bounds.min[offset : offset + nlp.nu, 0] = nlp.u_bounds.min.evaluate_at(shooting_point=k)
                    V_bounds.max[offset : offset + nlp.nu, 0] = nlp.u_bounds.max.evaluate_at(shooting_point=k)
                    offset += nlp.nu

            V_bounds.check_and_adjust_dimensions(nV, 1)
            self.V_bounds.concatenate(V_bounds)

    def __define_time(
        self,
        phase_time: Union[int, float, list, tuple],
        objective_functions: ObjectiveList,
        constraints: ConstraintList,
    ):
        """
        Declare the phase_time vector in V. If objective_functions or constraints defined a time optimization,
        a sanity check is perform and the values of initial guess and bounds for these particular phases

        Parameters
        ----------
        phase_time: Union[int, float, list, tuple]
            The time of all the phases
        objective_functions: ObjectiveList
            All the objective functions. It is used to scan if any time optimization was defined
        constraints: ConstraintList
            All the constraint functions. It is used to scan if any free time was defined
        """

        def define_parameters_phase_time(
            ocp: OptimalControlProgram,
            penalty_functions: Union[ObjectiveList, ConstraintList],
            initial_time_guess: list,
            phase_time: list,
            time_min: list,
            time_max: list,
            has_penalty: list = None,
        ) -> list:
            """
            Sanity check to ensure that only one time optimization is defined per phase. It also creates the time vector
            for initial guesses and bounds

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            penalty_functions: Union[ObjectiveList, ConstraintList]
                The list to parse to ensure no double free times are declared
            initial_time_guess: list
                The list of all initial guesses for the free time optimization
            phase_time: list
                Replaces the values where free time is found for MX or SX
            time_min: list
                Minimal bounds for the time parameter
            time_max: list
                Maximal bounds for the time parameter
            has_penalty: list[bool]
                If a penalty was previously found. This should be None on the first call to ensure proper initialization

            Returns
            -------
            The state of has_penalty
            """

            if has_penalty is None:
                has_penalty = [False] * ocp.n_phases

            for i, penalty_functions_phase in enumerate(penalty_functions):
                for pen_fun in penalty_functions_phase:
                    if not pen_fun:
                        continue
                    if (
                        pen_fun.type == ObjectiveFcn.Mayer.MINIMIZE_TIME
                        or pen_fun.type == ObjectiveFcn.Lagrange.MINIMIZE_TIME
                        or pen_fun.type == ConstraintFcn.TIME_CONSTRAINT
                    ):
                        if has_penalty[i]:
                            raise RuntimeError("Time constraint/objective cannot declare more than once")
                        has_penalty[i] = True

                        initial_time_guess.append(phase_time[i])
                        phase_time[i] = ocp.CX.sym(f"time_phase_{i}", 1, 1)
                        if pen_fun.type.get_type() == ConstraintFunction:
                            time_min.append(pen_fun.min_bound if pen_fun.min_bound else 0)
                            time_max.append(pen_fun.max_bound if pen_fun.max_bound else inf)
                        else:
                            time_min.append(pen_fun.params["min_bound"] if "min_bound" in pen_fun.params else 0)
                            time_max.append(pen_fun.params["max_bound"] if "max_bound" in pen_fun.params else inf)
            return has_penalty

        self.original_phase_time = phase_time
        if isinstance(phase_time, (int, float)):
            phase_time = [phase_time]
        phase_time = list(phase_time)
        initial_time_guess, time_min, time_max = [], [], []
        has_penalty = define_parameters_phase_time(
            self, objective_functions, initial_time_guess, phase_time, time_min, time_max
        )
        define_parameters_phase_time(
            self, constraints, initial_time_guess, phase_time, time_min, time_max, has_penalty=has_penalty
        )

        # Add to the nlp
        NLP.add(self, "tf", phase_time, False)
        NLP.add(self, "t0", [0] + [nlp.tf for i, nlp in enumerate(self.nlp) if i != len(self.nlp) - 1], False)
        NLP.add(self, "dt", [self.nlp[i].tf / max(self.nlp[i].ns, 1) for i in range(self.n_phases)], False)

        # Add to the V vector
        i = 0
        for nlp in self.nlp:
            if isinstance(nlp.tf, self.CX):
                time_bounds = Bounds(time_min[i], time_max[i], interpolation=InterpolationType.CONSTANT)
                time_init = InitialGuess(initial_time_guess[i])
                Parameters._add_to_v(self, "time", 1, None, time_bounds, time_init, nlp.tf)
                i += 1

    def update_objectives(self, new_objective_function: Union[Objective, ObjectiveList]):
        """
        The main user interface to add or modify objective functions in the ocp

        Parameters
        ----------
        new_objective_function: Union[Objective, ObjectiveList]
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

    def update_constraints(self, new_constraint: Union[Constraint, ConstraintList]):
        """
        The main user interface to add or modify constraint in the ocp

        Parameters
        ----------
        new_constraint: Union[Constraint, ConstraintList]
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

    def update_parameters(self, new_parameters: Union[Parameter, ParameterList]):
        """
        The main user interface to add or modify parameters in the ocp

        Parameters
        ----------
        new_parameters: Union[Parameter, ParameterList]
            The parameters to add to the ocp
        """

        if isinstance(new_parameters, Parameter):
            self.__modify_penalty(new_parameters)

        elif isinstance(new_parameters, ParameterList):
            for parameters_in_phase in new_parameters:
                for parameter in parameters_in_phase:
                    self.__modify_penalty(parameter)

        else:
            raise RuntimeError("new_parameter must be a Parameter or a ParameterList")

    def update_bounds(
        self, x_bounds: Union[Bounds, BoundsList] = BoundsList(), u_bounds: Union[Bounds, BoundsList] = BoundsList()
    ):
        """
        The main user interface to add bounds in the ocp

        Parameters
        ----------
        x_bounds: Union[Bounds, BoundsList]
            The state bounds to add
        u_bounds: Union[Bounds, BoundsList]
            The control bounds to add
        """

        if x_bounds:
            NLP.add_path_condition(self, x_bounds, "x_bounds", Bounds, BoundsList)
        if u_bounds:
            NLP.add_path_condition(self, u_bounds, "u_bounds", Bounds, BoundsList)
        if self.isdef_x_bounds and self.isdef_u_bounds:
            self.__define_bounds()

    def update_initial_guess(
        self,
        x_init: Union[InitialGuess, InitialGuessList] = InitialGuessList(),
        u_init: Union[InitialGuess, InitialGuessList] = InitialGuessList(),
        param_init: Union[InitialGuess, InitialGuessList] = InitialGuessList(),
    ):
        """
        The main user interface to add initial guesses in the ocp

        Parameters
        ----------
        x_init: Union[Bounds, BoundsList]
            The state initial guess to add
        u_init: Union[Bounds, BoundsList]
            The control initial guess to add
        param_init: Union[Bounds, BoundsList]
            The parameters initial guess to add
        """

        if x_init:
            NLP.add_path_condition(self, x_init, "x_init", InitialGuess, InitialGuessList)
        if u_init:
            NLP.add_path_condition(self, u_init, "u_init", InitialGuess, InitialGuessList)

        if isinstance(param_init, InitialGuess):
            param_init_list = InitialGuessList()
            param_init_list.add(param_init)
        else:
            param_init_list = param_init

        for param in param_init_list:
            if not param.name:
                raise ValueError("update_initial_guess must specify a name for the parameters")
            if param.name not in self.param_to_optimize:
                raise ValueError("update_initial_guess cannot declare new parameters")
            self.param_to_optimize[param.name].initial_guess.init = param.init

        if self.isdef_x_init and self.isdef_u_init:
            self.__define_initial_guess()

    def __modify_penalty(self, new_penalty: Union[PenaltyOption, Parameter]):
        """
        The internal function to modify a penalty. It is also stored in the original_values, meaning that if one
        overrides an objective only the latter is preserved when saved

        Parameters
        ----------
        new_penalty: PenaltyOption
            Any valid option to add to the program
        """

        if not new_penalty:
            return
        phase_idx = new_penalty.phase

        # Copy to self.original_values so it can be save/load
        pen = new_penalty.type.get_type()
        self.original_values[pen.penalty_nature()].add(deepcopy(new_penalty))
        pen.add_or_replace(self, self.nlp[phase_idx], new_penalty)

    def add_plot(self, fig_name: str, update_function: Callable, phase: int = -1, **parameters: Any):
        """
        The main user interface to add a new plot to the ocp

        Parameters
        ----------
        fig_name: str
            The name of the figure, it the name already exists, it is merged
        update_function: function
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

    def solve(
        self,
        solver: Solver = Solver.IPOPT,
        show_online_optim: bool = False,
        return_iterations: bool = False,
        return_objectives: bool = False,
        solver_options: dict = {},
    ) -> Any:
        """
        Call the solver to actually solve the ocp

        Parameters
        ----------
        solver: Solver
            The solver which will be used to solve the ocp
        show_online_optim: bool
            If the plot should be shown while optimizing. It will slow down the optimization a bit and is only
            available with Solver.IPOPT
        return_iterations: bool
            If each individual iterations should be returned. It will drastically slow down the optimization and is only
            available with Solver.IPOPT
        return_objectives: bool
            If the cost values for each objectives should be returned
        solver_options: dict
            Any options to change the behavior of the solver. To know which options are available, you can refer to the
            manual of the corresponding solver

        Returns
        -------
        The optimized solution structure, if return_xxx is defined, it also returns it in the same order as they are
        described in the Parameters sections
        """

        if return_iterations and not show_online_optim:
            raise RuntimeError("return_iterations without show_online_optim is not implemented yet.")

        if solver == Solver.IPOPT and self.solver_type != Solver.IPOPT:
            from ..interfaces.ipopt_interface import IpoptInterface

            self.solver = IpoptInterface(self)

        elif solver == Solver.ACADOS and self.solver_type != Solver.ACADOS:
            from ..interfaces.acados_interface import AcadosInterface

            self.solver = AcadosInterface(self, **solver_options)

        elif self.solver_type == Solver.NONE:
            raise RuntimeError("Solver not specified")
        self.solver_type = solver

        if show_online_optim:
            self.solver.online_optim(self)
            if return_iterations:
                self.solver.start_get_iterations()

        self.solver.configure(solver_options)
        self.solver.solve()

        if return_iterations:
            self.solver.finish_get_iterations()

        if return_objectives:
            self.solver.get_objectives()

        return self.solver.get_optimized_value()

    def save(self, sol: dict, file_path: str, sol_iterations: list = None):
        """
        Save the ocp and solution structure to the hard drive. It automatically create the required
        folder if it does not existsThis is the primary format if you need to reload the data later using bioptim. One
        can use the load function to reload the data structure

        Parameters
        ----------
        sol: dict
            The solution structure to save
        file_path: str
            The path to solve the structure. It creates a .bo (BiOptim file)
        sol_iterations: list
            The solution structure of each iteration to save
        """

        _, ext = os.path.splitext(file_path)
        if ext == "":
            file_path = file_path + ".bo"
        elif ext != ".bo":
            raise RuntimeError(f"Incorrect extension({ext}), it should be (.bo) or (.bob) if you use save_get_data.")
        data_to_save = {"ocp_initializer": self.original_values, "sol": sol, "versions": self.version}
        if sol_iterations is not None:
            data_to_save["sol_iterations"] = sol_iterations

        OptimalControlProgram.__save_with_pickle(data_to_save, file_path)

    def save_get_data(self, sol, file_path, sol_iterations=None, **parameters):
        """
        Save the solution into its get_data (numerical) form to the hard drive. It automatically create the required
        folder if it does not existsThis is the primary format if you need to load the data into another software

        Parameters
        ----------
        sol: dict
            The solution structure to save
        file_path: str
            The path to solve the structure. It creates a .bob (BiOptim file)
        sol_iterations: list
            The solution structure of each iteration to save
        """

        _, ext = os.path.splitext(file_path)
        if ext == "":
            file_path = file_path + ".bob"
        elif ext != ".bob":
            raise RuntimeError(f"Incorrect extension({ext}), it should be (.bob) or (.bo) if you use save.")

        data_to_save = {"data": Data.get_data(self, sol["x"], **parameters)}
        if sol_iterations is not None:
            get_data_sol_iterations = []
            for sol_iter in sol_iterations:
                get_data_sol_iterations.append(Data.get_data(self, sol_iter, **parameters))
            data_to_save["sol_iterations"] = get_data_sol_iterations

        OptimalControlProgram.__save_with_pickle(data_to_save, file_path)

    @staticmethod
    def __save_with_pickle(data: dict, file_path: str):
        """
        Internal interface for pickle. It automatically create the required folder if it does not exists

        Parameters
        ----------
        data: dict
            The data to save
        file_path: str
            The path to solve the data
        """

        directory, _ = os.path.split(file_path)
        if directory != "" and not os.path.isdir(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as file:
            pickle.dump(data, file)

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
            data = pickle.load(file)
            ocp = OptimalControlProgram(**data["ocp_initializer"])
            for key in data["versions"].keys():
                if data["versions"][key] != ocp.version[key]:
                    raise RuntimeError(
                        f"Version of {key} from file ({data['versions'][key]}) is not the same as the "
                        f"installed version ({ocp.version[key]})"
                    )
            out = [ocp, data["sol"]]
            if "sol_iterations" in data.keys():
                out.append(data["sol_iterations"])
        return out

    @staticmethod
    def print_bo(file_path: str):
        """
        Easy access to print a *.bo file to the console. It shows all the relevant ocp information

        Parameters
        ----------
        file_path: str
            The *.bo file to print to the console
        """

        with open(file_path, "rb") as file:
            data = pickle.load(file)
            original_values = data["ocp_initializer"]
            print("****************************** Information ******************************")
            for key in original_values.keys():
                if key not in ["x_init", "u_init", "x_bounds", "u_bounds"]:
                    print(f"{key} : ")
                    OptimalControlProgram._deep_print(original_values[key])
                    print("")

    @staticmethod
    def _deep_print(elem: Any, label: str = ""):
        """
        Dig into a list or a dictionary to print all numerical or str elements in it. It is call recursively to dig into
        structure

        Parameters
        ----------
        elem: Any
            The list or dict, or actual element to print
        label: str
            The key of the dict if elem is a dict. Otherwise it should be empty
        """

        if isinstance(elem, (list, tuple)):
            for k in range(len(elem)):
                OptimalControlProgram._deep_print(elem[k])
                if k != len(elem) - 1:
                    print("")
        elif isinstance(elem, dict):
            for key in elem.keys():
                OptimalControlProgram._deep_print(elem[key], label=key)
        else:
            if label == "":
                print(f"   {elem}")
            else:
                print(f"   [{label}] = {elem}")
