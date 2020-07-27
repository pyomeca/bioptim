import os
import pickle
from copy import deepcopy
from math import inf

import biorbd
import casadi
from casadi import MX, vertcat, SX

from .__version__ import __version__
from .data import Data
from .enums import ControlType, OdeSolver, Solver
from .mapping import BidirectionalMapping
from .options_lists import OptionList
from .parameters import Parameters, ParameterList, ParameterOption
from .utils import check_version
from ..dynamics.problem import Problem
from ..dynamics.dynamics_type import DynamicsTypeList, DynamicsTypeOption
from ..gui.plot import CustomPlot
from ..interfaces.biorbd_interface import BiorbdInterface
from ..interfaces.integrator import RK4
from ..limits.constraints import ConstraintFunction, Constraint, ConstraintList, ConstraintOption
from ..limits.continuity import ContinuityFunctions, StateTransitionFunctions, StateTransitionList
from ..limits.objective_functions import Objective, ObjectiveFunction, ObjectiveList, ObjectiveOption
from ..limits.path_conditions import Bounds, BoundsList, BoundsOption
from ..limits.path_conditions import InitialConditions, InitialConditionsList, InitialConditionsOption
from ..limits.path_conditions import InterpolationType

check_version(biorbd, "1.3.5", "1.4.0")


class OptimalControlProgram:
    """
    Constructor calls __prepare_dynamics and __define_multiple_shooting_nodes methods.

    To solve problem you have to call : OptimalControlProgram().solve()
    """

    def __init__(
        self,
        biorbd_model,
        dynamics_type,
        number_shooting_points,
        phase_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions=ObjectiveList(),
        constraints=ConstraintList(),
        parameters=ParameterList(),
        external_forces=(),
        ode_solver=OdeSolver.RK,
        nb_integration_steps=5,
        control_type=ControlType.CONSTANT,
        all_generalized_mapping=None,
        q_mapping=None,
        q_dot_mapping=None,
        tau_mapping=None,
        plot_mappings=None,
        state_transitions=StateTransitionList(),
        nb_threads=1,
        use_SX=False,
    ):
        """
        Prepare CasADi to solve a problem, defines some parameters, dynamic problem and ode solver.
        Defines also all constraints including continuity constraints.
        Defines the sum of all objective functions weight.

        :param biorbd_model: Biorbd model loaded from the biorbd.Model() function.
        :param dynamics_type: A selected method handler of the class dynamics.Dynamics.
        :param number_shooting_points: Subdivision number. (integer)
        :param phase_time: Simulation time in seconds. (float)
        :param X_init: States initial guess. (MX.sym from CasADi)
        :param U_init: Controls initial guess. (MX.sym from CasADi)
        :param X_bounds: States upper and lower bounds. (Instance of the class Bounds)
        :param U_bounds: Controls upper and lower bounds. (Instance of the class Bounds)
        :param objective_functions: Tuple of tuple of objectives functions handler's and weights.
        :param constraints: Tuple of constraints, instant (which node(s)) and tuple of geometric structures used.
        :param external_forces: Tuple of external forces.
        :param ode_solver: Name of chosen ode solver to use. (OdeSolver.COLLOCATION, OdeSolver.RK, OdeSolver.CVODES or
        OdeSolver.NO_SOLVER)
        :param all_generalized_mapping: States and controls mapping. (Instance of class Mapping)
        :param q_mapping: Generalized coordinates position states mapping. (Instance of class Mapping)
        :param q_dot_mapping: Generalized coordinates velocity states mapping. (Instance of class Mapping)
        :param tau_mapping: Torque controls mapping. (Instance of class Mapping)
        :param plot_mappings: Plot mapping. (Instance of class Mapping)
        :param state_transitions: State transitions (as a constraint, or an objective if there is a weight higher
        than zero)
        :param nb_threads: Number of threads used for the resolution of the problem. Default: not parallelized (integer)
        """

        if isinstance(biorbd_model, str):
            biorbd_model = [biorbd.Model(biorbd_model)]
        elif isinstance(biorbd_model, biorbd.biorbd.Model):
            biorbd_model = [biorbd_model]
        elif isinstance(biorbd_model, (list, tuple)):
            biorbd_model = [biorbd.Model(m) if isinstance(m, str) else m for m in biorbd_model]
        else:
            raise RuntimeError("biorbd_model must either be a string or an instance of biorbd.Model()")
        self.version = {"casadi": casadi.__version__, "biorbd": biorbd.__version__, "biorbd_optim": __version__}
        self.nb_phases = len(biorbd_model)

        biorbd_model_path = [m.path().relativePath().to_string() for m in biorbd_model]
        self.original_values = {
            "biorbd_model": biorbd_model_path,
            "dynamics_type": dynamics_type,
            "number_shooting_points": number_shooting_points,
            "phase_time": phase_time,
            "X_init": X_init,
            "U_init": U_init,
            "X_bounds": X_bounds,
            "U_bounds": U_bounds,
            "objective_functions": ObjectiveList(),
            "constraints": ConstraintList(),
            "parameters": ParameterList(),
            "external_forces": external_forces,
            "ode_solver": ode_solver,
            "nb_integration_steps": nb_integration_steps,
            "control_type": control_type,
            "all_generalized_mapping": all_generalized_mapping,
            "q_mapping": q_mapping,
            "q_dot_mapping": q_dot_mapping,
            "tau_mapping": tau_mapping,
            "plot_mappings": plot_mappings,
            "state_transitions": state_transitions,
            "nb_threads": nb_threads,
            "use_SX": use_SX,
        }

        # Check integrity of arguments
        if not isinstance(nb_threads, int) or isinstance(nb_threads, bool) or nb_threads < 1:
            raise RuntimeError("nb_threads should be a positive integer greater or equal than 1")

        if isinstance(dynamics_type, DynamicsTypeOption):
            dynamics_type_tp = DynamicsTypeList()
            dynamics_type_tp.add(dynamics_type)
            dynamics_type = dynamics_type_tp
        elif not isinstance(dynamics_type, DynamicsTypeList):
            raise RuntimeError("dynamics_type should be a DynamicsTypeOption or a DynamicsTypeList")

        ns = number_shooting_points
        if not isinstance(ns, int) or ns < 2:
            if isinstance(ns, (tuple, list)):
                if sum([True for i in ns if not isinstance(i, int) and not isinstance(i, bool)]) != 0:
                    raise RuntimeError(
                        "number_shooting_points should be a positive integer (or a list of) greater or equal than 2"
                    )
            else:
                raise RuntimeError(
                    "number_shooting_points should be a positive integer (or a list of) greater or equal than 2"
                )
        nstep = nb_integration_steps
        if not isinstance(nstep, int) or isinstance(nstep, bool) or nstep < 1:
            raise RuntimeError("nb_integration_steps should be a positive integer greater or equal than 1")

        if not isinstance(phase_time, (int, float)):
            if isinstance(phase_time, (tuple, list)):
                if sum([True for i in phase_time if not isinstance(i, (int, float))]) != 0:
                    raise RuntimeError("phase_time should be a number or a list of number")
            else:
                raise RuntimeError("phase_time should be a number or a list of number")

        if isinstance(X_init, InitialConditionsOption):
            X_init_tp = InitialConditionsList()
            X_init_tp.add(X_init)
            X_init = X_init_tp
        elif not isinstance(X_init, InitialConditionsList):
            raise RuntimeError("X_init should be built from a InitialConditionsOption or InitialConditionsList")

        if isinstance(U_init, InitialConditionsOption):
            U_init_tp = InitialConditionsList()
            U_init_tp.add(U_init)
            U_init = U_init_tp
        elif not isinstance(U_init, InitialConditionsList):
            raise RuntimeError("U_init should be built from a InitialConditionsOption or InitialConditionsList")

        if isinstance(X_bounds, BoundsOption):
            X_bounds_tp = BoundsList()
            X_bounds_tp.add(X_bounds)
            X_bounds = X_bounds_tp
        elif not isinstance(X_bounds, BoundsList):
            raise RuntimeError("X_bounds should be built from a BoundOption or a BoundsList")

        if isinstance(U_bounds, BoundsOption):
            U_bounds_tp = BoundsList()
            U_bounds_tp.add(U_bounds)
            U_bounds = U_bounds_tp
        elif not isinstance(U_bounds, BoundsList):
            raise RuntimeError("U_bounds should be built from a BoundOption or a BoundsList")

        if isinstance(objective_functions, ObjectiveOption):
            objective_functions_tp = ObjectiveList()
            objective_functions_tp.add(objective_functions)
            objective_functions = objective_functions_tp
        elif not isinstance(objective_functions, ObjectiveList):
            raise RuntimeError("objective_functions should be built from an ObjectiveOption or ObjectiveList")

        if isinstance(constraints, ConstraintOption):
            constraints_tp = ConstraintList()
            constraints_tp.add(constraints)
            constraints = constraints_tp
        elif not isinstance(constraints, ConstraintList):
            raise RuntimeError("constraints should be built from an ConstraintOption or ConstraintList")

        if not isinstance(parameters, ParameterList):
            raise RuntimeError("parameters should be built from an ParameterList")

        if not isinstance(state_transitions, StateTransitionList):
            raise RuntimeError("state_transitions should be built from an StateTransitionList")

        if not isinstance(ode_solver, OdeSolver):
            raise RuntimeError("ode_solver should be built an instance of OdeSolver")

        if not isinstance(use_SX, bool):
            raise RuntimeError("use_SX should be a bool")

        # Declare optimization variables
        self.J = []
        self.g = []
        self.g_bounds = []
        self.V = []
        self.V_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        self.V_init = InitialConditions(interpolation=InterpolationType.CONSTANT)
        self.param_to_optimize = {}

        # nlp is the core of a phase
        self.nlp = [{} for _ in range(self.nb_phases)]
        self.__add_to_nlp("model", biorbd_model, False)
        self.__add_to_nlp("phase_idx", [i for i in range(self.nb_phases)], False)

        # Type of CasADi graph
        if use_SX:
            self.CX = SX
        else:
            self.CX = MX

        # Define some aliases
        self.__add_to_nlp("ns", number_shooting_points, False)
        for nlp in self.nlp:
            if nlp["ns"] < 1:
                raise RuntimeError("Number of shooting points must be at least 1")
        self.initial_phase_time = phase_time
        phase_time, initial_time_guess, time_min, time_max = self.__init_phase_time(
            phase_time, objective_functions, constraints
        )
        self.__add_to_nlp("tf", phase_time, False)
        self.__add_to_nlp("t0", [0] + [nlp["tf"] for i, nlp in enumerate(self.nlp) if i != len(self.nlp) - 1], False)
        self.__add_to_nlp("dt", [self.nlp[i]["tf"] / max(self.nlp[i]["ns"], 1) for i in range(self.nb_phases)], False)
        self.nb_threads = nb_threads
        self.__add_to_nlp("nb_threads", nb_threads, True)
        self.solver_type = Solver.NONE
        self.solver = None

        # External forces
        if external_forces != ():
            external_forces = BiorbdInterface.convert_array_to_external_forces(external_forces)
            self.__add_to_nlp("external_forces", external_forces, False)

        # Compute problem size
        if all_generalized_mapping is not None:
            if q_mapping is not None or q_dot_mapping is not None or tau_mapping is not None:
                raise RuntimeError("all_generalized_mapping and a specified mapping cannot be used alongside")
            q_mapping = q_dot_mapping = tau_mapping = all_generalized_mapping
        self.__add_to_nlp("q_mapping", q_mapping, q_mapping is None, BidirectionalMapping)
        self.__add_to_nlp("q_dot_mapping", q_dot_mapping, q_dot_mapping is None, BidirectionalMapping)
        self.__add_to_nlp("tau_mapping", tau_mapping, tau_mapping is None, BidirectionalMapping)
        plot_mappings = plot_mappings if plot_mappings is not None else {}
        reshaped_plot_mappings = []
        for i in range(self.nb_phases):
            reshaped_plot_mappings.append({})
            for key in plot_mappings:
                reshaped_plot_mappings[i][key] = plot_mappings[key][i]
        self.__add_to_nlp("plot_mappings", reshaped_plot_mappings, False)

        # Prepare the parameters to optimize
        self.state_transitions = []
        if len(parameters) > 0:
            self.update_parameters(parameters)

        # Declare the time to optimize
        self.__define_variable_time(initial_time_guess, time_min, time_max)

        # Prepare the dynamics of the program
        self.__add_to_nlp("dynamics_type", dynamics_type, False)
        self.__add_to_nlp("ode_solver", ode_solver, True)
        self.__add_to_nlp("control_type", control_type, True)
        for i in range(self.nb_phases):
            self.__initialize_nlp(self.nlp[i])
            Problem.initialize(self, self.nlp[i])

        # Prepare path constraints
        self.__add_to_nlp("X_bounds", X_bounds, False)
        self.__add_to_nlp("U_bounds", U_bounds, False)
        for i in range(self.nb_phases):
            self.nlp[i]["X_bounds"].check_and_adjust_dimensions(self.nlp[i]["nx"], self.nlp[i]["ns"])
            if self.nlp[i]["control_type"] == ControlType.CONSTANT:
                self.nlp[i]["U_bounds"].check_and_adjust_dimensions(self.nlp[i]["nu"], self.nlp[i]["ns"] - 1)
            elif self.nlp[i]["control_type"] == ControlType.LINEAR_CONTINUOUS:
                self.nlp[i]["U_bounds"].check_and_adjust_dimensions(self.nlp[i]["nu"], self.nlp[i]["ns"])
            else:
                raise NotImplementedError(f"Plotting {self.nlp[i]['control_type']} is not implemented yet")

        # Prepare initial guesses
        self.__add_to_nlp("X_init", X_init, False)
        self.__add_to_nlp("U_init", U_init, False)
        for i in range(self.nb_phases):
            self.nlp[i]["X_init"].check_and_adjust_dimensions(self.nlp[i]["nx"], self.nlp[i]["ns"])
            if self.nlp[i]["control_type"] == ControlType.CONSTANT:
                self.nlp[i]["U_init"].check_and_adjust_dimensions(self.nlp[i]["nu"], self.nlp[i]["ns"] - 1)
            elif self.nlp[i]["control_type"] == ControlType.LINEAR_CONTINUOUS:
                self.nlp[i]["U_init"].check_and_adjust_dimensions(self.nlp[i]["nu"], self.nlp[i]["ns"])
            else:
                raise NotImplementedError(f"Plotting {self.nlp[i]['control_type']} is not implemented yet")

        # Variables and constraint for the optimization program
        for i in range(self.nb_phases):
            self.__define_multiple_shooting_nodes_per_phase(self.nlp[i], i)

        # Define dynamic problem
        self.__add_to_nlp(
            "nb_integration_steps", nb_integration_steps, True
        )  # Number of steps of integration (for now only RK4 steps are implemented)
        for i in range(self.nb_phases):
            if self.nlp[0]["nx"] != self.nlp[i]["nx"] or self.nlp[0]["nu"] != self.nlp[i]["nu"]:
                raise RuntimeError("Dynamics with different nx or nu is not supported yet")
            self.__prepare_dynamics(self.nlp[i])

        # Prepare phase transitions (Reminder, it is important that parameters are declared
        # before, otherwise they will erase the state_transitions)
        self.state_transitions = StateTransitionFunctions.prepare_state_transitions(self, state_transitions)

        # Inner- and inter-phase continuity
        ContinuityFunctions.continuity(self)

        # Prepare constraints
        self.update_constraints(constraints)

        # Prepare objectives
        self.update_objectives(objective_functions)

    def __initialize_nlp(self, nlp):
        """Start with an empty non linear problem"""
        nlp["nbQ"] = 0
        nlp["nbQdot"] = 0
        nlp["nbTau"] = 0
        nlp["nbMuscles"] = 0
        nlp["plot"] = {}
        nlp["var_states"] = {}
        nlp["var_controls"] = {}
        nlp["CX"] = self.CX
        nlp["x"] = nlp["CX"]()
        nlp["u"] = nlp["CX"]()
        nlp["J"] = []
        nlp["g"] = []
        nlp["g_bounds"] = []
        nlp["casadi_func"] = {}

    def __add_to_nlp(self, param_name, param, duplicate_if_size_is_one, _type=None):
        """Adds coupled parameters to the non linear problem"""
        if isinstance(param, (list, tuple)):
            if len(param) != self.nb_phases:
                raise RuntimeError(
                    f"{param_name} size({len(param)}) does not correspond to the number of phases({self.nb_phases})."
                )
            else:
                for i in range(self.nb_phases):
                    self.nlp[i][param_name] = param[i]
        elif isinstance(param, OptionList):
            if len(param) == self.nb_phases:
                for i in range(self.nb_phases):
                    self.nlp[i][param_name] = param[i]
            else:
                if len(param) == 1 and duplicate_if_size_is_one:
                    for i in range(self.nb_phases):
                        self.nlp[i][param_name] = param[0]
                else:
                    raise RuntimeError(
                        f"{param_name} size({len(param)}) does not correspond "
                        f"to the number of phases({self.nb_phases})."
                    )
        else:
            if self.nb_phases == 1:
                self.nlp[0][param_name] = param
            else:
                if duplicate_if_size_is_one:
                    for i in range(self.nb_phases):
                        self.nlp[i][param_name] = param
                else:
                    raise RuntimeError(f"{param_name} must be a list or tuple when number of phase is not equal to 1")

        if _type is not None:
            for nlp in self.nlp:
                if nlp[param_name] is not None and not isinstance(nlp[param_name], _type):
                    raise RuntimeError(f"Parameter {param_name} must be a {str(_type)}")

    def __prepare_dynamics(self, nlp):
        """
        Builds CasaDI dynamics function.
        :param nlp: The nlp problem
        """

        ode_opt = {"t0": 0, "tf": nlp["dt"]}
        if nlp["ode_solver"] == OdeSolver.COLLOCATION or nlp["ode_solver"] == OdeSolver.RK:
            ode_opt["number_of_finite_elements"] = nlp["nb_integration_steps"]

        dynamics = nlp["dynamics_func"]
        ode = {"x": nlp["x"], "p": nlp["u"], "ode": dynamics(nlp["x"], nlp["u"], nlp["p"])}
        nlp["dynamics"] = []
        nlp["par_dynamics"] = {}
        if nlp["ode_solver"] == OdeSolver.RK:
            ode_opt["model"] = nlp["model"]
            ode_opt["param"] = nlp["p"]
            ode_opt["CX"] = nlp["CX"]
            ode_opt["idx"] = 0
            ode["ode"] = dynamics
            ode_opt["control_type"] = nlp["control_type"]
            if "external_forces" in nlp:
                for idx in range(len(nlp["external_forces"])):
                    ode_opt["idx"] = idx
                    nlp["dynamics"].append(RK4(ode, ode_opt))
            else:
                if self.nb_threads > 1 and nlp["control_type"] == ControlType.LINEAR_CONTINUOUS:
                    raise RuntimeError("Piece-wise linear continuous controls cannot be used with multiple threads")
                nlp["dynamics"].append(RK4(ode, ode_opt))
        elif nlp["ode_solver"] == OdeSolver.COLLOCATION:
            if not isinstance(self.CX(), MX):
                raise RuntimeError("COLLOCATION integrator can only be used with MX graphs")
            if len(self.param_to_optimize) != 0:
                raise RuntimeError("COLLOCATION cannot be used while optimizing parameters")
            if "external_forces" in nlp:
                raise RuntimeError("COLLOCATION cannot be used with external_forces")
            if nlp["control_type"] == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("COLLOCATION cannot be used with piece-wise linear controls (only RK4)")
            nlp["dynamics"].append(casadi.integrator("integrator", "collocation", ode, ode_opt))
        elif nlp["ode_solver"] == OdeSolver.CVODES:
            if not isinstance(self.CX(), MX):
                raise RuntimeError("CVODES integrator can only be used with MX graphs")
            if len(self.param_to_optimize) != 0:
                raise RuntimeError("CVODES cannot be used while optimizing parameters")
            if "external_forces" in nlp:
                raise RuntimeError("CVODES cannot be used with external_forces")
            if nlp["control_type"] == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("CVODES cannot be used with piece-wise linear controls (only RK4)")
            nlp["dynamics"].append(casadi.integrator("integrator", "cvodes", ode, ode_opt))

        if len(nlp["dynamics"]) == 1:
            if self.nb_threads > 1:
                nlp["par_dynamics"] = nlp["dynamics"][0].map(nlp["ns"], "thread", self.nb_threads)
            nlp["dynamics"] = nlp["dynamics"] * nlp["ns"]

    def __define_multiple_shooting_nodes_per_phase(self, nlp, idx_phase):
        """
        For each node, puts X_bounds and U_bounds in V_bounds.
        Links X and U with V.
        :param nlp: The non linear problem.
        :param idx_phase: Index of the phase. (integer)
        """

        V = []
        X = []
        U = []

        if nlp["control_type"] == ControlType.CONSTANT:
            nV = nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * nlp["ns"]
        elif nlp["control_type"] == ControlType.LINEAR_CONTINUOUS:
            nV = (nlp["nx"] + nlp["nu"]) * (nlp["ns"] + 1)
        else:
            raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp['control_type']}")
        V_bounds = Bounds([0] * nV, [0] * nV, interpolation=InterpolationType.CONSTANT)
        V_init = InitialConditions([0] * nV, interpolation=InterpolationType.CONSTANT)

        offset = 0
        for k in range(nlp["ns"] + 1):
            X_ = nlp["CX"].sym("X_" + str(idx_phase) + "_" + str(k), nlp["nx"])
            X.append(X_)
            V_bounds.min[offset : offset + nlp["nx"], 0] = nlp["X_bounds"].min.evaluate_at(shooting_point=k)
            V_bounds.max[offset : offset + nlp["nx"], 0] = nlp["X_bounds"].max.evaluate_at(shooting_point=k)
            V_init.init[offset : offset + nlp["nx"], 0] = nlp["X_init"].init.evaluate_at(shooting_point=k)
            offset += nlp["nx"]
            V = vertcat(V, X_)

            if nlp["control_type"] != ControlType.CONSTANT or (
                nlp["control_type"] == ControlType.CONSTANT and k != nlp["ns"]
            ):
                U_ = nlp["CX"].sym("U_" + str(idx_phase) + "_" + str(k), nlp["nu"], 1)
                U.append(U_)
                V_bounds.min[offset : offset + nlp["nu"], 0] = nlp["U_bounds"].min.evaluate_at(shooting_point=k)
                V_bounds.max[offset : offset + nlp["nu"], 0] = nlp["U_bounds"].max.evaluate_at(shooting_point=k)
                V_init.init[offset : offset + nlp["nu"], 0] = nlp["U_init"].init.evaluate_at(shooting_point=k)
                offset += nlp["nu"]
                V = vertcat(V, U_)

        V_bounds.check_and_adjust_dimensions(nV, 1)
        V_init.check_and_adjust_dimensions(nV, 1)

        nlp["X"] = X
        nlp["U"] = U
        self.V = vertcat(self.V, V)

        self.V_bounds.concatenate(V_bounds)
        self.V_init.concatenate(V_init)

    def __init_phase_time(self, phase_time, objective_functions, constraints):
        """
        Initializes phase time bounds and guess.
        Defines the objectives for each phase.
        :param phase_time: Phases duration. (list of floats)?
        :param objective_functions: Instance of class ObjectiveFunction.
        :param constraints: Instance of class ConstraintFunction.
        :return: phase_time -> Phases duration. (list) , initial_time_guess -> Initial guess on the duration of the
        phases. (list), time_min -> Minimal bounds on the duration of the phases. (list)  and time_max -> Maximal
        bounds on the duration of the phases. (list)
        """
        if isinstance(phase_time, (int, float)):
            phase_time = [phase_time]
        phase_time = list(phase_time)
        initial_time_guess, time_min, time_max = [], [], []
        has_penalty = self.__define_parameters_phase_time(
            objective_functions, initial_time_guess, phase_time, time_min, time_max
        )
        self.__define_parameters_phase_time(
            constraints, initial_time_guess, phase_time, time_min, time_max, has_penalty=has_penalty
        )
        return phase_time, initial_time_guess, time_min, time_max

    def __define_parameters_phase_time(
        self, penalty_functions, initial_time_guess, phase_time, time_min, time_max, has_penalty=None
    ):
        if has_penalty is None:
            has_penalty = [False] * self.nb_phases

        for i, penalty_functions_phase in enumerate(penalty_functions):
            for pen_fun in penalty_functions_phase:
                if not pen_fun:
                    continue
                if (
                    pen_fun.type == Objective.Mayer.MINIMIZE_TIME
                    or pen_fun.type == Objective.Lagrange.MINIMIZE_TIME
                    or pen_fun.type == Constraint.TIME_CONSTRAINT
                ):
                    if has_penalty[i]:
                        raise RuntimeError("Time constraint/objective cannot declare more than once")
                    has_penalty[i] = True

                    initial_time_guess.append(phase_time[i])
                    phase_time[i] = self.CX.sym(f"time_phase_{i}", 1, 1)
                    if pen_fun.type.get_type() == ConstraintFunction:
                        time_min.append(pen_fun.minimum if pen_fun.minimum else 0)
                        time_max.append(pen_fun.maximum if pen_fun.maximum else inf)
                    else:
                        time_min.append(pen_fun.params["minimum"] if "minimum" in pen_fun.params else 0)
                        time_max.append(pen_fun.params["maximum"] if "maximum" in pen_fun.params else inf)
        return has_penalty

    def __define_variable_time(self, initial_guess, minimum, maximum):
        """
        For each variable time, sets initial guess and bounds.
        :param initial_guess: The initial values taken from the phase_time vector
        :param minimum: variable time minimums as set by user (default: 0)
        :param maximum: variable time maximums as set by user (default: inf)
        """
        i = 0
        for nlp in self.nlp:
            if isinstance(nlp["tf"], self.CX):
                time_bounds = Bounds(minimum[i], maximum[i], interpolation=InterpolationType.CONSTANT)
                time_init = InitialConditions(initial_guess[i])
                Parameters._add_to_v(self, "time", 1, None, time_bounds, time_init, nlp["tf"])
                i += 1

    def update_objectives(self, new_objective_function):
        if isinstance(new_objective_function, ObjectiveOption):
            self._modify_penalty(new_objective_function, "objective_functions")

        elif isinstance(new_objective_function, ObjectiveList):
            for objective_in_phase in new_objective_function:
                for objective in objective_in_phase:
                    self._modify_penalty(objective, "objective_functions")

        else:
            raise RuntimeError("new_objective_function must be a ObjectiveOption or an ObjectiveList")

    def update_constraints(self, new_constraint):
        if isinstance(new_constraint, ConstraintOption):
            self._modify_penalty(new_constraint, "constraints")

        elif isinstance(new_constraint, ConstraintList):
            for constraints_in_phase in new_constraint:
                for constraint in constraints_in_phase:
                    self._modify_penalty(constraint, "constraints")

        else:
            raise RuntimeError("new_constraint must be a ConstraintOption or a ConstraintList")

    def update_parameters(self, new_parameters):
        if isinstance(new_parameters, ParameterOption):
            self._modify_penalty(new_parameters, "parameters")

        elif isinstance(new_parameters, ParameterList):
            for parameters_in_phase in new_parameters:
                for parameter in parameters_in_phase:
                    self._modify_penalty(parameter, "parameters")

        else:
            raise RuntimeError("new_parameter must be a ParameterOption or a ParameterList")

    def _modify_penalty(self, new_penalty, penalty_name):
        """
        Modification of a penalty (constraint or objective)
        :param new_penalty: Penalty to keep after the modification.
        :param penalty: Index of the penalty to be modified. (integer)
        :param phase_idx: Index of the phase in which the penalty will be modified. (integer)
        :param penalty_name: Name of the penalty to modify. (string)
        """
        if not new_penalty:
            return
        phase_idx = new_penalty.phase

        # Copy to self.original_values so it can be save/load
        self.original_values[penalty_name].add(deepcopy(new_penalty))

        if penalty_name == "objective_functions":
            ObjectiveFunction.add_or_replace(self, self.nlp[phase_idx], new_penalty)
        elif penalty_name == "constraints":
            ConstraintFunction.add_or_replace(self, self.nlp[phase_idx], new_penalty)
        elif penalty_name == "parameters":
            Parameters.add_or_replace(self, new_penalty)
        else:
            raise RuntimeError("Unrecognized penalty")

    def add_plot(self, fig_name, update_function, phase_number=-1, **parameters):
        """
        Adds plots to show the result of the optimization
        :param fig_name: Name of the figure (string)
        :param update_function: Function to be plotted. (string) ???
        :param phase_number: Phase to be plotted. (integer)
        """
        if "combine_to" in parameters:
            raise RuntimeError(
                "'combine_to' cannot be specified in add_plot, please use same 'fig_name' to combine plots"
            )

        # --- Solve the program --- #
        if len(self.nlp) == 1:
            phase_number = 0
        else:
            if phase_number < 0:
                raise RuntimeError("phase_idx must be specified for multiphase OCP")
        nlp = self.nlp[phase_number]
        custom_plot = CustomPlot(update_function, **parameters)

        if fig_name in nlp["plot"]:
            # Make sure we add a unique name in the dict
            custom_plot.combine_to = fig_name

            if fig_name:
                cmp = 0
                while True:
                    plot_name = f"{fig_name}_{cmp}"
                    if plot_name not in nlp["plot"]:
                        break
                    cmp += 1
        else:
            plot_name = fig_name

        nlp["plot"][plot_name] = custom_plot

    def solve(
        self, solver=Solver.IPOPT, show_online_optim=False, return_iterations=False, solver_options={},
    ):
        """
        Gives to CasADi states, controls, constraints, sum of all objective functions and theirs bounds.
        Gives others parameters to control how solver works.
        :param solver: Name of the solver to use during the optimization. (string)
        :param show_online_optim: if True, optimization process is graphed in realtime. (bool)
        :param options_ipopt: See Ippot documentation for options. (dictionary)
        :return: Solution of the problem. (dictionary)
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

        return self.solver.get_optimized_value(self)

    def save(self, sol, file_path, sol_iterations=None):
        """
        :param sol: Solution of the optimization returned by CasADi.
        :param file_path: Path of the file where the solution is saved. (string)
        :param sol_iterations: The solutions for each iteration
        Saves results of the optimization into a .bo file
        """
        _, ext = os.path.splitext(file_path)
        if ext == "":
            file_path = file_path + ".bo"
        elif ext != ".bo":
            raise RuntimeError(f"Incorrect extension({ext}), it should be (.bo) or (.bob) if you use save_get_data.")
        dict = {"ocp_initilializer": self.original_values, "sol": sol, "versions": self.version}
        if sol_iterations != None:
            dict["sol_iterations"] = sol_iterations

        OptimalControlProgram._save_with_pickle(dict, file_path)

    def save_get_data(self, sol, file_path, sol_iterations=None, **parameters):
        _, ext = os.path.splitext(file_path)
        if ext == "":
            file_path = file_path + ".bob"
        elif ext != ".bob":
            raise RuntimeError(f"Incorrect extension({ext}), it should be (.bob) or (.bo) if you use save.")
        dict = {"data": Data.get_data(self, sol["x"], **parameters)}
        if sol_iterations != None:
            get_data_sol_iterations = []
            for iter in sol_iterations:
                get_data_sol_iterations.append(Data.get_data(self, iter, **parameters))
            dict["sol_iterations"] = get_data_sol_iterations

        OptimalControlProgram._save_with_pickle(dict, file_path)

    @staticmethod
    def _save_with_pickle(dict, file_path):
        dir, _ = os.path.split(file_path)
        if dir != "" and not os.path.isdir(dir):
            os.makedirs(dir)

        with open(file_path, "wb") as file:
            pickle.dump(dict, file)

    @staticmethod
    def load(file_path):
        """
        Loads results of a previous optimization from a .bo file
        :param file_path: Path of the file where the solution is saved. (string)
        :return: ocp -> Optimal control program. (instance of OptimalControlProgram class) and
        sol -> Solution of the optimization. (dictionary)
        """
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            ocp = OptimalControlProgram(**data["ocp_initilializer"])
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
    def read_information(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            original_values = data["ocp_initilializer"]
            print("****************************** Informations ******************************")
            for key in original_values.keys():
                if key not in ["X_init", "U_init", "X_bounds", "U_bounds"]:
                    print(f"{key} : ")
                    OptimalControlProgram._deep_print(original_values[key])
                    print("")

    @staticmethod
    def _deep_print(elem, label=""):
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
