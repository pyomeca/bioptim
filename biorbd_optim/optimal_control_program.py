from math import inf
from copy import deepcopy
import pickle
import os

import biorbd
import casadi
from casadi import MX, vertcat

from .enums import OdeSolver
from .mapping import Mapping, BidirectionalMapping
from .path_conditions import Bounds, InitialConditions, InterpolationType
from .constraints import Constraint, ConstraintFunction
from .objective_functions import Objective, ObjectiveFunction
from .plot import OnlineCallback
from .integrator import RK4
from .biorbd_interface import BiorbdInterface
from .variable_optimization import Data
from .__version__ import __version__


class OptimalControlProgram:
    """
    Constructor calls __prepare_dynamics and __define_multiple_shooting_nodes methods.

    To solve problem you have to call : OptimalControlProgram().solve()
    """

    def __init__(
        self,
        biorbd_model,
        problem_type,
        number_shooting_points,
        phase_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions=(),
        constraints=(),
        external_forces=(),
        ode_solver=OdeSolver.RK,
        all_generalized_mapping=None,
        q_mapping=None,
        q_dot_mapping=None,
        tau_mapping=None,
        plot_mappings=None,
        is_cyclic_objective=False,
        is_cyclic_constraint=False,
        show_online_optim=False,
    ):
        """
        Prepare CasADi to solve a problem, defines some parameters, dynamic problem and ode solver.
        Defines also all constraints including continuity constraints.
        Defines the sum of all objective functions weight.

        :param biorbd_model: Biorbd model loaded from the biorbd.Model() function
        :param problem_type: A selected method handler of the class problem_type.ProblemType.
        :param ode_solver: Name of chosen ode, available in OdeSolver enum class.
        :param number_shooting_points: Subdivision number.
        :param phase_time: Simulation time in seconds.
        :param objective_functions: Tuple of tuple of objectives functions handler's and weights.
        :param X_bounds: Instance of the class Bounds.
        :param U_bounds: Instance of the class Bounds.
        :param constraints: Tuple of constraints, instant (which node(s)) and tuple of geometric structures used.
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

        biorbd_model_path = [m.path().relativePath().to_string() for m in biorbd_model]
        self.original_values = {
            "biorbd_model": biorbd_model_path,
            "problem_type": problem_type,
            "number_shooting_points": number_shooting_points,
            "phase_time": phase_time,
            "X_init": X_init,
            "U_init": U_init,
            "X_bounds": X_bounds,
            "U_bounds": U_bounds,
            "objective_functions": deepcopy(objective_functions),
            "constraints": deepcopy(constraints),
            "external_forces": external_forces,
            "ode_solver": ode_solver,
            "all_generalized_mapping": all_generalized_mapping,
            "q_mapping": q_mapping,
            "q_dot_mapping": q_dot_mapping,
            "tau_mapping": tau_mapping,
            "plot_mappings": plot_mappings,
            "is_cyclic_objective": is_cyclic_objective,
            "is_cyclic_constraint": is_cyclic_constraint,
            "show_online_optim": show_online_optim,
        }

        self.nb_phases = len(biorbd_model)
        self.nlp = [{} for _ in range(self.nb_phases)]
        self.__add_to_nlp("model", biorbd_model, False)
        self.__add_to_nlp("phase_idx", [i for i in range(self.nb_phases)], False)

        # Prepare some variables
        self.__init_penality(constraints, "constraints")
        self.__init_penality(objective_functions, "objective_functions")

        # Define some aliases
        self.__add_to_nlp("ns", number_shooting_points, False)
        for nlp in self.nlp:
            if nlp["ns"] < 1:
                raise RuntimeError("Number of shooting points must be at least 1")
        self.initial_phase_time = phase_time
        phase_time, initial_time_guess, time_min, time_max = self.__init_phase_time(phase_time)
        self.__add_to_nlp("tf", phase_time, False)
        self.__add_to_nlp("t0", [0] + [nlp["tf"] for i, nlp in enumerate(self.nlp) if i != len(self.nlp) - 1], False)
        self.__add_to_nlp(
            "dt", [self.nlp[i]["tf"] / max(self.nlp[i]["ns"], 1) for i in range(self.nb_phases)], False,
        )
        self.is_cyclic_constraint = is_cyclic_constraint
        self.is_cyclic_objective = is_cyclic_objective

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
        self.__add_to_nlp("problem_type", problem_type, False)
        for i in range(self.nb_phases):
            self.nlp[i]["problem_type"](self.nlp[i])

        # Prepare path constraints
        self.__add_to_nlp("X_bounds", X_bounds, False)
        self.__add_to_nlp("U_bounds", U_bounds, False)
        for i in range(self.nb_phases):
            self.nlp[i]["X_bounds"].check_and_adjust_dimensions(self.nlp[i]["nx"], self.nlp[i]["ns"])
            self.nlp[i]["U_bounds"].check_and_adjust_dimensions(self.nlp[i]["nu"], self.nlp[i]["ns"] - 1)

        # Prepare initial guesses
        self.__add_to_nlp("X_init", X_init, False)
        self.__add_to_nlp("U_init", U_init, False)
        for i in range(self.nb_phases):
            self.nlp[i]["X_init"].check_and_adjust_dimensions(self.nlp[i]["nx"], self.nlp[i]["ns"])
            self.nlp[i]["U_init"].check_and_adjust_dimensions(self.nlp[i]["nu"], self.nlp[i]["ns"] - 1)

        # Variables and constraint for the optimization program
        self.V = []
        self.V_bounds = Bounds(interpolation_type=InterpolationType.CONSTANT)
        self.V_init = InitialConditions(interpolation_type=InterpolationType.CONSTANT)
        for i in range(self.nb_phases):
            self.__define_multiple_shooting_nodes_per_phase(self.nlp[i], i)

        # Declare the parameters to optimize
        self.param_to_optimize = {}
        self.__define_variable_time(initial_time_guess, time_min, time_max)

        # Define dynamic problem
        self.__add_to_nlp("ode_solver", ode_solver, True)
        for i in range(self.nb_phases):
            if self.nlp[0]["nx"] != self.nlp[i]["nx"] or self.nlp[0]["nu"] != self.nlp[i]["nu"]:
                raise RuntimeError("Dynamics with different nx or nu is not supported yet")
            self.__prepare_dynamics(self.nlp[i])

        # Prepare constraints
        self.g = []
        self.g_bounds = Bounds(interpolation_type=InterpolationType.CONSTANT)
        ConstraintFunction.continuity_constraint(self)
        if len(constraints) > 0:
            for i in range(self.nb_phases):
                ConstraintFunction.add(self, self.nlp[i])

        # Objective functions
        self.J = 0
        if len(objective_functions) > 0:
            for i in range(self.nb_phases):
                ObjectiveFunction.add(self, self.nlp[i])

        if show_online_optim:
            self.show_online_optim_callback = OnlineCallback(self)
        else:
            self.show_online_optim_callback = None

    def __add_to_nlp(self, param_name, param, duplicate_if_size_is_one, _type=None):
        if isinstance(param, (list, tuple)):
            if len(param) != self.nb_phases:
                raise RuntimeError(
                    f"{param_name} size({len(param)}) does not correspond to the number of phases({self.nb_phases})."
                )
            else:
                for i in range(self.nb_phases):
                    self.nlp[i][param_name] = param[i]
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
        :param dynamics_func: A selected method handler of the class dynamics.Dynamics.
        :param ode_solver: Name of chosen ode, available in OdeSolver enum class.
        """

        dynamics = nlp["dynamics_func"]
        ode_opt = {"t0": 0, "tf": nlp["dt"]}
        if nlp["ode_solver"] == OdeSolver.RK or nlp["ode_solver"] == OdeSolver.COLLOCATION:
            ode_opt["number_of_finite_elements"] = 5

        ode = {"x": nlp["x"], "p": nlp["u"], "ode": dynamics(nlp["x"], nlp["u"])}
        nlp["dynamics"] = []
        if nlp["ode_solver"] == OdeSolver.RK:
            ode_opt["idx"] = 0
            ode["ode"] = dynamics
            if "external_forces" in nlp:
                for idx in range(len(nlp["external_forces"])):
                    ode_opt["idx"] = idx
                    nlp["dynamics"].append(RK4(ode, ode_opt))
            else:
                nlp["dynamics"].append(RK4(ode, ode_opt))
        elif nlp["ode_solver"] == OdeSolver.COLLOCATION:
            if isinstance(nlp["tf"], casadi.MX):
                raise RuntimeError("OdeSolver.COLLOCATION cannot be used while optimizing the time parameter")
            if "external_forces" in nlp:
                raise RuntimeError("COLLOCATION cannot be used with external_forces")
            nlp["dynamics"].append(casadi.integrator("integrator", "collocation", ode, ode_opt))
        elif nlp["ode_solver"] == OdeSolver.CVODES:
            if isinstance(nlp["tf"], casadi.MX):
                raise RuntimeError("OdeSolver.CVODES cannot be used while optimizing the time parameter")
            if "external_forces" in nlp:
                raise RuntimeError("CVODES cannot be used with external_forces")
            nlp["dynamics"].append(casadi.integrator("integrator", "cvodes", ode, ode_opt))

        if len(nlp["dynamics"]) == 1:
            nlp["dynamics"] = nlp["dynamics"] * nlp["ns"]

    def __define_multiple_shooting_nodes_per_phase(self, nlp, idx_phase):
        """
        For each node, puts X_bounds and U_bounds in V_bounds.
        Links X and U with V.
        :param nlp: The nlp problem
        """
        X = []
        U = []

        nV = nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * nlp["ns"]
        V = MX.sym("V_" + str(idx_phase), nV)
        V_bounds = Bounds([0] * nV, [0] * nV, interpolation_type=InterpolationType.CONSTANT)
        V_init = InitialConditions([0] * nV, interpolation_type=InterpolationType.CONSTANT)

        offset = 0
        for k in range(nlp["ns"] + 1):
            X.append(V.nz[offset : offset + nlp["nx"]])
            V_bounds.min[offset : offset + nlp["nx"], 0] = nlp["X_bounds"].min.evaluate_at(shooting_point=k)
            V_bounds.max[offset : offset + nlp["nx"], 0] = nlp["X_bounds"].max.evaluate_at(shooting_point=k)
            V_init.init[offset : offset + nlp["nx"], 0] = nlp["X_init"].init.evaluate_at(shooting_point=k)
            offset += nlp["nx"]

            if k != nlp["ns"]:
                U.append(V.nz[offset : offset + nlp["nu"]])
                V_bounds.min[offset : offset + nlp["nu"], 0] = nlp["U_bounds"].min.evaluate_at(shooting_point=k)
                V_bounds.max[offset : offset + nlp["nu"], 0] = nlp["U_bounds"].max.evaluate_at(shooting_point=k)
                V_init.init[offset : offset + nlp["nu"], 0] = nlp["U_init"].init.evaluate_at(shooting_point=k)
                offset += nlp["nu"]

        V_bounds.check_and_adjust_dimensions(nV, 1)
        V_init.check_and_adjust_dimensions(nV, 1)

        nlp["X"] = X
        nlp["U"] = U
        self.V = vertcat(self.V, V)

        self.V_bounds.concatenate(V_bounds)
        self.V_init.concatenate(V_init)

    def __init_phase_time(self, phase_time):
        if isinstance(phase_time, (int, float)):
            phase_time = [phase_time]
        phase_time = list(phase_time)
        initial_time_guess, time_min, time_max = [], [], []
        for i, nlp in enumerate(self.nlp):
            if "objective_functions" in nlp:
                for obj_fun in nlp["objective_functions"]:
                    if (
                        obj_fun["type"] == Objective.Mayer.MINIMIZE_TIME
                        or obj_fun["type"] == Objective.Lagrange.MINIMIZE_TIME
                    ):
                        initial_time_guess.append(phase_time[i])
                        phase_time[i] = casadi.MX.sym(f"time_phase_{i}", 1, 1)
                        time_min.append(obj_fun["minimum"] if "minimum" in obj_fun else 0)
                        time_max.append(obj_fun["maximum"] if "maximum" in obj_fun else inf)
        return phase_time, initial_time_guess, time_min, time_max

    def __define_variable_time(self, initial_guess, minimum, maximum):
        """
        For each variable time, puts X_bounds and U_bounds in V_bounds.
        Links X and U with V.
        :param nlp: The nlp problem
        :param initial_guess: The initial values taken from the phase_time vector
        :param minimum: variable time minimums as set by user (default: 0)
        :param maximum: vairable time maximums as set by user (default: inf)
        """
        P = []
        for nlp in self.nlp:
            if isinstance(nlp["tf"], MX):
                self.V = vertcat(self.V, nlp["tf"])
                P.append(self.V[-1])
        self.param_to_optimize["time"] = P

        nV = len(initial_guess)
        V_bounds = Bounds(minimum, maximum, interpolation_type=InterpolationType.CONSTANT)
        V_bounds.check_and_adjust_dimensions(nV, 1)
        self.V_bounds.concatenate(V_bounds)

        V_init = InitialConditions(initial_guess, interpolation_type=InterpolationType.CONSTANT)
        V_init.check_and_adjust_dimensions(nV, 1)
        self.V_init.concatenate(V_init)

    def __init_penality(self, penalities, penality_type):
        if len(penalities) > 0:
            if self.nb_phases == 1:
                if isinstance(penalities, dict):
                    penalities = (penalities,)
                if isinstance(penalities[0], dict):
                    penalities = (penalities,)
            elif isinstance(penalities, (list, tuple)):
                for constraint in penalities:
                    if isinstance(constraint, dict):
                        raise RuntimeError(f"Each phase must declares its {penality_type} (even if it is empty)")
            self.__add_to_nlp(penality_type, penalities, False)

    def solve(self, solver="ipopt", options_ipopt={}):
        """
        Gives to CasADi states, controls, constraints, sum of all objective functions and theirs bounds.
        Gives others parameters to control how solver works.
        """

        # NLP
        nlp = {"x": self.V, "f": self.J, "g": self.g}

        options_common = {
            "iteration_callback": self.show_online_optim_callback,
        }
        if solver == "ipopt":
            options_default = {
                "ipopt.tol": 1e-6,
                "ipopt.max_iter": 2,
                "ipopt.hessian_approximation": "exact",  # "exact", "limited-memory"
                "ipopt.limited_memory_max_history": 50,
                "ipopt.linear_solver": "mumps",  # "ma57", "ma86", "mumps"
            }
            for key in options_ipopt:
                if key[:6] != "ipopt.":
                    options_ipopt[f"ipopt.{key}"] = options_ipopt[key]
                    del options_ipopt[key]
            opts = {**options_default, **options_common, **options_ipopt}
            solver = casadi.nlpsol("nlpsol", "ipopt", nlp, opts)
        else:
            raise RuntimeError("Available solvers are: 'ipopt'")

        # Bounds and initial guess
        arg = {
            "lbx": self.V_bounds.min,
            "ubx": self.V_bounds.max,
            "lbg": self.g_bounds.min,
            "ubg": self.g_bounds.max,
            "x0": self.V_init.init,
        }

        # Solve the problem
        return solver.call(arg)

    def save(self, sol, file_path, as_numpy=False):
        _, ext = os.path.splitext(file_path)
        if ext == "":
            file_path = file_path + ".bo"
        with open(file_path, "wb") as file:
            if as_numpy:
                pickle.dump({"data": Data.get_data(self, sol["x"])}, file)
            else:
                pickle.dump({"ocp_initilializer": self.original_values, "sol": sol, "versions": self.version}, file)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            ocp = OptimalControlProgram(**data["ocp_initilializer"])
            for key in data["versions"].keys():
                if data["versions"][key] != ocp.version[key]:
                    raise RuntimeError(
                        f"Version of {key} from file ({data['versions'][key]}) is not the same as the "
                        f"installed version ({ocp.version[key]})"
                    )
            sol = data["sol"]
        return (ocp, sol)
