import enum

import biorbd
import casadi
from casadi import MX, vertcat

from .constraints import Constraint
from .problem_type import ProblemType
from .plot import AnimateCallback
from .path_conditions import Bounds, InitialConditions
from .dynamics import Dynamics


class OdeSolver(enum.Enum):
    """
    Four models to solve.
    RK is pretty much good balance.
    """

    COLLOCATION = 0
    RK = 1
    CVODES = 2
    NO_SOLVER = 3


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
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        ode_solver=OdeSolver.RK,
        dof_mapping=None,
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
            biorbd_model = [
                biorbd.Model(m) if isinstance(m, str) else m for m in biorbd_model
            ]
        else:
            raise RuntimeError(
                "biorbd_model must either be a string or an instance of biorbd.Model()"
            )
        self.nb_phases = len(biorbd_model)
        self.nlp = [{}] * self.nb_phases
        self.__add_to_nlp("model", biorbd_model, False)

        # Define some aliases
        self.__add_to_nlp("ns", number_shooting_points, False)
        self.__add_to_nlp("tf", phase_time, False)
        self.__add_to_nlp(
            "dt",
            [
                self.nlp[i]["tf"] / max(self.nlp[i]["ns"], 1)
                for i in range(self.nb_phases)
            ],
            False,
        )
        self.is_cyclic_constraint = is_cyclic_constraint
        self.is_cyclic_objective = is_cyclic_objective

        # Compute problem size
        self.__add_to_nlp("dof_mapping", dof_mapping, dof_mapping is None)
        self.__add_to_nlp("problem_type", problem_type, False)
        for i in range(self.nb_phases):
            self.nlp[i]["problem_type"](self.nlp[i])

        # Prepare path constraints
        self.__add_to_nlp("X_bounds", X_bounds, False)
        self.__add_to_nlp("U_bounds", U_bounds, False)
        for i in range(self.nb_phases):
            self.nlp[i]["X_bounds"].regulation(self.nlp[i]["nx"])
            self.nlp[i]["U_bounds"].regulation(self.nlp[i]["nu"])

        # Prepare initial guesses
        self.__add_to_nlp("X_init", X_init, False)
        self.__add_to_nlp("U_init", U_init, False)
        for i in range(self.nb_phases):
            self.nlp[i]["X_init"].regulation(self.nlp[i]["nx"])
            self.nlp[i]["U_init"].regulation(self.nlp[i]["nu"])

        # Variables and constraint for the optimization program
        self.V = []
        self.V_bounds = Bounds()
        self.V_init = InitialConditions()
        for i in range(self.nb_phases):
            self.__define_multiple_shooting_nodes_per_phase(self.nlp[i])

        # Define dynamic problem
        self.__add_to_nlp("ode_solver", ode_solver, True)
        states = MX.sym("x", self.nlp[0]["nx"], 1)
        controls = MX.sym("u", self.nlp[0]["nu"], 1)
        for i in range(self.nb_phases):
            if (
                self.nlp[0]["nx"] != self.nlp[i]["nx"]
                or self.nlp[0]["nu"] != self.nlp[i]["nu"]
            ):
                raise RuntimeError(
                    "Dynamics with different nx or nu is not supported yet"
                )
            self.__prepare_dynamics(self.nlp[i], states, controls)

        # Prepare constraints
        self.g = []
        self.g_bounds = Bounds()
        Constraint.continuity_constraint(self)

        if self.nb_phases == 1:
            if (
                isinstance(constraints, (list, tuple))
                and isinstance(constraints[0], (list, tuple))
                and not isinstance(constraints[0][0], (list, tuple))
            ):
                constraints = (constraints,)
        self.__add_to_nlp("constraints", constraints, False)
        for i in range(self.nb_phases):
            Constraint.add_constraints(self, self.nlp[i])

        # Objective functions
        self.J = 0
        if self.nb_phases == 1:
            if (
                isinstance(objective_functions, (list, tuple))
                and isinstance(objective_functions[0], (list, tuple))
                and not isinstance(objective_functions[0][0], (list, tuple))
            ):
                objective_functions = (objective_functions,)
        self.__add_to_nlp("objective_functions", objective_functions, False)
        for i in range(self.nb_phases):
            for (func, params) in self.nlp[i]["objective_functions"]:
                if isinstance(params, dict):
                    func(self, self.nlp[i], **params)
                else:
                    func(self, self.nlp[i], params)

        if show_online_optim:
            self.show_online_optim_callback = AnimateCallback(self)
        else:
            self.show_online_optim_callback = None

    def __add_to_nlp(self, param_name, param, duplicate_if_size_is_one):
        if isinstance(param, (list, tuple)):
            if len(param) != self.nb_phases:
                raise RuntimeError(
                    param_name + " size does not correspond to the number of phases"
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
                    raise RuntimeError(
                        param_name
                        + " must be a list or tuple when number of phase is not equal to 1"
                    )

    @staticmethod
    def __prepare_dynamics(nlp, states_sym, controls_sym):
        """
        Builds CasaDI dynamics function.
        :param dynamics_func: A selected method handler of the class dynamics.Dynamics.
        :param ode_solver: Name of chosen ode, available in OdeSolver enum class.
        """

        dynamics = casadi.Function(
            "ForwardDyn",
            [states_sym, controls_sym],
            [nlp["dynamics_func"](states_sym, controls_sym, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()

        ode = {"x": nlp["x"], "p": nlp["u"], "ode": dynamics(nlp["x"], nlp["u"])}

        ode_opt = {"t0": 0, "tf": nlp["dt"]}
        if (
            nlp["ode_solver"] == OdeSolver.RK
            or nlp["ode_solver"] == OdeSolver.COLLOCATION
        ):
            ode_opt["number_of_finite_elements"] = 5

        if nlp["ode_solver"] == OdeSolver.RK:
            nlp["dynamics"] = casadi.integrator("integrator", "rk", ode, ode_opt)
        elif nlp["ode_solver"] == OdeSolver.COLLOCATION:
            nlp["dynamics"] = casadi.integrator(
                "integrator", "collocation", ode, ode_opt
            )
        elif nlp["ode_solver"] == OdeSolver.CVODES:
            nlp["dynamics"] = casadi.integrator("integrator", "cvodes", ode, ode_opt)

    def __define_multiple_shooting_nodes_per_phase(self, nlp):
        """
        For each node, puts X_bounds and U_bounds in V_bounds.
        Links X and U with V.
        :param nlp: The nlp problem
        """
        X = []
        U = []

        nV = nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * nlp["ns"]
        V = MX.sym("V", nV)
        V_bounds = Bounds([0] * nV, [0] * nV)
        V_init = InitialConditions([0] * nV)

        offset = 0
        for k in range(nlp["ns"]):
            X.append(V.nz[offset : offset + nlp["nx"]])
            if k == 0:
                V_bounds.min[offset : offset + nlp["nx"]] = nlp[
                    "X_bounds"
                ].first_node_min
                V_bounds.max[offset : offset + nlp["nx"]] = nlp[
                    "X_bounds"
                ].first_node_max
            else:
                V_bounds.min[offset : offset + nlp["nx"]] = nlp["X_bounds"].min
                V_bounds.max[offset : offset + nlp["nx"]] = nlp["X_bounds"].max
            V_init.init[offset : offset + nlp["nx"]] = nlp["X_init"].init
            offset += nlp["nx"]

            U.append(V.nz[offset : offset + nlp["nu"]])
            if k == 0:
                V_bounds.min[offset : offset + nlp["nu"]] = nlp[
                    "U_bounds"
                ].first_node_min
                V_bounds.max[offset : offset + nlp["nu"]] = nlp[
                    "U_bounds"
                ].first_node_max
            else:
                V_bounds.min[offset : offset + nlp["nu"]] = nlp["U_bounds"].min
                V_bounds.max[offset : offset + nlp["nu"]] = nlp["U_bounds"].max
            V_init.init[offset : offset + nlp["nu"]] = nlp["U_init"].init
            offset += nlp["nu"]

        X.append(V.nz[offset : offset + nlp["nx"]])
        V_bounds.min[offset : offset + nlp["nx"]] = nlp["X_bounds"].last_node_min
        V_bounds.max[offset : offset + nlp["nx"]] = nlp["X_bounds"].last_node_max
        V_init.init[offset : offset + nlp["nx"]] = nlp["X_init"].init

        V_bounds.regulation(nV)
        V_init.regulation(nV)

        nlp["X"] = X
        nlp["U"] = U
        self.V = vertcat(self.V, V)
        self.V_bounds.expand(V_bounds)
        self.V_init.expand(V_init)

    def solve(self):
        """
        Gives to CasADi states, controls, constraints, sum of all objective functions and theirs bounds.
        Gives others parameters to control how solver works.
        """

        # NLP
        nlp = {"x": self.V, "f": self.J, "g": self.g}

        opts = {
            "ipopt.tol": 1e-6,
            "ipopt.max_iter": 1000,
            "ipopt.hessian_approximation": "exact",  # "exact", "limited-memory"
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "mumps",  # "ma57", "ma86", "mumps"
            "iteration_callback": self.show_online_optim_callback,
        }
        solver = casadi.nlpsol("nlpsol", "ipopt", nlp, opts)

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

    def show(self):
        pass
