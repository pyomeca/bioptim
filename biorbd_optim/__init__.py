import enum

import biorbd
import casadi
from casadi import MX

from .constraints import Constraint
from .problem_type import ProblemType
from .plot import AnimateCallback
from .path_conditions import Bounds, InitialConditions


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
        show_online_optim=False,
        is_cyclic_objective=False,
        is_cyclic_constraint=False,
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

        self.nb_phases = len(biorbd_model)
        if isinstance(biorbd_model, str):
            self.model = biorbd.Model(biorbd_model)
        elif isinstance(biorbd_model, biorbd.biorbd.Model):
            self.model = biorbd_model
        else:
            raise RuntimeError(
                "biorbd_model must either be a string or an instance of biorbd.Model()"
            )

        # Define some aliases
        self.ns = number_shooting_points
        self.tf = phase_time
        self.dt = phase_time / max(number_shooting_points, 1)
        self.is_cyclic_constraint = is_cyclic_constraint
        self.is_cyclic_objective = is_cyclic_objective

        # Compute problem size
        self.x = MX()
        self.u = MX()
        self.nx = -1
        self.nu = -1
        self.nbQ = -1
        self.nbQdot = -1
        self.nbTau = -1
        self.nbMuscleTotal = -1
        self.dynamics_func = None
        self.dof_mapping = dof_mapping
        self.problem_type = problem_type
        self.problem_type(self)

        X_init.regulation(self.nx)
        X_bounds.regulation(self.nx)
        U_init.regulation(self.nu)
        U_bounds.regulation(self.nu)

        # Variables and constraint for the optimization program
        self.X = []
        self.U = []
        self.V = MX()
        self.V_init = InitialConditions()
        self.V_bounds = Bounds()
        self.g = []
        self.g_bounds = Bounds()
        self.__define_multiple_shooting_nodes(X_init, U_init, X_bounds, U_bounds)

        # Define dynamic problem
        self.ode_solver = ode_solver
        self.__prepare_dynamics()
        Constraint.continuity_constraint(self)

        # Constraint functions
        self.constraints = constraints
        Constraint.add_constraints(self)

        # Objective functions
        self.J = 0
        for (func, weight) in objective_functions:
            func(self, weight=weight)

        if show_online_optim:
            self.show_online_optim_callback = AnimateCallback(self)
        else:
            self.show_online_optim_callback = None

    @staticmethod
    def __convert_to_tuple(param):
        if not isinstance(param, (list, tuple)):
            return param,

    def __prepare_dynamics(self):
        """
        Builds CasaDI dynamics function.
        :param dynamics_func: A selected method handler of the class dynamics.Dynamics.
        :param ode_solver: Name of chosen ode, available in OdeSolver enum class.
        """

        states = MX.sym("x", self.nx, 1)
        controls = MX.sym("p", self.nu, 1)
        dynamics = casadi.Function(
            "ForwardDyn",
            [states, controls],
            [self.dynamics_func(states, controls, self)],
            ["states", "controls"],
            ["statesdot"],
        ).expand()
        ode = {"x": self.x, "p": self.u, "ode": dynamics(self.x, self.u)}

        ode_opt = {"t0": 0, "tf": self.dt}
        if self.ode_solver == OdeSolver.RK or self.ode_solver == OdeSolver.COLLOCATION:
            ode_opt["number_of_finite_elements"] = 5

        if self.ode_solver == OdeSolver.RK:
            self.dynamics = casadi.integrator("integrator", "rk", ode, ode_opt)
        elif self.ode_solver == OdeSolver.COLLOCATION:
            self.dynamics = casadi.integrator("integrator", "collocation", ode, ode_opt)
        elif self.ode_solver == OdeSolver.CVODES:
            self.dynamics = casadi.integrator("integrator", "cvodes", ode, ode_opt)

    def __define_multiple_shooting_nodes(self, X_init, U_init, X_bounds, U_bounds):
        """
        For each node, puts X_bounds and U_bounds in V_bounds.
        Links X and U with V.
        :param X_init: Instance of the class InitialConditions for the states.
        :param U_init: Instance of the class InitialConditions for the controls.
        :param X_bounds: Instance of the class Bounds for the states.
        :param U_bounds: Instance of the class Bounds for the controls.
        """
        nV = self.nx * (self.ns + 1) + self.nu * self.ns
        self.V = MX.sym("V", nV)
        self.V_bounds.min = [0] * nV
        self.V_bounds.max = [0] * nV
        self.V_init.init = [0] * nV

        offset = 0
        for k in range(self.ns):
            self.X.append(self.V.nz[offset : offset + self.nx])
            if k == 0:
                self.V_bounds.min[offset : offset + self.nx] = X_bounds.first_node_min
                self.V_bounds.max[offset : offset + self.nx] = X_bounds.first_node_max
            else:
                self.V_bounds.min[offset : offset + self.nx] = X_bounds.min
                self.V_bounds.max[offset : offset + self.nx] = X_bounds.max
            self.V_init.init[offset : offset + self.nx] = X_init.init
            offset += self.nx

            self.U.append(self.V.nz[offset : offset + self.nu])
            if k == 0:
                self.V_bounds.min[offset : offset + self.nu] = U_bounds.first_node_min
                self.V_bounds.max[offset : offset + self.nu] = U_bounds.first_node_max
            else:
                self.V_bounds.min[offset : offset + self.nu] = U_bounds.min
                self.V_bounds.max[offset : offset + self.nu] = U_bounds.max
            self.V_init.init[offset : offset + self.nu] = U_init.init
            offset += self.nu

        self.X.append(self.V.nz[offset : offset + self.nx])
        self.V_bounds.min[offset : offset + self.nx] = X_bounds.last_node_min
        self.V_bounds.max[offset : offset + self.nx] = X_bounds.last_node_max
        self.V_init.init[offset : offset + self.nx] = X_init.init

        self.V_init.regulation(nV)
        self.V_bounds.regulation(nV)

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
