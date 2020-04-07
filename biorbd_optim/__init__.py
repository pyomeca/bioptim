import enum

import casadi
from casadi import MX

from .constraints import Constraint
from .variable import Variable
from .plot import AnimateCallback
from .mapping import Mapping


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
        variable_type,
        dynamics_func,
        ode_solver,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
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
        :param variable_type: A selected method handler of the class biorbd_optim.Variable.
        :param dynamics_func: A selected method handler of the class dynamics.Dynamics.
        :param ode_solver: Name of chosen ode, available in OdeSolver enum class.
        :param number_shooting_points: Subdivision number.
        :param final_time: Simulation time in seconds.
        :param objective_functions: Tuple of tuple of objectives functions handler's and weights.
        :param X_bounds: Instance of the class Bounds.
        :param U_bounds: Instance of the class Bounds.
        :param constraints: Tuple of constraints, instant (which node(s)) and tuple of geometric structures used.
        """
        self.model = biorbd_model

        # Define some aliases
        self.ns = number_shooting_points
        self.tf = final_time
        self.dt = final_time / max(number_shooting_points, 1)
        self.is_cyclic_constraint = is_cyclic_constraint
        self.is_cyclic_objective = is_cyclic_objective

        # Compute problem size
        self.x = MX()
        self.u = MX()
        self.nx = -1
        self.nu = -1
        self.variable_type = variable_type
        if dof_mapping is None:
            if self.variable_type == Variable.torque_driven:
                self.dof_mapping = Mapping(
                    range(self.model.nbQ()), range(self.model.nbQ())
                )
            else:
                raise RuntimeError("This Variable has no default dof_mapping")
        else:
            self.dof_mapping = dof_mapping
        self.nbQ = -1
        self.variable_type(self)

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
        self.__prepare_dynamics(biorbd_model, dynamics_func, ode_solver)
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

    def __prepare_dynamics(self, biorbd_model, dynamics_func, ode_solver):
        """
        Builds CasaDI dynamics function.
        :param biorbd_model: Biorbd model loaded from the biorbd.Model() function.
        :param dynamics_func: A selected method handler of the class dynamics.Dynamics.
        :param ode_solver: Name of chosen ode, available in OdeSolver enum class.
        """

        states = MX.sym("x", self.nx, 1)
        controls = MX.sym("p", self.nu, 1)
        dynamics = casadi.Function(
            "ForwardDyn",
            [states, controls],
            [dynamics_func(states, controls, self)],
            ["states", "controls"],
            ["statesdot"],
        ).expand()
        ode = {"x": self.x, "p": self.u, "ode": dynamics(self.x, self.u)}

        ode_opt = {"t0": 0, "tf": self.dt}
        if ode_solver == OdeSolver.RK or ode_solver == OdeSolver.COLLOCATION:
            ode_opt["number_of_finite_elements"] = 5

        if ode_solver == OdeSolver.RK:
            self.dynamics = casadi.integrator("integrator", "rk", ode, ode_opt)
        elif ode_solver == OdeSolver.COLLOCATION:
            self.dynamics = casadi.integrator("integrator", "collocation", ode, ode_opt)
        elif ode_solver == OdeSolver.CVODES:
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


class PathCondition:
    """
    Parent class of Bounds and InitialConditions.
    Uses only for methods overloading.
    """

    @staticmethod
    def regulation(var, nb_elements):
        pass

    @staticmethod
    def regulation_private(var, nb_elements, type):
        if len(var) != nb_elements:
            raise RuntimeError(f"Invalid number of {type}")


class Bounds(PathCondition):
    """
    Organizes bounds of states("X"), controls("U") and "V".
    """

    def __init__(self):
        """
        There are 3 groups of nodes :
        1. First node
        2. Intermediates (= all nodes except first and last nodes)
        3. Last node
        Each group have 2 lists of bounds : one of minimum and one of maximum values.

        For X and Y bounds, lists have the number of degree of freedom elements.
        For V bounds, lists have number of degree of freedom elements * number of shooting points.
        """
        self.min = []
        self.first_node_min = []
        self.last_node_min = []

        self.max = []
        self.first_node_max = []
        self.last_node_max = []

    def regulation(self, nb_elements):
        """
        Detects if bounds are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same bounds than intermediates nodes.
        :param nb_elements: Length of each list.
        """
        self.regulation_private(self.min, nb_elements, "Bound min")
        self.regulation_private(self.max, nb_elements, "Bound max")

        if len(self.first_node_min) == 0:
            self.first_node_min = self.min
        if len(self.last_node_min) == 0:
            self.last_node_min = self.min

        if len(self.first_node_max) == 0:
            self.first_node_max = self.max
        if len(self.last_node_max) == 0:
            self.last_node_max = self.max

        self.regulation_private(
            self.first_node_min, nb_elements, "Bound first node min"
        )
        self.regulation_private(
            self.first_node_max, nb_elements, "Bound first node max"
        )
        self.regulation_private(self.last_node_min, nb_elements, "Bound last node min")
        self.regulation_private(self.last_node_max, nb_elements, "Bound last node max")


class InitialConditions(PathCondition):
    def __init__(self):
        """
        Organises initial values (for solver)
        There are 3 groups of nodes :
        1. First node
        2. Intermediates (= all nodes without first and last nodes)
        3. Last node
        Each group have a list of initial values.
        """
        self.first_node_init = []
        self.init = []
        self.last_node_init = []

    def regulation(self, nb_elements):
        """
        Detects if initial values are not given, in that case "0" is given for all degrees of freedom.
        Detects if initial values are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same  values than intermediates nodes.
        """
        if len(self.init) == 0:
            self.init = [0] * nb_elements
        self.regulation_private(self.init, nb_elements, "Init")

        if len(self.first_node_init) == 0:
            self.first_node_init = self.init
        if len(self.last_node_init) == 0:
            self.last_node_init = self.init

        self.regulation_private(self.first_node_init, nb_elements, "First node init")
        self.regulation_private(self.last_node_init, nb_elements, "Last node init")
