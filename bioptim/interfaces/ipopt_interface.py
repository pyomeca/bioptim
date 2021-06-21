from time import time
from sys import platform

from casadi import horzcat, vertcat, sum1, nlpsol, SX, MX

from .solver_interface import SolverInterface
from ..gui.plot import OnlineCallback
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType
from ..optimization.solution import Solution


class IpoptInterface(SolverInterface):
    """
    The Ipopt solver interface

    Attributes
    ----------
    options_common: dict
        Options irrelevant of a specific ocp
    opts: dict
        Options of the current ocp
    ipopt_nlp: dict
        The declaration of the variables Ipopt-friendly
    ipopt_limits: dict
        The declaration of the bound Ipopt-friendly
    lam_g: np.ndarray
        The lagrange multiplier of the constraints to initialize the solver
    lam_x: np.ndarray
        The lagrange multiplier of the variables to initialize the solver

    Methods
    -------
    online_optim(self, ocp: OptimalControlProgram)
        Declare the online callback to update the graphs while optimizing
    configure(self, solver_options: dict)
        Set some Ipopt options
    solve(self) -> dict
        Solve the prepared ocp
    set_lagrange_multiplier(self, sol: dict)
        Set the lagrange multiplier from a solution structure
    __dispatch_bounds(self)
        Parse the bounds of the full ocp to a Ipopt-friendly one
    __dispatch_obj_func(self)
        Parse the objective functions of the full ocp to a Ipopt-friendly one
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        super().__init__(ocp)

        self.options_common = {}
        self.opts = {}

        self.ipopt_nlp = {}
        self.ipopt_limits = {}
        self.ocp_solver = None

        self.lam_g = None
        self.lam_x = None

    def online_optim(self, ocp):
        """
        Declare the online callback to update the graphs while optimizing

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """
        if platform == "win32":
            raise RuntimeError("Online graphics are not available on Windows")
        self.options_common["iteration_callback"] = OnlineCallback(ocp)

    def configure(self, solver_options: dict):
        """
        Set some Ipopt options

        Parameters
        ----------
        solver_options: dict
            The dictionary of options
        """
        if solver_options is None:
            if self.opts:
                return
            else:
                solver_options = {}

        options = {
            "ipopt.tol": 1e-6,
            "ipopt.max_iter": 1000,
            "ipopt.hessian_approximation": "exact",  # "exact", "limited-memory"
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "mumps",  # "ma57", "ma86", "mumps"
        }
        for key in solver_options:
            ipopt_key = key
            if key[:6] != "ipopt.":
                ipopt_key = "ipopt." + key
            options[ipopt_key] = solver_options[key]
        self.opts = {**options, **self.options_common}

    def solve(self) -> dict:
        """
        Solve the prepared ocp

        Returns
        -------
        A reference to the solution
        """

        all_objectives = self.__dispatch_obj_func()
        all_g, all_g_bounds = self.__dispatch_bounds()

        self.ipopt_nlp = {"x": self.ocp.v.vector, "f": sum1(all_objectives), "g": all_g}
        v_bounds = self.ocp.v.bounds
        v_init = self.ocp.v.init
        self.ipopt_limits = {
            "lbx": v_bounds.min,
            "ubx": v_bounds.max,
            "lbg": all_g_bounds.min,
            "ubg": all_g_bounds.max,
            "x0": v_init.init,
        }

        if self.lam_g is not None:
            self.ipopt_limits["lam_g0"] = self.lam_g
        if self.lam_x is not None:
            self.ipopt_limits["lam_x0"] = self.lam_x

        solver = nlpsol("nlpsol", "ipopt", self.ipopt_nlp, self.opts)

        # Solve the problem
        tic = time()
        self.out = {"sol": solver.call(self.ipopt_limits)}
        self.out["sol"]["time_tot"] = time() - tic
        self.out["sol"]["iter"] = solver.stats()["iter_count"]
        self.out["sol"]["inf_du"] = solver.stats()["iterations"]["inf_du"] if "iteration" in solver.stats() else None
        self.out["sol"]["inf_pr"] = solver.stats()["iterations"]["inf_pr"] if "iteration" in solver.stats() else None
        # To match acados convention (0 = success, 1 = error)
        self.out["sol"]["status"] = int(not solver.stats()["success"])

        return self.out

    def set_lagrange_multiplier(self, sol: Solution):
        """
        Set the lagrange multiplier from a solution structure

        Parameters
        ----------
        sol: dict
            A solution structure where the lagrange multipliers are set
        """

        self.lam_g = sol.lam_g
        self.lam_x = sol.lam_x

    def __dispatch_bounds(self):
        """
        Parse the bounds of the full ocp to a Ipopt-friendly one
        """
        # TODO: This should be done in bounds, so it is available for all the code

        all_g = self.ocp.cx()
        all_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for i in range(len(self.ocp.g)):
            for j in range(len(self.ocp.g[i])):
                all_g = vertcat(all_g, self.ocp.g[i][j]["val"])
                all_g_bounds.concatenate(self.ocp.g[i][j]["bounds"])
        for nlp in self.ocp.nlp:
            for i in range(len(nlp.g)):
                for j in range(len(nlp.g[i])):
                    if nlp.g[i][j]["constraint"].target is not None:
                        # TODO This is not tested and therefore it is not sure it works..
                        # TODO Add an example and test or remove?
                        all_g = vertcat(all_g, nlp.g[i][j]["val"] - nlp.g[i][j]["target"])
                    else:
                        all_g = vertcat(all_g, nlp.g[i][j]["val"])
                    all_g_bounds.concatenate(nlp.g[i][j]["bounds"])

        if isinstance(all_g_bounds.min, (SX, MX)) or isinstance(all_g_bounds.max, (SX, MX)):
            raise RuntimeError("Ipopt doesn't support SX/MX types in constraints bounds")
        return all_g, all_g_bounds

    def __dispatch_obj_func(self):
        """
        Parse the objective functions of the full ocp to a Ipopt-friendly one
        """

        param = self.ocp.cx(self.ocp.v.parameters_in_list.cx)

        def get_all_obj(objectives):
            out = self.ocp.cx()
            for penalty in objectives:
                for idx in penalty.node_idx:
                    target = penalty.target[:, idx] if penalty.target is not None and idx < len(penalty.target) else []
                    if penalty.derivative:
                        x = horzcat(*nlp.X[idx:idx+2])
                        u = horzcat(*nlp.U[idx:idx+2]) if idx < len(nlp.U) - 1 else []

                    else:
                        x = nlp.X[idx]
                        u = nlp.U[idx] if idx < len(nlp.U) else []
                    p = penalty.weighted_function(x, u, param, penalty.weight, target, penalty.dt)
                    out = vertcat(out, p)
            return out

        all_objectives = self.ocp.cx()
        all_objectives = vertcat(all_objectives, get_all_obj(self.ocp.J))
        for nlp in self.ocp.nlp:
            all_objectives = vertcat(all_objectives, get_all_obj(nlp.J))

        return all_objectives
