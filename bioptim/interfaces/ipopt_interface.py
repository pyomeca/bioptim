import os
import pickle

from casadi import vertcat, sum1, nlpsol, SX, MX

from .solver_interface import SolverInterface
from ..gui.plot import OnlineCallback
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType


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

        self.options_common["iteration_callback"] = OnlineCallback(ocp)

    def configure(self, solver_options: dict):
        """
        Set some Ipopt options

        Parameters
        ----------
        solver_options: dict
            The dictionary of options
        """

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

        all_J = self.__dispatch_obj_func()
        all_g, all_g_bounds = self.__dispatch_bounds()

        self.ipopt_nlp = {"x": self.ocp.v.vector, "f": sum1(all_J), "g": all_g}
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
        self.out = {"sol": solver.call(self.ipopt_limits)}
        self.out["sol"]["time_tot"] = solver.stats()["t_wall_total"]
        # To match acados convention (0 = success, 1 = error)
        self.out["sol"]["status"] = int(not solver.stats()["success"])

        return self.out

    def set_lagrange_multiplier(self, sol: dict):
        """
        Set the lagrange multiplier from a solution structure

        Parameters
        ----------
        sol: dict
            A solution structure where the lagrange multipliers are set
        """

        self.lam_g = sol["lam_g"]
        self.lam_x = sol["lam_x"]

    def __dispatch_bounds(self):
        """
        Parse the bounds of the full ocp to a Ipopt-friendly one
        """
        # TODO: This should be done in bounds, so it is available for all the code

        all_g = self.ocp.CX()
        all_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for i in range(len(self.ocp.g)):
            for j in range(len(self.ocp.g[i])):
                all_g = vertcat(all_g, self.ocp.g[i][j]["val"])
                all_g_bounds.concatenate(self.ocp.g[i][j]["bounds"])
        for nlp in self.ocp.nlp:
            for i in range(len(nlp.g)):
                for j in range(len(nlp.g[i])):
                    all_g = vertcat(all_g, nlp.g[i][j]["val"])
                    all_g_bounds.concatenate(nlp.g[i][j]["bounds"])

        if isinstance(all_g_bounds.min, (SX, MX)) or isinstance(all_g_bounds.max, (SX, MX)):
            raise RuntimeError("Ipopt doesn't support SX/MX types in constraints bounds")
        return all_g, all_g_bounds

    def __dispatch_obj_func(self):
        """
        Parse the objective functions of the full ocp to a Ipopt-friendly one
        """
        # TODO: This should be done in bounds, so it is available for all the code

        all_J = self.ocp.CX()
        for j_nodes in self.ocp.J:
            for obj in j_nodes:
                all_J = vertcat(all_J, IpoptInterface.finalize_objective_value(obj))
        for nlp in self.ocp.nlp:
            for obj_nodes in nlp.J:
                for obj in obj_nodes:
                    all_J = vertcat(all_J, IpoptInterface.finalize_objective_value(obj))

        return all_J
