import os
import pickle

from casadi import vertcat, sum1, nlpsol

from .solver_interface import SolverInterface
from ..gui.plot import OnlineCallback
from ..limits.objective_functions import get_objective_value
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType


class IpoptInterface(SolverInterface):
    def __init__(self, ocp):
        super().__init__(ocp)

        self.options_common = {}
        self.opts = None

        self.ipopt_nlp = None
        self.ipopt_limits = None
        self.ocp_solver = None

        self.bobo_directory = ".__tmp_biorbd_optim"
        self.bobo_file_path = ".__tmp_biorbd_optim/temp_save_iter.bobo"

    def online_optim(self, ocp):
        self.options_common["iteration_callback"] = OnlineCallback(ocp)

    def start_get_iterations(self):
        if os.path.isfile(self.bobo_file_path):
            os.remove(self.bobo_file_path)
            os.rmdir(self.bobo_directory)
        os.mkdir(self.bobo_directory)

        with open(self.bobo_file_path, "wb") as file:
            pickle.dump([], file)

    def finish_get_iterations(self):
        with open(self.bobo_file_path, "rb") as file:
            self.out["sol_iterations"] = pickle.load(file)
            os.remove(self.bobo_file_path)
            os.rmdir(self.bobo_directory)

    def configure(self, solver_options):
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

    def solve(self):
        self.__dispatch_bounds()
        solver = nlpsol("nlpsol", "ipopt", self.ipopt_nlp, self.opts)

        # Solve the problem
        self.out = {"sol": solver.call(self.ipopt_limits)}
        self.out["sol"]["time_tot"] = solver.stats()["t_wall_total"]

        return self.out

    def __dispatch_bounds(self):
        all_J = self.__dispatch_obj_func()
        all_g = self.ocp.CX()
        all_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for i in range(len(self.ocp.g)):
            for j in range(len(self.ocp.g[i])):
                all_g = vertcat(all_g, self.ocp.g[i][j])
                all_g_bounds.concatenate(self.ocp.g_bounds[i][j])
        for nlp in self.ocp.nlp:
            for i in range(len(nlp.g)):
                for j in range(len(nlp.g[i])):
                    all_g = vertcat(all_g, nlp.g[i][j])
                    all_g_bounds.concatenate(nlp.g_bounds[i][j])

        self.ipopt_nlp = {"x": self.ocp.V, "f": sum1(all_J), "g": all_g}

        self.ipopt_limits = {
            "lbx": self.ocp.V_bounds.min,
            "ubx": self.ocp.V_bounds.max,
            "lbg": all_g_bounds.min,
            "ubg": all_g_bounds.max,
            "x0": self.ocp.V_init.init,
        }

    def __dispatch_obj_func(self):
        all_J = self.ocp.CX()
        for j_nodes in self.ocp.J:
            for obj in j_nodes:
                all_J = vertcat(all_J, get_objective_value(obj))
        for nlp in self.ocp.nlp:
            for obj_nodes in nlp.J:
                for obj in obj_nodes:
                    all_J = vertcat(all_J, get_objective_value(obj))

        return all_J
