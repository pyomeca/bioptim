import os
import pickle

from casadi import vertcat, sum1, nlpsol, dot

from .solver_interface import SolverInterface
from ..gui.plot import OnlineCallback
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
        self.out = None

    def online_optim(self, ocp):
        self.options_common["iteration_callback"] = OnlineCallback(ocp)

    def start_get_iterations(self):
        self.directory = ".__tmp_biorbd_optim"
        self.file_path = ".__tmp_biorbd_optim/temp_save_iter.bobo"

        if os.path.isfile(self.file_path):
            os.remove(self.file_path)
            os.rmdir(self.directory)
        os.mkdir(self.directory)

        with open(self.file_path, "wb") as file:
            pickle.dump([], file)

    def finish_get_iterations(self):
        with open(self.file_path, "rb") as file:
            self.out = self.out, pickle.load(file)
            os.remove(self.file_path)
            os.rmdir(self.directory)

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
        self.out = solver.call(self.ipopt_limits)
        self.out["time_tot"] = solver.stats()["t_wall_total"]

        return self.out

    def get_optimized_value(self, ocp):
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
            for i in range(len(nlp["g"])):
                for j in range(len(nlp["g"][i])):
                    all_g = vertcat(all_g, nlp["g"][i][j])
                    all_g_bounds.concatenate(nlp["g_bounds"][i][j])

        self.ipopt_nlp = {"x": self.ocp.V, "f": sum1(all_J), "g": all_g}

        self.ipopt_limits = {
            "lbx": self.ocp.V_bounds.min,
            "ubx": self.ocp.V_bounds.max,
            "lbg": all_g_bounds.min,
            "ubg": all_g_bounds.max,
            "x0": self.ocp.V_init.init,
        }

    def __dispatch_obj_func(self):
        def get_objective_value(j_dict):
            val = j_dict["val"]
            if j_dict["target"] is not None:
                val -= j_dict["target"]

            if j_dict["objective"].quadratic:
                val = dot(val, val)
            else:
                val = sum1(val)

            val *= j_dict["objective"].weight * j_dict["dt"]
            return val

        all_J = self.ocp.CX()
        for j_nodes in self.ocp.J:
            for obj in j_nodes:
                all_J = vertcat(all_J, get_objective_value(obj))
        for nlp in self.ocp.nlp:
            for obj_nodes in nlp["J"]:
                for obj in obj_nodes:
                    all_J = vertcat(all_J, get_objective_value(obj))

        return all_J
