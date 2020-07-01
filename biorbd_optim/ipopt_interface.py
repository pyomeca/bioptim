import os
import pickle

from casadi import vertcat, sum1, nlpsol

from .plot import OnlineCallback
from .solver_interface import SolverInterface
from .path_conditions import Bounds, InterpolationType


class IpoptInterface(SolverInterface):
    def __init__(self, ocp):
        super().__init__()

        self.options_common = {}
        self.opts = None

        self.nlp, self.arg = self.__dispatch_bounds(ocp)
        self.ocp_solver = None
        self.out = None

    def online_optim(self, ocp):
        self.options_common["iteration_callback"] = OnlineCallback(ocp)

    def get_iterations(self):
        directory = ".__tmp_biorbd_optim"
        file_path = ".__tmp_biorbd_optim/temp_save_iter.bobo"
        os.mkdir(directory)
        if os.path.isfile(file_path):
            os.remove(file_path)

        with open(file_path, "wb") as file:
            pickle.dump([], file)
        with open(file_path, "rb") as file:
            self.out = self.out, pickle.load(file)
            os.remove(file_path)
            os.rmdir(directory)

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
        solver = nlpsol("nlpsol", "ipopt", self.nlp, self.opts)

        # Solve the problem
        self.out = solver.call(self.arg)
        self.out["time_tot"] = solver.stats()["t_wall_total"]

        return self.out

    def get_optimized_value(self, ocp):
        return self.out

    @staticmethod
    def __dispatch_bounds(ocp):
        all_J = IpoptInterface.dispatch_obj_func(ocp)
        all_g = ocp.CX()
        all_g_bounds = Bounds(interpolation_type=InterpolationType.CONSTANT)
        for i in range(len(ocp.g)):
            for j in range(len(ocp.g[i])):
                all_g = vertcat(all_g, ocp.g[i][j])
                all_g_bounds.concatenate(ocp.g_bounds[i][j])
        for nlp in ocp.nlp:
            for i in range(len(nlp["g"])):
                for j in range(len(nlp["g"][i])):
                    all_g = vertcat(all_g, nlp["g"][i][j])
                    all_g_bounds.concatenate(nlp["g_bounds"][i][j])

        nlp = {"x": ocp.V, "f": sum1(all_J), "g": all_g}

        var = {
            "lbx": ocp.V_bounds.min,
            "ubx": ocp.V_bounds.max,
            "lbg": all_g_bounds.min,
            "ubg": all_g_bounds.max,
            "x0": ocp.V_init.init,
        }

        return nlp, var

    @staticmethod
    def dispatch_obj_func(ocp):
        all_J = ocp.CX()
        for j_nodes in ocp.J:
            for j in j_nodes:
                all_J = vertcat(all_J, j)
        for nlp in ocp.nlp:
            for obj_nodes in nlp["J"]:
                for obj in obj_nodes:
                    all_J = vertcat(all_J, obj)

        return all_J
