import os
import biorbd
from casadi import MX, Function, SX, vertcat
import numpy as np
import casadi
from .solver_interface import SolverInterface
from .path_conditions import Bounds, InterpolationType


class IpoptInterface(SolverInterface):
    def __init__(self, ocp):
        super().__init__()
        self.ocp_solver = None

    def prepare_ipopt(self, ocp):
        # Dispatch the objective function values
        all_J = SX()
        for j_nodes in ocp.J:
            for j in j_nodes:
                all_J = vertcat(all_J, j)
        for nlp in ocp.nlp:
            for obj_nodes in nlp["J"]:
                for obj in obj_nodes:
                    all_J = vertcat(all_J, obj)

        #
        all_g = SX()
        all_g_bounds = Bounds(interpolation_type=InterpolationType.CONSTANT)
        for i in range(len(ocp.g)):
            for j in range(len(ocp.g[i])):
                all_g = vertcat(all_g, ocp.g[i][j])
                all_g_bounds.concatenate(ocp.g_bounds[i][j])
        for nlp in ocp.nlp:
            for i in range(len(nlp["g"])):
                for j in range(len(ocp.nlp["g"][i])):
                    all_g = vertcat(all_g, ocp.nlp["g"][i][j])
                    all_g_bounds.concatenate(ocp.nlp["g_bounds"][i][j])

        self.nlp = {"x": nlp.V, "f": ocp.sum1(all_J), "g": all_g}

        # Bounds and initial guess
        self.arg = {
            "lbx": ocp.V_bounds.min,
            "ubx": ocp.V_bounds.max,
            "lbg": all_g_bounds.min,
            "ubg": all_g_bounds.max,
            "x0": ocp.V_init.init,
        }


    def configure(self, solver_options, options_common):
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
        self.opts = {**options, **options_common}

    def get_iterations(self):
        pass

    def get_optimized_value(self):
        return self.out

    def solve(self):
        solver = casadi.nlpsol('nlpsol', 'ipopt', self.nlp, self.opts)

        # Solve the problem
        self.out = solver.call(self.arg)
        self.out['time_tot'] = solver.stats()['t_wall_total']
        # if return_iterations:
        #     with open(file_path, "rb") as file:
        #         self.out = self.out, pickle.load(file)
        #         os.remove(file_path)
        #         os.rmdir(directory)
        return self.out
