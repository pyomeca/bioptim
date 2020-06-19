import pickle
import casadi
from .plot import OnlineCallback
from .acados_interface import *


class IpoptInterface(SolverInterface):
    def __init__(self, ocp):
        super().__init__()
        self.ocp_solver = None
        self.options_common = {}


    def prepare_ipopt(self, ocp):
        # Dispatch the objective function values and create arg
        self.nlp, self.arg = ocp.dispatch_bounds()

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
        solver = casadi.nlpsol('nlpsol', 'ipopt', self.nlp, self.opts)

        # Solve the problem
        self.out = solver.call(self.arg)
        self.out['time_tot'] = solver.stats()['t_wall_total']

        return self.out

    def get_optimized_value(self, ocp):
        return self.out