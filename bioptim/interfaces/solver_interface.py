from ..limits.objective_functions import get_objective_values


class SolverInterface:
    def __init__(self, ocp):
        self.ocp = ocp
        self.solver = None
        self.out = {}

    def configure(self, **options):
        raise RuntimeError("SolverInterface is an abstract class")

    def solve(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_iterations(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_optimized_value(self):
        out = []
        for key in self.out.keys():
            out.append(self.out[key])
        return out[0] if len(out) == 1 else out

    def online_optim(self, ocp):
        raise RuntimeError("SolverInterface is an abstract class")

    def start_get_iterations(self):
        raise RuntimeError("Get Iteration not implemented for solver")

    def finish_get_iterations(self):
        raise RuntimeError("Get Iteration not implemented for solver")

    def get_objective(self):
        self.out["sol_obj"] = get_objective_values(self.ocp, self.out["sol"])
