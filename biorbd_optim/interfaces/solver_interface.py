class SolverInterface:
    def __init__(self):
        self.solver = None

    def configure(self, **options):
        raise RuntimeError("SolverInterface is an abstract class")

    def solve(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_iterations(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_optimized_value(self, ocp):
        raise RuntimeError("SolverInterface is an abstract class")

    def online_optim(self, ocp):
        raise RuntimeError("SolverInterface is an abstract class")
