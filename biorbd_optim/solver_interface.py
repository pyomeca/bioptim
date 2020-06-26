class SolverInterface:
    def __init__(self):
        self.solver = None

    def configure(self, **options):
        raise RuntimeError("SolverInterface is an abstract class")

    def solve(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_iterations(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_optimized_value(self):
        raise RuntimeError("SolverInterface is an abstract class")
