import numpy as np
import casadi

from ..misc.enums import Instant
from ..limits.objective_functions import get_objective_value


class SolverInterface:
    def __init__(self, ocp):
        self.ocp = ocp
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

    def start_get_iterations(self):
        raise RuntimeError("Get Iteration not implemented for solver")

    def finish_get_iterations(self):
        raise RuntimeError("Get Iteration not implemented for solver")

    def get_objective_values(self):
        def __get_instant(instants, nlp):
            nodes = []
            for node in instants:
                if isinstance(node, int):
                    if node < 0 or node > nlp["ns"]:
                        raise RuntimeError(f"Invalid instant, {node} must be between 0 and {nlp['ns']}")
                    nodes.append(node)

                elif node == Instant.START:
                    nodes.append(0)

                elif node == Instant.MID:
                    if nlp["ns"] % 2 == 1:
                        raise (ValueError("Number of shooting points must be even to use MID"))
                    nodes.append(nlp["ns"] // 2)

                elif node == Instant.INTERMEDIATES:
                    for i in range(1, nlp["ns"] - 1):
                        nodes.append(i)

                elif node == Instant.END:
                    nodes.append(nlp["ns"] - 1)

                elif node == Instant.ALL:
                    for i in range(nlp["ns"]):
                        nodes.append(i)
            return nodes

        sol = self.out["sol"]["x"]
        out = []
        for idx_phase, nlp in enumerate(self.ocp.nlp):
            nJ = len(nlp["J"]) - idx_phase
            out.append(np.ndarray((nJ, nlp["ns"])))
            out[-1][:][:] = np.nan
            for idx_obj_func in range(nJ):
                nodes = __get_instant(nlp["J"][idx_phase + idx_obj_func][0]["objective"].instant, nlp)
                nodes = nodes[: len(nlp["J"][idx_phase + idx_obj_func])]
                for node, idx_node in enumerate(nodes):
                    obj = casadi.Function(
                        "obj", [self.ocp.V], [get_objective_value(nlp["J"][idx_phase + idx_obj_func][node])]
                    )
                    out[-1][idx_obj_func][idx_node] = obj(sol)
        self.out["sol_obj"] = out
