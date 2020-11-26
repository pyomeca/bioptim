import numpy as np
from casadi import Function, sum1, sum2

from ..misc.enums import Node


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

    def get_objectives(self):
        def get_objective_values(ocp, sol):
            def __get_nodes(all_nodes, nlp):
                nodes = []
                for node in all_nodes:
                    if isinstance(node, int):
                        if node < 0 or node > nlp.ns:
                            raise RuntimeError(f"Invalid node, {node} must be between 0 and {nlp.ns}")
                        nodes.append(node)

                    elif node == Node.START:
                        nodes.append(0)

                    elif node == Node.MID:
                        if nlp.ns % 2 == 1:
                            raise (ValueError("Number of shooting points must be even to use MID"))
                        nodes.append(nlp.ns // 2)

                    elif node == Node.INTERMEDIATES:
                        for i in range(1, nlp.ns - 1):
                            nodes.append(i)

                    elif node == Node.END:
                        nodes.append(nlp.ns - 1)

                    elif node == Node.ALL:
                        for i in range(nlp.ns):
                            nodes.append(i)
                return nodes

            sol = sol["x"]
            out = []
            for nlp in ocp.nlp:
                nJ = len(nlp.J)
                out.append(np.ndarray((nJ, nlp.ns)))
                out[-1][:, :] = np.nan
                for idx_obj_func in range(nJ):
                    nodes = __get_nodes(nlp.J[idx_obj_func][0]["objective"].node, nlp)
                    nodes = nodes[: len(nlp.J[idx_obj_func])]
                    for node, idx_node in enumerate(nodes):
                        obj = Function("obj", [ocp.V], [self.finalize_objective_value(nlp.J[idx_obj_func][node])])
                        out[-1][idx_obj_func, idx_node] = obj(sol)
            return out

        self.out["sol_obj"] = get_objective_values(self.ocp, self.out["sol"])

    @staticmethod
    def finalize_objective_value(j_dict):
        val = j_dict["val"]
        if j_dict["target"] is not None:
            nan_idx = np.isnan(j_dict["target"])
            j_dict["target"][nan_idx] = 0
            val -= j_dict["target"]
            if np.any(nan_idx):
                val[np.where(nan_idx)] = 0

        if j_dict["objective"].quadratic:
            val = val ** 2
        return sum1(sum2(j_dict["objective"].weight * val * j_dict["dt"]))
