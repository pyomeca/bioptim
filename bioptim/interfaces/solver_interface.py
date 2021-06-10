from typing import Union
import numpy as np
from casadi import Function, sum1, sum2, MX, SX

from ..misc.enums import Node


class SolverInterface:
    """
    Abstract class for an ocp solver

    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the current OptimalControlProgram
    solver: SolverInterface
        A non-abstract implementation of SolverInterface
    out: dict
        The solution structure

    Methods
    -------
    configure(self, **options)
        Set some options
    solve(self) -> dict
        Solve the prepared ocp
    get_optimized_value(self) -> Union[list[dict], dict]
        Get the previously optimized solution
    start_get_iterations(self)
        Create the necessary folder and create the file to store the iterations while optimizing
    finish_get_iterations(self)
        Close the file where iterations are saved and remove temporary folders
    get_objectives(self)
        Retrieve the objective values and put them in the out dict
    finalize_objective_value(j: dict) -> Union[MX, SX]
        Apply weight and dt to all objective values and convert them to scalar value
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
        A reference to the current OptimalControlProgram
        """

        self.ocp = ocp
        self.solver = None
        self.out = {}

    def configure(self, **options):
        """
        Set some options

        Parameters
        ----------
        options: dict
            The dictionary of options
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def solve(self) -> dict:
        """
        Solve the prepared ocp

        Returns
        -------
        A reference to the solution
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def get_optimized_value(self) -> Union[list, dict]:
        """
        Get the previously optimized solution

        Returns
        -------
        A solution or a list of solution depending on the number of phases
        """

        out = []
        for key in self.out.keys():
            out.append(self.out[key])
        return out[0] if len(out) == 1 else out

    def online_optim(self, ocp):
        """
        Declare the online callback to update the graphs while optimizing

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def start_get_iterations(self):
        """
        Create the necessary folder and create the file to store the iterations while optimizing
        """

        raise RuntimeError("Get Iteration not implemented for solver")

    def finish_get_iterations(self):
        """
        Close the file where iterations are saved and remove temporary folders
        """

        raise RuntimeError("Get Iteration not implemented for solver")

    def get_objectives(self):
        """
        Retrieve the objective values and put them in the out dict
        """

        def get_objective_values(ocp, sol: dict) -> list:
            """
            Get all the objective values

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the current OptimalControlProgram
            sol: dict
                A solution structure where the lagrange multipliers are set

            Returns
            -------
            A list of objective values
            """

            def __get_nodes(all_nodes: list, nlp) -> list:
                """
                Get the objective values of a specific nlp

                Parameters
                ----------
                all_nodes: list
                    All the node of an nlp
                nlp: NonLinearProgram
                    The current nlp

                Returns
                -------
                A list of objective values
                """
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

                    # elif node == Node.TRANSITION:
                    #     nodes.append(nlp.ns - 1)

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
    def finalize_objective_value(j: dict) -> Union[MX, SX]:
        """
        Apply weight and dt to all objective values and convert them to scalar value

        Parameters
        ----------
        j: dict
            The dictionary of all the objective functions

        Returns
        -------
        Scalar values of the objective functions weighted
        """

        val = j["val"]
        if j["target"] is not None:
            nan_idx = np.isnan(j["target"])
            j["target"][nan_idx] = 0
            val -= j["target"]
            if np.any(nan_idx):
                val[np.where(nan_idx)] = 0

        if j["objective"].quadratic:
            val = val ** 2
        return sum1(sum2(j["objective"].weight * val * j["dt"]))
