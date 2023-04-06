from multiprocessing import Pool
from itertools import product
from typing import Any, Callable

from ..optimization.optimal_control_program import OptimalControlProgram
from ..interfaces.solver_options import Solver
from ..optimization.solution import Solution


class MultiStart:
    """
    The main class to define a multi-start. This class executes the optimal control problems with the possibility to
    vary parameters.

    Methods
    -------
    solve()
        Run the multi-start in the pools for multi-threading
    """

    def __init__(
        self,
        combinatorial_parameters: dict[tuple, ...],
        prepare_ocp_callback: Callable[[Any], OptimalControlProgram],
        post_optimization_callback: Callable[[Solution, Any], None],
        save_folder: str = None,
        should_solve_callback: Callable[[Any], bool] = None,
        solver: Solver = None,
        n_pools: int = 1,
    ):
        """
        Parameters
        ----------
        combinatorial_parameters:
            Dictionary of tuples defined by the user to identify each of the combination that will be solved.
        prepare_ocp_callback: Callable
            The function which is called to prepare the optimal control problem.
            The inputs are a combination of the combinatorial_parameters for this particular ocp.
            The output should be the OptimalControlProgram to solve
        post_optimization_callback: Callable
            The function which is called after the ocp is solved.
            The inputs are the solution and the combination of the combinatorial_parameters.
        should_solve_callback: Callable
            Returns if the current combination of combinatorial_parameters should [True] or not [False] be solved.
            This is to give the opportunity to the user to skip specific combination. If the callback is not defined,
            then all the combination are solved
        solver: Solver
            The solver to use for the ocp. Default is IPOPT
        n_pools: int
            The number of pools to be used for multi-threading. If 1 is sent, then the built-in for loop is used
        """
        # errors : post, prep,
       # if prepare_ocp_callback ==

        self.prepare_ocp_callback = prepare_ocp_callback
        self.post_optimization_callback = post_optimization_callback
        self.should_solve_callback = should_solve_callback
        self.solver = solver if solver else Solver.IPOPT()
        self.n_pools = n_pools
        self.combined_ocp_parameters = self._generate_parameters_combinations(combinatorial_parameters)
        self.save_folder = save_folder

    @staticmethod
    def _generate_parameters_combinations(combinatorial_parameters):
        """
        Combine the varying arguments of the multi-start into a list of arguments to be passed to the solve_ocp function
        inside the pools
        """
        combined_args = [instance for instance in product(*combinatorial_parameters.values())]
        combined_args_to_list = []
        for i in range(len(combined_args)):
            combined_args_to_list += [[instance for instance in combined_args[i]]]
        return combined_args_to_list

    def _prepare_and_solve_ocp(self, ocp_parameters):
        if self.should_solve_callback is None or self.should_solve_callback(ocp_parameters, save_folder = self.save_folder):
            sol = self.prepare_ocp_callback(*ocp_parameters).solve(self.solver)
            self.post_optimization_callback(sol, *ocp_parameters,save_folder= self.save_folder)

    def solve(self):
        """
        Run the multi-start in the pools for multi-threading
        """
        if self.n_pools == 1:
            for ocp_parameters in self.combined_ocp_parameters:
                self._prepare_and_solve_ocp(ocp_parameters)
        else:
            with Pool(self.n_pools) as p:
                p.map(self._prepare_and_solve_ocp, self.combined_ocp_parameters)
