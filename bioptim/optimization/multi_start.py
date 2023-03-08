from multiprocessing import Pool
from itertools import product
from typing import Callable

from ..optimization.optimal_control_program import OptimalControlProgram
from ..interfaces.solver_options import Solver
import os

class MultiStart:
    """
    The main class to define a multi-start. This class executes the optimal control problems with the possibility to
    vary parameters.

    Methods
    -------
    combine_args_to_list()
        Combine the varying arguments of the multi-start into a list of arguments to be passed to the solve_ocp function
        inside the pools
    run()
        Run the multi-start in the pools for multi-threading
    check already_done()
        Check if the OCP has already a saved solution
    """

    def __init__(
        self,
        prepare_ocp: OptimalControlProgram,
        solver: Solver,
        post_optimization_callback: Callable = None,
        n_pools: int = 1,
        already_done_filenames: list = None,
        single_process_debug_flag: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        prepare_ocp: callable
            The function to be called to prepare the optimal control problem
        solver: Solver
            The solver to use for the ocp
        post_optimization_callback:
            The function to call when the ocp is solved
        n_pools: int
            The number of pools to be used for multi-threading
        already_done_filenames:
            List of all the filenames that are in the designated folder
        single_process_debug_flag:
            True if you want to use the for loop to debug
        args_dict: dict
            The dictionary of arguments to be passed to the solve_ocp function
        combined_args_to_list :
            ??
        """

        self.prepare_ocp = prepare_ocp
        self.solver = solver
        self.post_optimization_callback = post_optimization_callback
        self.n_pools = n_pools
        self.already_done_filenames = already_done_filenames
        self.single_process_debug_flag = single_process_debug_flag
        self.args_dict = kwargs
        self.combined_args_to_list = self.combine_args_to_list()


    def check_already_done(self, args):
        """
        Check if the filename already appears in the folder where files are saved
        """
        return self.post_optimization_callback([None], *args, only_save_filename=True) in self.already_done_filenames

    def combine_args_to_list(self):
        """
        Combine the varying arguments of the multi-start into a list of arguments to be passed to the solve_ocp function
        inside the pools
        """
        combined_args = [instance for instance in product(*self.args_dict.values())]
        combined_args_to_list = []

        for i in range(len(combined_args)):
            combined_args_to_list += [[instance for instance in combined_args[i]]]
        return combined_args_to_list

    def solve_ocp_func(self, args):
        check_already_done = self.check_already_done(args)
        if not check_already_done:
            sol = self.prepare_ocp(*args).solve(self.solver)
            self.post_optimization_callback(sol, *args)
        else:
            print("This OCP already has a saved solution in the designated folder.")

    def solve(self):
        """
        Run the multi-start in the pools for multi-threading
        """
        single_process_debug_flag = self.single_process_debug_flag
        if single_process_debug_flag == True :
            for args in self.combined_args_to_list:
                self.solve_ocp_func(args)
        else:
            with Pool(self.n_pools) as p:
                p.map(self.solve_ocp_func, self.combined_args_to_list)
