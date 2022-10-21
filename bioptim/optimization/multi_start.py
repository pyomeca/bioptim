from multiprocessing import Pool
from itertools import product

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
    """

    def __init__(
        self,
        solve_ocp,
        n_pools: int = 1,
        args_dict: dict = None,
    ):
        """
        Parameters
        ----------
        solve_ocp: callable
            The function to be called to solve the optimal control problem
        n_random: int
            The number of random initial guess to be used
        n_pools: int
            The number of pools to be used for multi-threading
        args_dict: dict
            The dictionary of arguments to be passed to the solve_ocp function
        """

        self.solve_ocp = solve_ocp
        self.n_pools = n_pools
        self.args_dict = args_dict
        self.combined_args_to_list = self.combine_args_to_list()

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

    def run(self):
        """
        Run the multi-start in the pools for multi-threading
        """
        with Pool(self.n_pools) as p:
            p.map(self.solve_ocp, self.combined_args_to_list)
