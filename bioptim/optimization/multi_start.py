from multiprocessing import Pool

class MultiStart():
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
        n_random: int = 1,
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
        self.n_random = n_random
        self.n_pools = n_pools
        self.args_dict = args_dict
        self.combined_args_to_list = self.combine_args_to_list()


    def combine_args_to_list(self):
        """
        Combine the varying arguments of the multi-start into a list of arguments to be passed to the solve_ocp function
        inside the pools
        """
        args_names_list = list(self.args_dict.keys())
        if len(args_names_list) == 0:
            list_out = [[]]
        else:
            list_out = [[elem] for elem in self.args_dict[args_names_list[0]]]
            for i in range(1, len(args_names_list)):
                for j in range(len(list_out)):
                    list_out_before = list_out
                    list_out_temporary = list_out
                    for k in range(len(self.args_dict[args_names_list[i]])):
                        list_now = [list_out_temporary[l] + [self.args_dict[args_names_list[i]][k]] for l in range(len(list_out_before))]
                        if k == 0:
                            list_out = list_now
                        else:
                            list_out += list_now

        combined_args_to_list = []
        for i in range(self.n_random):
            combined_args_to_list += [list_out[j] + [i] for j in range(len(list_out))]

        return combined_args_to_list

    def run(self):
        """
        Run the multi-start in the pools for multi-threading
        """
        with Pool(self.n_pools) as p:
            p.map(self.solve_ocp, self.combined_args_to_list)
