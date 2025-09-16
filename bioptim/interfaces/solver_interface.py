from ..misc.parameters_types import (
    Bool,
    AnyDict,
    AnyListorDict,
)


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
    get_optimized_value(self) -> list[dict] | dict
        Get the previously optimized solution
    start_get_iterations(self)
        Create the necessary folder and create the file to store the iterations while optimizing
    finish_get_iterations(self)
        Close the file where iterations are saved and remove temporary folders
    finalize_objective_value(j: dict) -> MX | SX
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

        # This is to perform long preparation only once (if not changed)
        self.pre_shake_tree_objectives = None
        self.shaked_objectives = None
        self.pre_shake_tree_constraints = None
        self.shaked_constraints = None
        self.shaked_ocp_solver = None

    def configure(self, **options):
        """
        Set some options

        Parameters
        ----------
        options: dict
            The dictionary of options
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def solve(self, expand_during_shake_tree: Bool) -> AnyDict:
        """
        Solve the prepared ocp

        Parameters
        ----------
        expand_during_shake_tree: bool
            If the graph should be expanded during the shake tree

        Returns
        -------
        A reference to the solution
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def get_optimized_value(self) -> AnyListorDict:
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
