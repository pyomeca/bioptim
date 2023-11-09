import numpy as np

from .interface_utils import (
    generic_online_optim,
    generic_solve,
    generic_dispatch_bounds,
    generic_dispatch_obj_func,
    generic_get_all_penalties,
)
from .solver_interface import SolverInterface
from ..interfaces import Solver
from bioptim.optimization.solution.solution import Solution
from ..optimization.non_linear_program import NonLinearProgram
from ..misc.enums import (
    SolverType,
)


class SQPInterface(SolverInterface):
    """
    The SQP method solver interface

    Attributes
    ----------
    options_common: dict
        Options irrelevant of a specific ocp
    opts: SQP
        Options of the current ocp
    sqp_nlp: dict
        The declaration of the variables SQP-friendly
    sqp_limits: dict
        The declaration of the bound SQP-friendly
    lam_g: np.ndarray
        The lagrange multiplier of the constraints to initialize the solver
    lam_x: np.ndarray
        The lagrange multiplier of the variables to initialize the solver

    Methods
    -------
    online_optim(self, ocp: OptimalControlProgram)
        Declare the online callback to update the graphs while optimizing
    solve(self) -> dict
        Solve the prepared ocp
    set_lagrange_multiplier(self, sol: dict)
        Set the lagrange multiplier from a solution structure
    __dispatch_bounds(self)
        Parse the bounds of the full ocp to a SQP-friendly one
    __dispatch_obj_func(self)
        Parse the objective functions of the full ocp to a SQP-friendly one
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        super().__init__(ocp)

        self.options_common = {}
        self.opts = Solver.SQP_METHOD()
        self.solver_name = SolverType.SQP.value

        self.sqp_nlp = {}
        self.sqp_limits = {}
        self.ocp_solver = None
        self.c_compile = False

        self.lam_g = None
        self.lam_x = None

    def online_optim(self, ocp, show_options: dict = None):
        """
        Declare the online callback to update the graphs while optimizing

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        show_options: dict
            The options to pass to PlotOcp
        """
        generic_online_optim(self, ocp, show_options)

    def solve(self) -> dict:
        """
        Solve the prepared ocp

        Returns
        -------
        A reference to the solution
        """
        return generic_solve(self)

    def set_lagrange_multiplier(self, sol: Solution):
        """
        Set the lagrange multiplier from a solution structure

        Parameters
        ----------
        sol: dict
            A solution structure where the lagrange multipliers are set
        """
        raise NotImplementedError("This is broken")
        # generic_set_lagrange_multiplier(self, sol)

    def dispatch_bounds(self):
        """
        Parse the bounds of the full ocp to a SQP-friendly one
        """
        return generic_dispatch_bounds(self)

    def dispatch_obj_func(self):
        """
        Parse the objective functions of the full ocp to a SQP-friendly one

        Returns
        -------
        SX | MX
            The objective function
        """
        return generic_dispatch_obj_func(self)

    def get_all_penalties(self, nlp: NonLinearProgram, penalties):
        """
        Parse the penalties of the full ocp to a SQP-friendly one

        Parameters
        ----------
        nlp: NonLinearProgram
            The nonlinear program to parse the penalties from
        penalties:
            The penalties to parse
        Returns
        -------

        """
        return generic_get_all_penalties(self, nlp, penalties, is_unscaled=False)
