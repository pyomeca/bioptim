from dataclasses import dataclass

from ..misc.enums import SolverType
from .abstract_options import GenericSolver


@dataclass
class SQP_METHOD(GenericSolver):
    """
    This class is used to set the SQP method options

    Methods
    -------
    set_beta(beta: float):
        Line-search parameter, restoration factor of stepsize
    set_c1(c1: float):
        Armijo condition, coefficient of decrease in merit
    set_hessian_approximation(hessian_approximation: str):
        Hessian approximation method
    set_nlp_scaling_method(scaling_method: str):
        Method used to scale the NLP
    set_lbfgs_memory(lbfgs_memory: int):
        Size of L-BFGS memory.
    set_maximum_iterations(max_iter: int):
        Maximum number of SQP iterations
    set_max_iter_ls(max_iter_ls: int):
        Maximum number of linesearch iterations
    set_merit_memory(merit_memory: int):
        Size of memory to store history of merit function values
    set_print_header(print_header: bool):
        Print the header with problem statistics
    set_print_time(print_time: bool):
        Print information about execution time
    set_qpsol(qpsol: str):
        The QP solver to be used by the SQP method
    set_tol_du(tol_du: float):
        Stopping criterion for dual infeasability
    set_tol_pr(tol_pr: float):
        Stopping criterion for primal infeasibility
    set_set_option_unsafe(val, name):
        Seting an option that is not in the list of options
    set_convergence_tolerance(tol: float):
        Set the convergence tolerance NA
    set_constraint_tolerance(tol: float):
        Set the constraint tolerance NA
    as_dict(solver):
        Return the options as a dictionary
    set_print_level(num: int):
        Set the print level of the solver NA

    Attributes
    ----------
    _beta: float
    _c1: float
    _hessian_approximation: str
    _lbfgs_memory: int
    _max_iter: int
    _max_iter_ls: int
    _merit_memory: int
    _print_header: bool
    _print_time: bool
    _qpsol: str
    _tol_du: float
    _tol_pr: float

    """

    type: SolverType = SolverType.SQP
    show_online_optim: bool = False
    show_options: dict = None
    c_compile = False
    _beta: float = 0.8
    _c1: float = 1e-4
    _hessian_approximation: str = "exact"  # "exact", "limited-memory"
    _lbfgs_memory: int = 10
    _max_iter: int = 50
    _max_iter_ls: int = 3
    _merit_memory: int = 4
    _print_header: bool = True
    _print_time: bool = True
    _qpsol: str = "qpoases"
    _tol_du: float = 1e-6
    _tol_pr: float = 1e-6

    @property
    def beta(self):
        return self._beta

    @property
    def c1(self):
        return self._c1

    @property
    def hessian_approximation(self):
        return self._hessian_approximation

    @property
    def lbfgs_memory(self):
        return self._lbfgs_memory

    @property
    def maximum_iterations(self):
        return self._max_iter

    @property
    def max_iter_ls(self):
        return self._max_iter_ls

    @property
    def merit_memory(self):
        return self._merit_memory

    @property
    def print_header(self):
        return self._print_header

    @property
    def print_time(self):
        return self._print_time

    @property
    def qpsol(self):
        return self._qpsol

    @property
    def tol_du(self):
        return self._tol_du

    @property
    def tol_pr(self):
        return self._tol_pr

    def set_beta(self, beta: float):
        """
        Line-search parameter, restoration factor of stepsize
        """
        self._beta = beta

    def set_c1(self, c1: float):
        """
        Armijo condition, coefficient of decrease in merit
        """
        self._c1 = c1

    def set_hessian_approximation(self, hessian_approximation: str):
        self._hessian_approximation = hessian_approximation

    def set_nlp_scaling_method(self, nlp_scaling_metod: str):
        self._nlp_scaling_metod = nlp_scaling_metod

    def set_lbfgs_memory(self, lbfgs_memory: int):
        """
        Size of L-BFGS memory.
        """
        self._lbfgs_memory = lbfgs_memory

    def set_maximum_iterations(self, max_iter: int):
        """
        Maximum number of SQP iterations
        """
        self._max_iter = max_iter

    def set_max_iter_ls(self, max_iter_ls: int):
        """
        Maximum number of linesearch iterations
        """
        self._max_iter_ls = max_iter_ls

    def set_merit_memory(self, merit_memory: int):
        """
        Size of memory to store history of merit function values
        """
        self._merit_memory = merit_memory

    def set_print_header(self, print_header: bool):
        """
        Print the header with problem statistics
        """
        self._print_header = print_header

    def set_print_time(self, print_time: bool):
        """
        Print information about execution time
        """
        self._print_time = print_time

    def set_qp_solver(self, qpsol: str):
        """
        The QP solver to be used by the SQP method
        """
        self._qpsol = qpsol

    def set_tol_du(self, tol_du: float):
        """
        Stopping criterion for dual infeasability
        """
        self._tol_du = tol_du

    def set_tol_pr(self, tol_pr: float):
        """
        Stopping criterion for primal infeasibility
        """
        self._tol_pr = tol_pr

    def set_set_option_unsafe(self, val, name):
        """
        This function is unsafe because we did not check if the option exist in the solver option list.
        If it's not it just will be ignored. Please make sure that the option you're asking for exist.
        """
        if f"_{name}" not in self.__dict__.keys():
            self.__dict__[f"_{name}"] = val

    def set_convergence_tolerance(self, tol: float):
        raise RuntimeError(
            "At the moment, set_convergence_tolerance cannot be set for SQP method solver."
            "\nPlease use set_tol_du to set the tolerance on the dual condition and set_tol_pr to set"
            " the tolerance on the primal condition"
        )

    def set_constraint_tolerance(self, tol: float):
        raise RuntimeError(
            "At the moment, set_constraint_tolerance cannot be set for SQP method solver."
            "\nPlease use set_tol_du to set the tolerance on the dual condition and set_tol_pr to set"
            " the tolerance on the primal condition"
        )

    def as_dict(self, solver):
        solver_options = self.__dict__
        options = {}
        non_python_options = ["type", "show_online_optim", "show_options"]
        for key in solver_options:
            if key not in non_python_options:
                sqp_key = key[1:]
                options[sqp_key] = solver_options[key]
        return {**options, **solver.options_common}

    def set_print_level(self, num: int):
        raise RuntimeError("At the moment, set_print_level cannot be set for SQP method solver")
