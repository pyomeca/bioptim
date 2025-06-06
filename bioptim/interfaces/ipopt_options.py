from dataclasses import dataclass
from typing import Any

from ..misc.enums import SolverType, OnlineOptim
from .abstract_options import GenericSolver


from ..misc.parameters_types import (
    Bool,
    BoolOptional,
    Int,
    Float,
    Str,
    AnyDictOptional,
)


@dataclass
class IPOPT(GenericSolver):
    """
    Class for Solver Options of IPOPT

    Attributes
    ----------
    show_online_optim: bool | None
        If the plot should be shown while optimizing. If set to True, it will the default online_optim. online_optim
        and show_online_optim cannot be simultaneous set
    online_optim: OnlineOptim
        The type of online plot to show. If set to None (default), then no plot will be shown. If set to DEFAULT, it
        will use the fastest method for your OS (multiprocessing on Linux and multiprocessing_server on Windows).
        In all cases, it will slow down the optimization a bit.
    show_options: dict
        The graphs option to pass to PlotOcp
    _tol: float
        Desired convergence tolerance (relative)
    _dual_inf_tol: float
        Desired threshold for the dual infeasibility
    _constr_viol_tol: float
        Desired threshold for the constraint and variable bound violation.
    _compl_inf_tol: float
        Desired threshold for the complementarity conditions.
    _acceptable_tol: float
        Acceptable convergence tolerance (relative).
    _acceptable_dual_inf_tol: float
        Acceptance threshold for the dual infeasibility
    _acceptable_constr_viol_tol: float
        Acceptance threshold for the constraint violation.
    _acceptable_compl_inf_tol: float
        "Acceptance" threshold for the complementarity conditions.
    _max_iter: int
        Maximum number of iterations.
    _hessian_approximation: str
        Indicates what Hessian information is to be used.
    _nlp_scaling_method: str
        Indicates the method used by IPOPT to scale the nlp
    _limited_memory_max_history: int
        Maximum size of the history for the limited quasi-Newton Hessian approximation.
    _linear_solver: str
        Linear solver used for step computations.
    _mu_init: float
        Initial value for the barrier parameter.
    _warm_start_init_point: float
        Warm-start for initial point
    _warm_start_mult_bound_push: float
        same as mult_bound_push for the regular initializer
    _warm_start_slack_bound_push: float
        same as slack_bound_push for the regular initializer
    _warm_start_slack_bound_frac: float
        same as slack_bound_frac for the regular initializer
    _warm_start_bound_frac: float
        same as bound_frac for the regular initializer
    _bound_push: float
        Desired minimum absolute distance from the initial point to bound.
    _bound_frac: float
        Desired minimum relative distance from the initial point to bound.
    _print_level: float
        Output verbosity level. Sets the default verbosity level for console output.
        The larger this value the more detailed is the output.
        The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.
    _c_compile: bool
        True if you want to compile in C the code.
    _check_derivatives_for_naninf: bool
        If true, the Hessian will be checked for nan/inf values. If false this computational problem is silent.
    """

    type: SolverType = SolverType.IPOPT
    show_online_optim: BoolOptional = None
    online_optim: OnlineOptim | None = None
    show_options: AnyDictOptional = None
    _tol: Float = 1e-6  # default in ipopt 1e-8
    _dual_inf_tol: Float = 1.0
    _constr_viol_tol: Float = 0.0001
    _compl_inf_tol: Float = 0.0001
    _acceptable_tol: Float = 1e-6
    _acceptable_dual_inf_tol: Float = 1e10
    _acceptable_constr_viol_tol: Float = 1e-2
    _acceptable_compl_inf_tol: Float = 1e-2
    _max_iter: Int = 1000
    _hessian_approximation: Str = "exact"  # "exact", "limited-memory"
    _nlp_scaling_method: Str = "gradient-based"  # "none"
    _limited_memory_max_history: Int = 50
    _linear_solver: Str = "mumps"  # "ma57", "ma86", "mumps"
    _mu_init: Float = 0.1
    _warm_start_init_point: Str = "no"
    _warm_start_mult_bound_push: Float = 0.001
    _warm_start_slack_bound_push: Float = 0.001
    _warm_start_bound_push: Float = 0.001
    _warm_start_slack_bound_frac: Float = 0.001
    _warm_start_bound_frac: Float = 0.001
    _bound_push: Float = 0.01
    _bound_frac: Float = 0.01
    _print_level: Int = 5
    _c_compile: Bool = False
    _check_derivatives_for_naninf: Str = "no"  # "yes"

    @property
    def tol(self) -> Float:
        return self._tol

    @property
    def dual_inf_tol(self) -> Float:
        return self._dual_inf_tol

    @property
    def constr_viol_tol(self) -> Float:
        return self._constr_viol_tol

    @property
    def compl_inf_tol(self) -> Float:
        return self._compl_inf_tol

    @property
    def acceptable_tol(self) -> Float:
        return self._acceptable_tol

    @property
    def acceptable_dual_inf_tol(self) -> Float:
        return self._acceptable_dual_inf_tol

    @property
    def acceptable_constr_viol_tol(self) -> Float:
        return self._acceptable_constr_viol_tol

    @property
    def acceptable_compl_inf_tol(self) -> Float:
        return self._acceptable_compl_inf_tol

    @property
    def max_iter(self) -> Int:
        return self._max_iter

    @property
    def hessian_approximation(self) -> Str:
        return self._hessian_approximation

    @property
    def nlp_scaling_method(self) -> Str:
        return self._nlp_scaling_method

    @property
    def limited_memory_max_history(self) -> Int:
        return self._limited_memory_max_history

    @property
    def linear_solver(self) -> Str:
        return self._linear_solver

    @property
    def mu_init(self) -> Float:
        return self._mu_init

    @property
    def warm_start_init_point(self) -> Str:
        return self._warm_start_init_point

    @property
    def warm_start_mult_bound_push(self) -> Float:
        return self._warm_start_mult_bound_push

    @property
    def warm_start_slack_bound_push(self) -> Float:
        return self._warm_start_slack_bound_push

    @property
    def warm_start_bound_push(self) -> Float:
        return self._warm_start_bound_push

    @property
    def warm_start_slack_bound_frac(self) -> Float:
        return self._warm_start_slack_bound_frac

    @property
    def warm_start_bound_frac(self) -> Float:
        return self._warm_start_bound_frac

    @property
    def bound_push(self) -> Float:
        return self._bound_push

    @property
    def bound_frac(self) -> Float:
        return self._bound_frac

    @property
    def print_level(self) -> Int:
        return self._print_level

    @property
    def c_compile(self) -> Bool:
        return self._c_compile

    @property
    def check_derivatives_for_naninf(self) -> Bool:
        return self._check_derivatives_for_naninf

    def set_tol(self, val: Float) -> None:
        self._tol = val

    def set_dual_inf_tol(self, val: Float) -> None:
        self._dual_inf_tol = val

    def set_constr_viol_tol(self, val: Float) -> None:
        self._constr_viol_tol = val

    def set_compl_inf_tol(self, val: Float) -> None:
        self._compl_inf_tol = val

    def set_acceptable_tol(self, val: Float) -> None:
        self._acceptable_tol = val

    def set_acceptable_dual_inf_tol(self, val: Float) -> None:
        self._acceptable_dual_inf_tol = val

    def set_acceptable_constr_viol_tol(self, val: Float) -> None:
        self._acceptable_constr_viol_tol = val

    def set_acceptable_compl_inf_tol(self, val: Float) -> None:
        self._acceptable_compl_inf_tol = val

    def set_maximum_iterations(self, num: Int) -> None:
        self._max_iter = num

    def set_hessian_approximation(self, val: Str) -> None:
        self._hessian_approximation = val

    def set_nlp_scaling_method(self, val: Str) -> None:
        self._nlp_scaling_method = val

    def set_limited_memory_max_history(self, num: Int) -> None:
        self._limited_memory_max_history = num

    def set_linear_solver(self, val: Str) -> None:
        self._linear_solver = val

    def set_mu_init(self, val: Float) -> None:
        self._mu_init = val

    def set_warm_start_init_point(self, val: Str) -> None:
        self._warm_start_init_point = val

    def set_warm_start_mult_bound_push(self, val: Float) -> None:
        self._warm_start_mult_bound_push = val

    def set_warm_start_slack_bound_push(self, val: Float) -> None:
        self._warm_start_slack_bound_push = val

    def set_warm_start_bound_push(self, val: Float) -> None:
        self._warm_start_bound_push = val

    def set_warm_start_slack_bound_frac(self, val: Float) -> None:
        self._warm_start_slack_bound_frac = val

    def set_warm_start_bound_frac(self, val: Float) -> None:
        self._warm_start_bound_frac = val

    def set_bound_push(self, val: Float) -> None:
        self._bound_push = val

    def set_bound_frac(self, val: Float) -> None:
        self._bound_frac = val

    def set_print_level(self, num: Int) -> None:
        self._print_level = num

    def set_c_compile(self, val: Bool) -> None:
        self._c_compile = val

    def set_check_derivatives_for_naninf(self, val: Bool) -> None:
        string_val = "yes" if val else "no"
        self._check_derivatives_for_naninf = string_val

    def set_convergence_tolerance(self, val: Float) -> None:
        self._tol = val
        self._compl_inf_tol = val
        self._acceptable_tol = val
        self._acceptable_compl_inf_tol = val

    def set_constraint_tolerance(self, val: Float) -> None:
        self._constr_viol_tol = val
        self._acceptable_constr_viol_tol = val

    def set_warm_start_options(self, val: Float = 1e-10) -> None:
        """
        This function set global warm start options

        Parameters
        ----------
        val: float
            warm start value
        """

        self._warm_start_init_point = "yes"
        self._mu_init = val
        self._warm_start_mult_bound_push = val
        self._warm_start_slack_bound_push = val
        self._warm_start_bound_push = val
        self._warm_start_slack_bound_frac = val
        self._warm_start_bound_frac = val

    def set_initialization_options(self, val: Float) -> None:
        """
        This function set global initialization options

        Parameters
        ----------
        val: float
            warm start value
        """
        self._bound_push = val
        self._bound_frac = val

    def set_option_unsafe(self, val: Any, name: Str) -> None:
        """
        This function is unsafe because we did not check if the option exist in the solver option list.
        If it's not it just will be ignored. Please make sure that the option you're asking for exist.
        """
        if f"_{name}" not in self.__dict__.keys():
            self.__dict__[f"_{name}"] = val

    def as_dict(self, solver):
        solver_options = self.__dict__
        options = {}
        non_python_options = ["_c_compile", "type", "show_online_optim", "online_optim", "show_options"]
        for key in solver_options:
            if key not in non_python_options:
                ipopt_key = "ipopt." + key[1:]
                options[ipopt_key] = solver_options[key]
        return {**options, **solver.options_common}
