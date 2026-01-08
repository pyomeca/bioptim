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
class FATROP(GenericSolver):
    """
    Class for Solver Options of FATROP

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
    _structure_detection: str
        If the structure of the problem should be detected automatically ("auto") [default] or manually ("manual").
    _tol: float
        Desired convergence tolerance (relative)
    _constr_viol_tol: float
        Desired threshold for the constraint and variable bound violation.
    _acceptable_tol: float
        Acceptable convergence tolerance (relative).
    _max_iter: int
        Maximum number of iterations.
    _mu_init: float
        Initial value for the barrier parameter.
    _warm_start_init_point: float
        Warm-start for initial point
    _warm_start_mult_bound_push: float
        same as mult_bound_push for the regular initializer
    _bound_push: float
        Desired minimum absolute distance from the initial point to bound.
    _print_level: float
        Output verbosity level. Sets the default verbosity level for console output.
        The larger this value the more detailed is the output.
        The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.
    _c_compile: bool
        True if you want to compile in C the code.
    """

    type: SolverType = SolverType.FATROP
    show_online_optim: BoolOptional = None
    online_optim: OnlineOptim | None = None
    show_options: AnyDictOptional = None
    _tol: Float = 1e-6
    _structure_detection: Str = "auto"  # "auto", "manual"
    _constr_viol_tol: Float = 0.0001
    _acceptable_tol: Float = 1e-6
    _max_iter: Int = 1000
    _mu_init: Float = 0.1
    _warm_start_init_point: bool = False
    _warm_start_mult_bound_push: Float = 0.001
    _bound_push: Float = 0.01
    _print_level: Int = 5
    _c_compile: Bool = False

    def __post_init__(self):
        if self.online_optim == OnlineOptim.DEFAULT:
            self.online_optim = None

    @property
    def tol(self) -> Float:
        return self._tol

    @property
    def constr_viol_tol(self) -> Float:
        return self._constr_viol_tol

    @property
    def acceptable_tol(self) -> Float:
        return self._acceptable_tol

    @property
    def max_iter(self) -> Int:
        return self._max_iter

    @property
    def mu_init(self) -> Float:
        return self._mu_init

    @property
    def warm_start_init_point(self) -> Str:
        return "yes" if self._warm_start_init_point else "no"

    @property
    def warm_start_mult_bound_push(self) -> Float:
        return self._warm_start_mult_bound_push

    @property
    def bound_push(self) -> Float:
        return self._bound_push

    @property
    def print_level(self) -> Int:
        return self._print_level

    @property
    def c_compile(self) -> Bool:
        return self._c_compile

    def set_tol(self, val: Float) -> None:
        self._tol = val

    def set_constr_viol_tol(self, val: Float) -> None:
        self._constr_viol_tol = val

    def set_acceptable_tol(self, val: Float) -> None:
        self._acceptable_tol = val

    def set_maximum_iterations(self, num: Int) -> None:
        self._max_iter = num

    def set_mu_init(self, val: Float) -> None:
        self._mu_init = val

    def set_warm_start_init_point(self, val: Str) -> None:
        self._warm_start_init_point = val == "yes"

    def set_warm_start_mult_bound_push(self, val: Float) -> None:
        self._warm_start_mult_bound_push = val

    def set_bound_push(self, val: Float) -> None:
        self._bound_push = val

    def set_print_level(self, num: Int) -> None:
        self._print_level = num

    def set_c_compile(self, val: Bool) -> None:
        self._c_compile = val

    def set_convergence_tolerance(self, val: Float) -> None:
        self._tol = val
        self._acceptable_tol = val

    def set_constraint_tolerance(self, val: Float) -> None:
        self._constr_viol_tol = val

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

    def set_initialization_options(self, val: Float) -> None:
        """
        This function set global initialization options

        Parameters
        ----------
        val: float
            warm start value
        """
        self._bound_push = val

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
        non_python_options = [
            "_c_compile",
            "type",
            "show_online_optim",
            "online_optim",
            "show_options",
            "_structure_detection",
        ]
        for key in solver_options:
            if key not in non_python_options:
                fatrop_key = f"fatrop.{key[1:]}"
                options[fatrop_key] = solver_options[key]
        options["structure_detection"] = self._structure_detection
        options["equality"] = (solver.limits["lbg"] == solver.limits["ubg"])[:, 0].tolist()

        return {**options, **solver.options_common}
