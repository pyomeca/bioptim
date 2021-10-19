from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class SolverOptions(ABC):
    """
    Abstract class for Solver Options

    Methods
    -------
    set_convergence_tolerance(self,tol: float):
        Set some convergence tolerance
    set_constraint_tolerance(self, tol: float):
        Set some constraint tolerance
    """

    @abstractmethod
    def set_convergence_tolerance(self, tol: float):
        """
        This function set the convergence tolerance

        Parameters
        ----------
        tol: float
            Global converge tolerance value
        """

    @abstractmethod
    def set_constraint_tolerance(self, tol: float):
        """
        This function set the constraint tolerance.

        Parameters
        ----------
        tol: float
            Global constraint tolerance value
        """

    @abstractmethod
    def set_maximum_iterations(self, num: int):
        """
        This function set the number of maximal iterations.

        Parameters
        ----------
        num: int
            Number of iterations
        """

    @abstractmethod
    def as_dict(self, solver) -> Any:
        """
        This function return the dict options to launch the optimization

        Parameters
        ----------
        solver: SolverInterface
            Ipopt ou Acados interface
        """

    @abstractmethod
    def set_print_level(self, num: int):
        """
        This function set Output verbosity level.

        Parameters
        ----------
        num: int
            print_level
        """


@dataclass
class SolverOptionsIpopt(SolverOptions):
    """
    Class for Solver Options of IPOPT

    Attributes
    ----------
    tol: float
        Desired convergence tolerance (relative)
    dual_inf_tol: float
        Desired threshold for the dual infeasibility
    constr_viol_tol: float
        Desired threshold for the constraint and variable bound violation.
    compl_inf_tol: float
        Desired threshold for the complementarity conditions.
    acceptable_tol: float
        Acceptable convergence tolerance (relative).
    acceptable_dual_inf_tol: float
        Acceptance threshold for the dual infeasibility
    acceptable_constr_viol_tol: float
        Acceptance threshold for the constraint violation.
    acceptable_compl_inf_tol: float
        "Acceptance" threshold for the complementarity conditions.
    max_iter: int
        Maximum number of iterations.
    hessian_approximation: str
        Indicates what Hessian information is to be used.
    limited_memory_max_history: int
        Maximum size of the history for the limited quasi-Newton Hessian approximation.
    linear_solver: str
        Linear solver used for step computations.
    mu_init: float
        Initial value for the barrier parameter.
    warm_start_init_point: float
        Warm-start for initial point
    warm_start_mult_bound_push: float
        same as mult_bound_push for the regular initializer
    warm_start_slack_bound_push: float
        same as slack_bound_push for the regular initializer
    warm_start_slack_bound_frac: float
        same as slack_bound_frac for the regular initializer
    warm_start_bound_frac: float
        same as bound_frac for the regular initializer
    bound_push: float
        Desired minimum absolute distance from the initial point to bound.
    bound_frac: float
        Desired minimum relative distance from the initial point to bound.
    print_level: float
    Output verbosity level. Sets the default verbosity level for console output. The larger this value the more detailed is the output. The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.
    """

    tol: float = 1e-6  # default in ipopt 1e-8
    dual_inf_tol: float = 1.0
    constr_viol_tol: float = 0.0001
    compl_inf_tol: float = 0.0001
    acceptable_tol: float = 1e-2
    acceptable_dual_inf_tol: float = 1e-2
    acceptable_constr_viol_tol: float = 1e-2
    acceptable_compl_inf_tol: float = 1e-2
    max_iter: int = 1000
    hessian_approximation: str = "exact"  # "exact", "limited-memory"
    limited_memory_max_history: int = 50
    linear_solver: str = "mumps"  # "ma57", "ma86", "mumps"
    mu_init: float = 0.1
    warm_start_init_point: str = "no"
    warm_start_mult_bound_push: float = 0.001
    warm_start_slack_bound_push: float = 0.001
    warm_start_bound_push: float = 0.001
    warm_start_slack_bound_frac: float = 0.001
    warm_start_bound_frac: float = 0.001
    bound_push: float = 0.01
    bound_frac: float = 0.01
    print_level: int = 5

    def set_convergence_tolerance(self, val: float):
        self.tol = val
        self.compl_inf_tol = val
        self.acceptable_tol = val
        self.acceptable_compl_inf_tol = val

    def set_constraint_tolerance(self, val: float):
        self.constr_viol_tol = val
        self.acceptable_constr_viol_tol = val

    def set_maximum_iterations(self, num):
        self.max_iter = num

    def set_warm_start_options(self, val: float = 1e-10):
        """
        This function set global warm start options

        Parameters
        ----------
        val: float
            warm start value
        """

        self.warm_start_init_point = "yes"
        self.mu_init = val
        self.warm_start_mult_bound_push = val
        self.warm_start_slack_bound_push = val
        self.warm_start_bound_push = val
        self.warm_start_slack_bound_frac = val
        self.warm_start_bound_frac = val

    def set_initialization_options(self, val: float):
        """
        This function set global initialization options

        Parameters
        ----------
        val: float
            warm start value
        """
        self.bound_push = val
        self.bound_frac = val

    def as_dict(self, solver):
        solver_options = self.__dict__
        options = {}
        for key in solver_options:
            ipopt_key = "ipopt." + key
            options[ipopt_key] = solver_options[key]
        return {**options, **solver.options_common}

    def set_print_level(self, num: int):
        self.print_level = num


@dataclass
class SolverOptionsAcados(SolverOptions):
    """
    Class for Solver Options of ACADOS

    Methods
    ----------
    get_tolerance_keys
        return the keys of the optimizer tolerance

    Attributes
    ----------
    _qp_solver: str
        QP solver to be used in the NLP solver. String in (‘PARTIAL_CONDENSING_HPIPM’, ‘FULL_CONDENSING_QPOASES’,
        ‘FULL_CONDENSING_HPIPM’, ‘PARTIAL_CONDENSING_QPDUNES’, ‘PARTIAL_CONDENSING_OSQP’).
        Default: ‘PARTIAL_CONDENSING_HPIPM’
    _hessian_approx: str
        Hessian approximation.
    _integrator_type: str
        Integrator type.
    _nlp_solver_type: str
        Desired threshold for the complementarity conditions.
    _nlp_solver_tol_comp: float
        NLP solver complementarity tolerance
    _nlp_solver_tol_eq: float
        NLP solver equality tolerance
    _nlp_solver_tol_ineq: float
        NLP solver inequality tolerance
    _nlp_solver_tol_stat: float
        NLP solver stationarity tolerance. Type: float > 0 Default: 1e-6
    _nlp_solver_max_iter: int
        NLP solver maximum number of iterations.
    _sim_method_newton_iter: int
        Number of Newton iterations in simulation method. Type: int > 0 Default: 3
    _sim_method_num_stages: int
        Number of stages in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 4
    _sim_method_num_steps: int
        Number of steps in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 1
    _print_level: int
        Verbosity of printing.
    _cost_type: int
        type of cost functions for cost.cost_type and cost.cost_type_e
    _constr_type: int
        type of constraint functions for constraints.constr_type and constraints.constr_type_e
    _acados_dir: str
        If Acados is installed using the acados_install.sh file, you probably can leave this unset
    _has_tolerance_changed: bool
        True if the tolerance has been modified, use ful for moving horizon estimation
    _only_first_options_has_changed
        True if non editable options has been modified in options.
    """

    _qp_solver: str = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    _hessian_approx: str = "GAUSS_NEWTON"
    _integrator_type: str = "IRK"
    _nlp_solver_type: str = "SQP"
    _nlp_solver_tol_comp: float = 1e-06
    _nlp_solver_tol_eq: float = 1e-06
    _nlp_solver_tol_ineq: float = 1e-06
    _nlp_solver_tol_stat: float = 1e-06
    _nlp_solver_max_iter: int = 200
    _sim_method_newton_iter: int = 5
    _sim_method_num_stages: int = 4
    _sim_method_num_steps: int = 1
    _print_level: int = 1
    _cost_type: str = "NONLINEAR_LS"
    _constr_type: str = "BGH"
    _acados_dir: str = ""
    _has_tolerance_changed: bool = False
    _only_first_options_has_changed: bool = False

    @property
    def qp_solver(self):
        return self._qp_solver

    def set_qp_solver(self, val: str):
        self._qp_solver = val
        self.set_only_first_options_has_changed(True)

    @property
    def hessian_approx(self):
        return self._hessian_approx

    def set_hessian_approx(self, val: str):
        self._hessian_approx = val
        self.set_only_first_options_has_changed(True)

    @property
    def integrator_type(self):
        return self._integrator_type

    def set_integrator_type(self, val: str):
        self._integrator_type = val
        self.set_only_first_options_has_changed(True)

    @property
    def nlp_solver_type(self):
        return self._nlp_solver_type

    def set_nlp_solver_type(self, val: str):
        self._nlp_solver_type = val
        self.set_only_first_options_has_changed(True)

    @property
    def sim_method_newton_iter(self):
        return self._sim_method_newton_iter

    def set_sim_method_newton_iter(self, val: int):
        self._sim_method_newton_iter = val
        self.set_only_first_options_has_changed(True)

    @property
    def sim_method_num_stages(self):
        return self._sim_method_num_stages

    def set_sim_method_num_stages(self, val: int):
        self._sim_method_num_stages = val
        self.set_only_first_options_has_changed(True)

    @property
    def sim_method_num_steps(self):
        return self._sim_method_num_steps

    def set_sim_method_num_steps(self, val: int):
        self._sim_method_num_steps = val
        self.set_only_first_options_has_changed(True)

    @property
    def cost_type(self):
        return self._cost_type

    def set_cost_type(self, val: str):
        self._cost_type = val

    @property
    def constr_type(self):
        return self._constr_type

    def set_constr_type(self, val: str):
        self._constr_type = val

    @property
    def acados_dir(self):
        return self._acados_dir

    def set_acados_dir(self, val: str):
        self._acados_dir = val

    @property
    def nlp_solver_tol_comp(self):
        return self._nlp_solver_tol_comp

    def set_nlp_solver_tol_comp(self, val: float):
        self._nlp_solver_tol_comp = val
        self._has_tolerance_changed = True

    @property
    def nlp_solver_tol_eq(self):
        return self._nlp_solver_tol_eq

    def set_nlp_solver_tol_eq(self, val: float):
        self._nlp_solver_tol_comp = val
        self.set_has_tolerance_changed(True)

    @property
    def nlp_solver_tol_ineq(self):
        return self._nlp_solver_tol_ineq

    def set_nlp_solver_tol_ineq(self, val: float):
        self._nlp_solver_tol_ineq = val
        self.set_has_tolerance_changed(True)

    @property
    def nlp_solver_tol_stat(self):
        return self._nlp_solver_tol_stat

    def set_nlp_solver_tol_stat(self, val: float):
        self._nlp_solver_tol_stat = val
        self.set_has_tolerance_changed(True)

    def set_convergence_tolerance(self, val: float):
        self.set_nlp_solver_tol_eq(val)
        self.set_nlp_solver_tol_ineq(val)
        self.set_nlp_solver_tol_comp(val)
        self.set_nlp_solver_tol_stat(val)
        self.set_has_tolerance_changed(True)

    def set_constraint_tolerance(self, val: float):
        self.set_nlp_solver_tol_eq(val)
        self.set_nlp_solver_tol_ineq(val)
        self.set_has_tolerance_changed(True)

    @property
    def has_tolerance_changed(self):
        return self._has_tolerance_changed

    def set_has_tolerance_changed(self, val: bool):
        self._has_tolerance_changed = val

    @property
    def only_first_options_has_changed(self):
        return self._only_first_options_has_changed

    def set_only_first_options_has_changed(self, val: bool):
        self._only_first_options_has_changed = val

    @property
    def nlp_solver_max_iter(self):
        return self._nlp_solver_max_iter

    def set_maximum_iterations(self, num):
        self._nlp_solver_max_iter = num
        self.set_only_first_options_has_changed(True)

    def as_dict(self, solver):
        options = {}
        for key in self.__annotations__.keys():
            if key == "_acados_dir" or key == "_cost_type" or key == "_constr_type" \
                    or key == "_has_tolerance_changed" or key == "_only_first_options_has_changed":
                continue
            if key[0] == "_":
                options[key[1:]] = self.__getattribute__(key)
            else:
                options[key] = self.__getattribute__(key)

        return options

    @property
    def print_level(self):
        return self._print_level

    def set_print_level(self, num: int):
        self._print_level = num
        self.set_only_first_options_has_changed(True)

    @staticmethod
    def get_tolerance_keys():
        return [
            "nlp_solver_tol_comp",
            "nlp_solver_tol_eq",
            "nlp_solver_tol_ineq",
            "nlp_solver_tol_stat",
        ]
