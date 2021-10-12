from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    def finalize_options(self, solver):
        """
        This function return the finalize options structure to launch the optimization

        Parameters
        ----------
        solver: SolverInterface
            Ipopt ou Acados interface
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
        This function global set warm start options

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

    def finalize_options(self, solver):
        solver_options = self.__dict__
        options = {}
        for key in solver_options:
            ipopt_key = "ipopt." + key
            options[ipopt_key] = solver_options[key]
        return {**options, **solver.options_common}


@dataclass
class SolverOptionsAcados(SolverOptions):
    """
    Class for Solver Options of ACADOS

    Attributes
    ----------
    qp_solver: str
        QP solver to be used in the NLP solver. String in (‘PARTIAL_CONDENSING_HPIPM’, ‘FULL_CONDENSING_QPOASES’,
        ‘FULL_CONDENSING_HPIPM’, ‘PARTIAL_CONDENSING_QPDUNES’, ‘PARTIAL_CONDENSING_OSQP’).
        Default: ‘PARTIAL_CONDENSING_HPIPM’
    hessian_approx: str
        Hessian approximation.
    integrator_type: str
        Integrator type.
    nlp_solver_type: str
        Desired threshold for the complementarity conditions.
    nlp_solver_tol_comp: float
        NLP solver complementarity tolerance
    nlp_solver_tol_eq: float
        NLP solver equality tolerance
    nlp_solver_tol_ineq: float
        NLP solver inequality tolerance
    nlp_solver_tol_stat: float
        NLP solver stationarity tolerance. Type: float > 0 Default: 1e-6
    nlp_solver_max_iter: int
        NLP solver maximum number of iterations.
    sim_method_newton_iter: int
        Number of Newton iterations in simulation method. Type: int > 0 Default: 3
    sim_method_num_stages: int
        Number of stages in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 4
    sim_method_num_steps: int
        Number of steps in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 1
    print_level: int
        Verbosity of printing.
    cost_type: int
        type of cost functions for cost.cost_type and cost.cost_type_e
    constr_type: int
        type of constraint functions for constraints.constr_type and constraints.constr_type_e
    """

    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "IRK"
    nlp_solver_type: str = "SQP"
    nlp_solver_tol_comp: float = 1e-06
    nlp_solver_tol_eq: float = 1e-06
    nlp_solver_tol_ineq: float = 1e-06
    nlp_solver_tol_stat: float = 1e-06
    nlp_solver_max_iter: int = 200
    sim_method_newton_iter: int = 5
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    print_level: int = 1
    cost_type: str = "NONLINEAR_LS"
    constr_type: str = "BGH"

    def set_convergence_tolerance(self, val: float):
        self.nlp_solver_tol_eq = val
        self.nlp_solver_tol_ineq = val
        self.nlp_solver_tol_comp = val
        self.nlp_solver_tol_stat = val

    def set_constraint_tolerance(self, val: float):
        self.nlp_solver_tol_eq = val
        self.nlp_solver_tol_ineq = val

    def set_maximum_iterations(self, num):
        self.nlp_solver_max_iter = num

    def finalize_options(self, solver):
        options = self.__dict__

        if "acados_dir" in options:
            del options["acados_dir"]
        if "cost_type" in options:
            del options["cost_type"]
        if "constr_type" in options:
            del options["constr_type"]

        for key in options:
            setattr(solver.acados_ocp.solver_options, key, options[key])

        return solver.acados_ocp.solver_options
