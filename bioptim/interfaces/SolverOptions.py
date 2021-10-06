from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SolverOptions(ABC):
    @abstractmethod
    def set_convergence_tol(self, tol):
        """
         This function set the convergence tolerance
        """

    @abstractmethod
    def set_constraint_tol(self, tol):
        """
         This function set the constraint tolerance.
        """

    @abstractmethod
    def set_solver(self, name):
        """
         This function set the solver
        """

    def set_max_iter(self, num):
        """
         This function set the maximal number of iterations
        """


@dataclass
class SolverOptionsIPOPT(SolverOptions):
    tol: float = 1e-6  # default in ipopt 1e-8
    dual_inf_tol: float = 1.0  # default in ipotpt 1
    constr_viol_tol: float = 0.0001  # default in ipotpt 0.0001
    compl_inf_tol: float = 0.0001  # default in ipotpt 0.0001
    acceptable_tol: float = 1e-2  # default in ipopt 1e-2
    acceptable_dual_inf_tol: float = 1e-2  # default in ipopt 1e-2
    acceptable_constr_viol_tol: float = 1e-2  # default in ipopt 1e-2
    acceptable_compl_inf_tol: float = 1e-2  # default in ipopt 1e-2
    max_iter: int = 1000
    hessian_approximation: str = "exact"  # "exact", "limited-memory"
    limited_memory_max_history: int = 50
    linear_solver: str = "mumps"  # "ma57", "ma86", "mumps"

    def set_convergence_tol(self, val):
        self.tol = val
        self.compl_inf_tol = val
        self.acceptable_tol = val
        self.acceptable_compl_inf_tol = val

    def set_constraint_tol(self, val):
        self.constr_viol_tol = val
        self.acceptable_constr_viol_tol = val

    def set_solver(self, name):
        self.linear_solver = name

    def set_max_iter(self, num):
        self.max_iter = num


@dataclass
class SolverOptionsACADOS(SolverOptions):
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "IRK"
    nlp_solver_type: str = "SQP"
    nlp_solver_tol_comp: float = 1e-06
    nlp_solver_tol_eq: float = 1e-06
    nlp_solver_tol_ineq: float = 1e-06
    nlp_solver_tol_stat: float = 1e-06
    nlp_solver_max_iter: float = 200
    sim_method_newton_iter: float = 5
    sim_method_num_stages: float = 4
    sim_method_num_steps: float = 1
    print_level: float = 1
    cost_type: str = "NONLINEAR_LS"
    constr_type: str = "BGH"

    def set_convergence_tol(self, val):
        self.nlp_solver_tol_eq = val
        self.nlp_solver_tol_ineq = val
        self.nlp_solver_tol_comp = val
        self.nlp_solver_tol_stat = val

    def set_constraint_tol(self, val):
        self.nlp_solver_tol_eq = val
        self.nlp_solver_tol_ineq = val

    def set_solver(self, name):
        self.qp_solver = name

    def set_max_iter(self, num):
        self.nlp_solver_max_iter = num
