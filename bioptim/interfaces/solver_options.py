from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..misc.enums import SolverType


class Solver:
    @dataclass
    class Generic(ABC):
        """
        Abstract class for Solver Options

        Methods
        -------
        set_convergence_tolerance(self,tol: float):
            Set some convergence tolerance
        set_constraint_tolerance(self, tol: float):
            Set some constraint tolerance
        """

        type: SolverType = SolverType.NONE

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
        def as_dict(self, solver) -> dict:
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
    class IPOPT(Generic):
        """
        Class for Solver Options of IPOPT

        Attributes
        ----------
        show_online_optim: bool
            If the plot should be shown while optimizing. It will slow down the optimization a bit
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
        """

        type: SolverType = SolverType.IPOPT
        show_online_optim: bool = False
        show_options: dict = None
        _tol: float = 1e-6  # default in ipopt 1e-8
        _dual_inf_tol: float = 1.0
        _constr_viol_tol: float = 0.0001
        _compl_inf_tol: float = 0.0001
        _acceptable_tol: float = 1e-6
        _acceptable_dual_inf_tol: float = 1e10
        _acceptable_constr_viol_tol: float = 1e-2
        _acceptable_compl_inf_tol: float = 1e-2
        _max_iter: int = 1000
        _hessian_approximation: str = "exact"  # "exact", "limited-memory"
        _limited_memory_max_history: int = 50
        _linear_solver: str = "mumps"  # "ma57", "ma86", "mumps"
        _mu_init: float = 0.1
        _warm_start_init_point: str = "no"
        _warm_start_mult_bound_push: float = 0.001
        _warm_start_slack_bound_push: float = 0.001
        _warm_start_bound_push: float = 0.001
        _warm_start_slack_bound_frac: float = 0.001
        _warm_start_bound_frac: float = 0.001
        _bound_push: float = 0.01
        _bound_frac: float = 0.01
        _print_level: int = 5
        _c_compile: bool = False

        @property
        def tol(self):
            return self._tol

        @property
        def dual_inf_tol(self):
            return self._constr_viol_tol

        @property
        def constr_viol_tol(self):
            return self._tol

        @property
        def compl_inf_tol(self):
            return self._compl_inf_tol

        @property
        def acceptable_tol(self):
            return self._acceptable_tol

        @property
        def acceptable_dual_inf_tol(self):
            return self._acceptable_dual_inf_tol

        @property
        def acceptable_constr_viol_tol(self):
            return self._acceptable_constr_viol_tol

        @property
        def acceptable_compl_inf_tol(self):
            return self._acceptable_compl_inf_tol

        @property
        def max_iter(self):
            return self._max_iter

        @property
        def hessian_approximation(self):
            return self._hessian_approximation

        @property
        def limited_memory_max_history(self):
            return self._limited_memory_max_history

        @property
        def linear_solver(self):
            return self._linear_solver

        @property
        def mu_init(self):
            return self._mu_init

        @property
        def warm_start_init_point(self):
            return self._warm_start_init_point

        @property
        def warm_start_mult_bound_push(self):
            return self._warm_start_mult_bound_push

        @property
        def warm_start_slack_bound_push(self):
            return self._warm_start_slack_bound_push

        @property
        def warm_start_bound_push(self):
            return self._warm_start_bound_push

        @property
        def warm_start_slack_bound_frac(self):
            return self._warm_start_init_point

        @property
        def warm_start_bound_frac(self):
            return self._warm_start_bound_frac

        @property
        def bound_push(self):
            return self._bound_push

        @property
        def bound_frac(self):
            return self._bound_frac

        @property
        def print_level(self):
            return self._print_level

        @property
        def c_compile(self):
            return self._c_compile

        def set_tol(self, val: float):
            self._tol = val

        def set_dual_inf_tol(self, val: float):
            self._constr_viol_tol = val

        def set_constr_viol_tol(self, val: float):
            self._tol = val

        def set_compl_inf_tol(self, val: float):
            self._compl_inf_tol = val

        def set_acceptable_tol(self, val: float):
            self._acceptable_tol = val

        def set_acceptable_dual_inf_tol(self, val: float):
            self._acceptable_dual_inf_tol = val

        def set_acceptable_constr_viol_tol(self, val: float):
            self._acceptable_constr_viol_tol = val

        def set_acceptable_compl_inf_tol(self, val: float):
            self._acceptable_compl_inf_tol = val

        def set_maximum_iterations(self, num):
            self._max_iter = num

        def set_hessian_approximation(self, val: str):
            self._hessian_approximation = val

        def set_limited_memory_max_history(self, num: int):
            self._limited_memory_max_history = num

        def set_linear_solver(self, val: str):
            self._linear_solver = val

        def set_mu_init(self, val: float):
            self._mu_init = val

        def set_warm_start_init_point(self, val: str):
            self._warm_start_init_point = val

        def set_warm_start_mult_bound_push(self, val: float):
            self._warm_start_mult_bound_push = val

        def set_warm_start_slack_bound_push(self, val: float):
            self._warm_start_slack_bound_push = val

        def set_warm_start_bound_push(self, val: float):
            self._warm_start_bound_push = val

        def set_warm_start_slack_bound_frac(self, val: float):
            self._warm_start_slack_bound_frac = val

        def set_warm_start_bound_frac(self, val: float):
            self._warm_start_bound_frac = val

        def set_bound_push(self, val: float):
            self._bound_push = val

        def set_bound_frac(self, val: float):
            self._bound_frac = val

        def set_print_level(self, num: int):
            self._print_level = num

        def set_c_compile(self, val: bool):
            self._c_compile = val

        def set_convergence_tolerance(self, val: float):
            self._tol = val
            self._compl_inf_tol = val
            self._acceptable_tol = val
            self._acceptable_compl_inf_tol = val

        def set_constraint_tolerance(self, val: float):
            self._constr_viol_tol = val
            self._acceptable_constr_viol_tol = val

        def set_warm_start_options(self, val: float = 1e-10):
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

        def set_initialization_options(self, val: float):
            """
            This function set global initialization options

            Parameters
            ----------
            val: float
                warm start value
            """
            self._bound_push = val
            self._bound_frac = val

        def set_option_unsafe(self, val, name):
            """
            This function is unsafe because we did not check if the option exist in the solver option list.
            If it's not it just will be ignored. Please make sure that the option you're asking for exist.
            """
            if f"_{name}" not in self.__dict__.keys():
                self.__dict__[f"_{name}"] = val

        def as_dict(self, solver):
            solver_options = self.__dict__
            options = {}
            non_python_options = ["_c_compile", "type", "show_online_optim", "show_options"]
            for key in solver_options:
                if key not in non_python_options:
                    ipopt_key = "ipopt." + key[1:]
                    options[ipopt_key] = solver_options[key]
            return {**options, **solver.options_common}

    @dataclass
    class SQP_METHOD(Generic):
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

    @dataclass
    class ACADOS(Generic):
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
        _c_compile: bool
            True if you want to compile in C the code.
        _c_generated_code_path: str
            Directory of the generated code (default: "c_generated_code").
        _acados_model_name: str
            Name of the Acados model name used to build an existing c_generated library.
        """

        type: SolverType = SolverType.ACADOS
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
        _c_compile: bool = True
        _c_generated_code_path: str = "c_generated_code"
        _acados_model_name: str = None

        @property
        def qp_solver(self):
            return self._qp_solver

        def set_qp_solver(self, val: str):
            self._qp_solver = val
            self.set_only_first_options_has_changed(True)

        def set_option_unsafe(self, val: float | int | str, name: str):
            """
            This function is unsafe because we did not check if the option exist in the solver option list.
            If it's not it just will be ignored. Please make sure that the option you're asking for exist.
            """
            if f"_{name}" not in self.__annotations__.keys():
                self.__annotations__[f"_{name}"] = val
                self.__setattr__(f"_{name}", val)
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
            self._nlp_solver_tol_eq = val
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

        def set_convergence_tolerance(self, val: float | int):
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

        def set_maximum_iterations(self, num: int):
            self._nlp_solver_max_iter = num
            self.set_only_first_options_has_changed(True)

        def as_dict(self, solver):
            options = {}
            for key in self.__annotations__.keys():
                if (
                    key == "_acados_dir"
                    or key == "_cost_type"
                    or key == "_constr_type"
                    or key == "_has_tolerance_changed"
                    or key == "_only_first_options_has_changed"
                    or key == "type"
                    or key == "_c_compile"
                    or key == "_c_generated_code_path"
                    or key == "_acados_model_name"
                ):
                    continue
                if key[0] == "_":
                    options[key[1:]] = self.__getattribute__(key)
                else:
                    options[key] = self.__getattribute__(key)
            return options

        @property
        def print_level(self):
            return self._print_level

        @property
        def c_compile(self):
            return self._c_compile

        @property
        def c_generated_code_path(self):
            return self._c_generated_code_path

        @property
        def acados_model_name(self):
            return self._acados_model_name

        def set_print_level(self, num: int):
            self._print_level = num
            self.set_only_first_options_has_changed(True)

        def set_c_compile(self, val: bool):
            self._c_compile = val

        def set_c_generated_code_path(self, val: str):
            self._c_generated_code_path = val

        def set_acados_model_name(self, val: str):
            self._acados_model_name = val

        @staticmethod
        def get_tolerance_keys():
            return [
                "_nlp_solver_tol_comp",
                "_nlp_solver_tol_eq",
                "_nlp_solver_tol_ineq",
                "_nlp_solver_tol_stat",
            ]
