from dataclasses import dataclass

from ..misc.enums import SolverType
from .abstract_options import GenericSolver


from ..misc.parameters_types import (
    Str,
    Bool,
    Int,
    IntorFloat,
    Float,
    StrOptional,
)


@dataclass
class ACADOS(GenericSolver):
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
    _qp_solver: Str = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    _hessian_approx: Str = "GAUSS_NEWTON"
    _integrator_type: Str = "IRK"
    _nlp_solver_type: Str = "SQP"
    _nlp_solver_tol_comp: Float = 1e-06
    _nlp_solver_tol_eq: Float = 1e-06
    _nlp_solver_tol_ineq: Float = 1e-06
    _nlp_solver_tol_stat: Float = 1e-06
    _nlp_solver_max_iter: Int = 200
    _sim_method_newton_iter: Int = 5
    _sim_method_num_stages: Int = 4
    _sim_method_num_steps: Int = 1
    _print_level: Int = 1
    _cost_type: Str = "NONLINEAR_LS"
    _constr_type: Str = "BGH"
    _acados_dir: Str = ""
    _has_tolerance_changed: Bool = False
    _only_first_options_has_changed: Bool = False
    _c_compile: Bool = True
    _c_generated_code_path: Str = "c_generated_code"
    _acados_model_name: StrOptional = None

    @property
    def qp_solver(self):
        return self._qp_solver

    def set_qp_solver(self, val: Str):
        self._qp_solver = val
        self.set_only_first_options_has_changed(True)

    def set_option_unsafe(self, val: IntorFloat | Str, name: Str):
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

    def set_hessian_approx(self, val: Str):
        self._hessian_approx = val
        self.set_only_first_options_has_changed(True)

    @property
    def integrator_type(self):
        return self._integrator_type

    def set_integrator_type(self, val: Str):
        self._integrator_type = val
        self.set_only_first_options_has_changed(True)

    @property
    def nlp_solver_type(self):
        return self._nlp_solver_type

    def set_nlp_solver_type(self, val: Str):
        self._nlp_solver_type = val
        self.set_only_first_options_has_changed(True)

    @property
    def sim_method_newton_iter(self):
        return self._sim_method_newton_iter

    def set_sim_method_newton_iter(self, val: Int):
        self._sim_method_newton_iter = val
        self.set_only_first_options_has_changed(True)

    @property
    def sim_method_num_stages(self):
        return self._sim_method_num_stages

    def set_sim_method_num_stages(self, val: Int):
        self._sim_method_num_stages = val
        self.set_only_first_options_has_changed(True)

    @property
    def sim_method_num_steps(self):
        return self._sim_method_num_steps

    def set_sim_method_num_steps(self, val: Int):
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

    def set_constr_type(self, val: Str):
        self._constr_type = val

    @property
    def acados_dir(self):
        return self._acados_dir

    def set_acados_dir(self, val: Str):
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

    def set_nlp_solver_tol_eq(self, val: Float):
        self._nlp_solver_tol_eq = val
        self.set_has_tolerance_changed(True)

    @property
    def nlp_solver_tol_ineq(self):
        return self._nlp_solver_tol_ineq

    def set_nlp_solver_tol_ineq(self, val: Float):
        self._nlp_solver_tol_ineq = val
        self.set_has_tolerance_changed(True)

    @property
    def nlp_solver_tol_stat(self):
        return self._nlp_solver_tol_stat

    def set_nlp_solver_tol_stat(self, val: Float):
        self._nlp_solver_tol_stat = val
        self.set_has_tolerance_changed(True)

    def set_convergence_tolerance(self, val: IntorFloat):
        self.set_nlp_solver_tol_eq(val)
        self.set_nlp_solver_tol_ineq(val)
        self.set_nlp_solver_tol_comp(val)
        self.set_nlp_solver_tol_stat(val)
        self.set_has_tolerance_changed(True)

    def set_constraint_tolerance(self, val: Float):
        self.set_nlp_solver_tol_eq(val)
        self.set_nlp_solver_tol_ineq(val)
        self.set_has_tolerance_changed(True)

    @property
    def has_tolerance_changed(self):
        return self._has_tolerance_changed

    def set_has_tolerance_changed(self, val: Bool):
        self._has_tolerance_changed = val

    @property
    def only_first_options_has_changed(self):
        return self._only_first_options_has_changed

    def set_only_first_options_has_changed(self, val: Bool):
        self._only_first_options_has_changed = val

    @property
    def nlp_solver_max_iter(self):
        return self._nlp_solver_max_iter

    def set_maximum_iterations(self, num: Int):
        self._nlp_solver_max_iter = num
        self.set_only_first_options_has_changed(True)

    def as_dict(self, solver):
        options = {}
        keys_to_skip = {
            "_acados_dir",
            "_cost_type",
            "_constr_type",
            "_has_tolerance_changed",
            "_only_first_options_has_changed",
            "type",
            "_c_compile",
            "_c_generated_code_path",
            "_acados_model_name",
        }

        # Select the set of relevant keys before entering the loop
        relevant_keys = set(self.__annotations__.keys()) - keys_to_skip

        # Iterate only over relevant keys
        for key in relevant_keys:
            option_key = key[1:] if key[0] == "_" else key
            options[option_key] = getattr(self, key)

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

    def set_c_compile(self, val: Bool):
        self._c_compile = val

    def set_c_generated_code_path(self, val: Str):
        self._c_generated_code_path = val

    def set_acados_model_name(self, val: Str):
        self._acados_model_name = val

    @staticmethod
    def get_tolerance_keys():
        return [
            "_nlp_solver_tol_comp",
            "_nlp_solver_tol_eq",
            "_nlp_solver_tol_ineq",
            "_nlp_solver_tol_stat",
        ]
