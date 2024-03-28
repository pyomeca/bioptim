from time import perf_counter
from datetime import datetime

import numpy as np
from scipy import linalg
from casadi import SX, vertcat, Function
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from .solver_interface import SolverInterface
from ..interfaces import Solver
from ..misc.enums import Node, SolverType, PhaseDynamics
from ..limits.objective_functions import ObjectiveFunction, ObjectiveFcn
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType


class AcadosInterface(SolverInterface):
    """
    The ACADOS solver interface

    Attributes
    ----------
    acados_ocp: AcadosOcp
        The current AcadosOcp reference
    acados_model: AcadosModel
        The current AcadosModel reference
    lagrange_costs: SX
        The lagrange cost function
    mayer_costs: SX
        The mayer cost function
    y_ref = list[np.ndarray]
        The lagrange targets
    y_ref_end = list[np.ndarray]
        The mayer targets
    params = dict
        All the parameters to optimize
    W: np.ndarray
        The Lagrange weights
    W_e: np.ndarray
        The Mayer weights
    status: int
        The status of the optimization
    all_constr: SX
        All the Lagrange constraints
    end_constr: SX
        All the Mayer constraints
    all_g_bounds = Bounds
        All the Lagrange bounds on the variables
    end_g_bounds = Bounds
        All the Mayer bounds on the variables
    x_bound_max = np.ndarray
        All the bounds max
    x_bound_min = np.ndarray
        All the bounds min
    Vu: np.ndarray
        The control objective functions
    Vx: np.ndarray
        The Lagrange state objective functions
    Vxe: np.ndarray
        The Mayer state objective functions
    opts: ACADOS
        Options of Acados from ACADOS
    Methods
    -------
    __acados_export_model(self, ocp: OptimalControlProgram)
        Creating a generic ACADOS model
    __prepare_acados(self, ocp: OptimalControlProgram)
        Set some important ACADOS variables
    __set_constr_type(self, constr_type: str = "BGH")
        Set the type of constraints
    __set_constraints(self, ocp: OptimalControlProgram)
        Set the constraints from the ocp
    __set_cost_type(self, cost_type: str = "NONLINEAR_LS")
        Set the type of cost functions
    __set_costs(self, ocp: OptimalControlProgram)
        Set the cost functions from ocp
    __update_solver(self)
        Update the ACADOS solver to new values
    get_optimized_value(self) -> list[dict] | dict
        Get the previously optimized solution
    solve(self) -> "AcadosInterface"
        Solve the prepared ocp
    """

    def __init__(self, ocp, solver_options: Solver.ACADOS = None):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        solver_options: ACADOS
            The options to pass to the solver
        """

        if not isinstance(ocp.cx(), SX):
            raise RuntimeError("CasADi graph must be SX to be solved with ACADOS. Please set use_sx to True in OCP")

        if ocp.nlp[0].phase_dynamics != PhaseDynamics.SHARED_DURING_THE_PHASE:
            raise RuntimeError("ACADOS necessitate phase_dynamics==PhaseDynamics.SHARED_DURING_THE_PHASE")

        if ocp.nlp[0].algebraic_states.cx_start.shape[0] != 0:
            raise RuntimeError("ACADOS does not support algebraic states yet")

        super().__init__(ocp)

        # solver_options = solver_options.__dict__
        if solver_options is None:
            solver_options = Solver.ACADOS()
        self.opts = solver_options

        self.acados_ocp = AcadosOcp(acados_path=solver_options.acados_dir)
        self.acados_ocp.code_export_directory = solver_options.c_generated_code_path
        self.acados_model = AcadosModel()

        self.__set_cost_type(solver_options.cost_type)
        self.__set_constr_type(solver_options.constr_type)

        self.lagrange_costs = SX()
        self.mayer_costs_e = SX()
        self.mayer_costs = SX()
        self.y_ref = []
        self.y_ref_end = []
        self.y_ref_start = []
        self.nparams = 0
        self.__acados_export_model(ocp)
        self.__prepare_acados(ocp)
        self.ocp_solver = None
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
        self.W_0 = np.zeros((0, 0))
        self.status = None
        self.out = {}
        self.real_time_to_optimize = -1

        self.all_constr = None
        self.end_constr = SX()
        self.all_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        self.end_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        self.x_bound_max = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.x_bound_min = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.Vu = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].controls.shape)
        self.Vx = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)
        self.Vxe = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)
        self.Vx0 = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)

    def __acados_export_model(self, ocp):
        """
        Creating a generic ACADOS model

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram

        """

        if ocp.n_phases > 1:
            raise NotImplementedError("More than 1 phase is not implemented yet with ACADOS backend")

        # Declare model variables
        t = ocp.nlp[0].time_cx
        x = ocp.nlp[0].states.cx_start
        x_sym = ocp.nlp[0].states.scaled.cx_start
        u = ocp.nlp[0].controls.cx_start
        u_sym = ocp.nlp[0].controls.scaled.cx_start
        p = ocp.nlp[0].parameters.cx
        p_sym = ocp.nlp[0].parameters.scaled.cx
        a = ocp.nlp[0].algebraic_states.cx_start
        a_sym = ocp.nlp[0].algebraic_states.scaled.cx_start

        if ocp.parameters:
            for key in ocp.parameters:
                if str(ocp.parameters[key].cx)[:11] == f"time_phase_":
                    raise RuntimeError("Time constraint not implemented yet with Acados.")
        if a_sym.shape[0] != 0:
            raise RuntimeError("Algebraic states not implemented yet with Acados.")

        self.nparams = ocp.nlp[0].parameters.shape

        x_sym = vertcat(p_sym, x_sym)
        x_dot_sym = SX.sym("x_dot", x_sym.shape[0], x_sym.shape[1])

        f_expl = vertcat([0] * self.nparams, ocp.nlp[0].dynamics_func[0](t, x, u, p, a))
        f_impl = x_dot_sym - f_expl

        self.acados_model.f_impl_expr = f_impl
        self.acados_model.f_expl_expr = f_expl
        self.acados_model.x = x_sym
        self.acados_model.xdot = x_dot_sym
        self.acados_model.u = u_sym
        self.acados_model.con_h_expr = np.zeros((0, 0))
        self.acados_model.con_h_expr_e = np.zeros((0, 0))
        self.acados_model.p = []
        if not self.opts.acados_model_name:
            if self.opts.c_compile:
                now = datetime.now()  # current date and time
                self.acados_model.name = f"model_{now.strftime('%Y_%m_%d_%H%M%S%f')[:-4]}"
            else:
                raise RuntimeError(
                    "If not compiling the library you must provide the name of the model" " you want to use."
                )
        else:
            self.acados_model.name = self.opts.acados_model_name

    def __prepare_acados(self, ocp):
        """
        Set some important ACADOS variables

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        # set model
        self.acados_ocp.model = self.acados_model

        # set time
        tf_init = ocp.dt_parameter_initial_guess.init[0, 0]
        tf = float(Function("tf", [ocp.nlp[0].dt], [ocp.nlp[0].tf])(tf_init))
        self.acados_ocp.solver_options.tf = tf

        # set dimensions
        self.acados_ocp.dims.nx = ocp.nlp[0].parameters.shape + ocp.nlp[0].states.shape
        self.acados_ocp.dims.nu = ocp.nlp[0].controls.shape
        self.acados_ocp.dims.N = ocp.nlp[0].ns

    def __set_constr_type(self, constr_type: str = "BGH"):
        """
        Set the type of constraints

        Parameters
        ----------
        constr_type: str
            The requested type of constraints
        """

        self.acados_ocp.constraints.constr_type = constr_type
        self.acados_ocp.constraints.constr_type_e = constr_type

    def __set_constraints(self, ocp):
        """
        Set the constraints from the ocp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        # constraints handling in self.acados_ocp
        if ocp.nlp[0].x_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            raise NotImplementedError(
                "ACADOS must declare an InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT " "for the x_bounds"
            )
        if ocp.nlp[0].u_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            raise NotImplementedError(
                "ACADOS must declare an InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT " "for the u_bounds"
            )

        for key in ocp.nlp[0].controls.keys():
            if not np.all(np.all(ocp.nlp[0].u_bounds[key].min.T == ocp.nlp[0].u_bounds[key].min.T[0, :], axis=0)):
                raise NotImplementedError("u_bounds min must be the same at each shooting point with ACADOS")
            if not np.all(np.all(ocp.nlp[0].u_bounds[key].max.T == ocp.nlp[0].u_bounds[key].max.T[0, :], axis=0)):
                raise NotImplementedError("u_bounds max must be the same at each shooting point with ACADOS")

            if (
                not np.isfinite(ocp.nlp[0].u_bounds[key].min).all()
                or not np.isfinite(ocp.nlp[0].u_bounds[key].max).all()
            ):
                raise NotImplementedError(
                    "u_bounds and x_bounds cannot be set to infinity in ACADOS. Consider changing it "
                    "to a big value instead."
                )

        for key in ocp.nlp[0].states.keys():
            if (
                not np.isfinite(ocp.nlp[0].x_bounds[key].min).all()
                or not np.isfinite(ocp.nlp[0].x_bounds[key].max).all()
            ):
                raise NotImplementedError(
                    "u_bounds and x_bounds cannot be set to infinity in ACADOS. Consider changing it "
                    "to a big value instead."
                )

        self.all_constr = SX()
        self.end_constr = SX()
        # TODO:change for more node flexibility on bounds
        self.all_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        self.end_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        for i, nlp in enumerate(ocp.nlp):
            t = nlp.time_cx
            dt = nlp.dt
            x = nlp.states.cx_start
            u = nlp.controls.cx_start
            p = nlp.parameters.cx
            a = nlp.algebraic_states.cx_start

            for g, G in enumerate(nlp.g):
                if not G:
                    continue

                if G.node[0] == Node.ALL or G.node[0] == Node.ALL_SHOOTING:
                    x_tp = x
                    u_tp = u
                    if x.shape[0] * 2 == G.function[0].size_in("x")[0]:
                        x_tp = vertcat(x_tp, x_tp)
                    if u.shape[0] * 2 == G.function[0].size_in("u")[0]:
                        u_tp = vertcat(u_tp, u_tp)

                    self.all_constr = vertcat(self.all_constr, G.function[0](t, dt, x_tp, u_tp, p, a))
                    self.all_g_bounds.concatenate(G.bounds)
                    if G.node[0] == Node.ALL:
                        self.end_constr = vertcat(self.end_constr, G.function[0](t, dt, x_tp, u_tp, p, a))
                        self.end_g_bounds.concatenate(G.bounds)

                elif G.node[0] == Node.END:
                    x_tp = x
                    u_tp = u
                    if x.shape[0] * 2 == G.function[-1].size_in("x")[0]:
                        x_tp = vertcat(x_tp, x_tp)
                    if u.shape[0] * 2 == G.function[-1].size_in("u")[0]:
                        u_tp = vertcat(u_tp, u_tp)

                    self.end_constr = vertcat(self.end_constr, G.function[-1](t, dt, x_tp, u_tp, p, a))
                    self.end_g_bounds.concatenate(G.bounds)

                else:
                    raise RuntimeError(
                        "Except for states and controls, Acados solver only handles constraints on last or all nodes."
                    )

        self.acados_model.con_h_expr = self.all_constr
        self.acados_model.con_h_expr_e = self.end_constr

        # setup state constraints
        # TODO replace all these np.concatenate by proper bound and initial_guess classes
        self.x_bound_max = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.x_bound_min = np.ndarray((self.acados_ocp.dims.nx, 3))
        param_bounds_max = []
        param_bounds_min = []

        for key in ocp.parameter_bounds.keys():
            param_bounds_scale = ocp.parameter_bounds[key].scale(ocp.parameters[key].scaling.scaling)
            param_bounds_max = np.concatenate((param_bounds_max, param_bounds_scale.max[:, 0]))
            param_bounds_min = np.concatenate((param_bounds_min, param_bounds_scale.min[:, 0]))

        if self.nparams > 0:
            self.x_bound_max[: self.nparams, :] = np.repeat(param_bounds_max[:, np.newaxis], 3, axis=1)
            self.x_bound_min[: self.nparams, :] = np.repeat(param_bounds_min[:, np.newaxis], 3, axis=1)

        for key in ocp.nlp[0].states.keys():
            x_tp = ocp.nlp[0].x_bounds[key].scale(ocp.nlp[0].x_scaling[key].scaling)
            index = [i + self.nparams for i in ocp.nlp[0].states[key].index]
            for i in range(3):
                self.x_bound_max[index, i] = x_tp.max[:, i]
                self.x_bound_min[index, i] = x_tp.min[:, i]

        # setup control constraints
        u_bounds_max = np.ndarray((self.acados_ocp.dims.nu, 1))
        u_bounds_min = np.ndarray((self.acados_ocp.dims.nu, 1))
        for key in ocp.nlp[0].controls.keys():
            u_tp = ocp.nlp[0].u_bounds[key].scale(ocp.nlp[0].u_scaling[key].scaling)
            index = ocp.nlp[0].controls[key].index
            u_bounds_max[index, 0] = np.array(u_tp.max[:, 0])
            u_bounds_min[index, 0] = np.array(u_tp.min[:, 0])

        self.acados_ocp.constraints.lbu = u_bounds_max
        self.acados_ocp.constraints.ubu = u_bounds_min
        self.acados_ocp.constraints.idxbu = np.array(range(self.acados_ocp.dims.nu))
        self.acados_ocp.dims.nbu = self.acados_ocp.dims.nu

        # initial state constraints
        self.acados_ocp.constraints.Jbx_0 = np.eye(self.acados_ocp.dims.nx)
        self.acados_ocp.constraints.lbx_0 = self.x_bound_min[:, 0]
        self.acados_ocp.constraints.ubx_0 = self.x_bound_max[:, 0]
        self.acados_ocp.constraints.idxbx_0 = np.array(range(self.acados_ocp.dims.nx))
        self.acados_ocp.dims.nbx_0 = self.acados_ocp.dims.nx

        # setup path state constraints
        self.acados_ocp.constraints.Jbx = np.eye(self.acados_ocp.dims.nx)
        self.acados_ocp.constraints.lbx = self.x_bound_min[:, 1]
        self.acados_ocp.constraints.ubx = self.x_bound_max[:, 1]
        self.acados_ocp.constraints.idxbx = np.array(range(self.acados_ocp.dims.nx))
        self.acados_ocp.dims.nbx = self.acados_ocp.dims.nx

        # setup terminal state constraints
        self.acados_ocp.constraints.Jbx_e = np.eye(self.acados_ocp.dims.nx)
        self.acados_ocp.constraints.lbx_e = self.x_bound_min[:, -1]
        self.acados_ocp.constraints.ubx_e = self.x_bound_max[:, -1]
        self.acados_ocp.constraints.idxbx_e = np.array(range(self.acados_ocp.dims.nx))
        self.acados_ocp.dims.nbx_e = self.acados_ocp.dims.nx

        # setup algebraic constraint
        self.acados_ocp.constraints.lh = np.array(self.all_g_bounds.min[:, 0])
        self.acados_ocp.constraints.uh = np.array(self.all_g_bounds.max[:, 0])

        # setup terminal algebraic constraint
        self.acados_ocp.constraints.lh_e = np.array(self.end_g_bounds.min[:, 0])
        self.acados_ocp.constraints.uh_e = np.array(self.end_g_bounds.max[:, 0])

    def __set_cost_type(self, cost_type: str = "NONLINEAR_LS"):
        """
        Set the type of cost functions

        Parameters
        ----------
        cost_type: str
            The type of cost function
        """

        self.acados_ocp.cost.cost_type = cost_type
        self.acados_ocp.cost.cost_type_e = cost_type
        self.acados_ocp.cost.cost_type_0 = cost_type

    def __set_costs(self, ocp):
        """
        Set the cost functions from ocp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        def add_linear_ls_lagrange(acados, objectives):
            def add_objective(n_variables, is_state):
                v_var = np.zeros(n_variables)
                var_type = acados.ocp.nlp[0].states if is_state else acados.ocp.nlp[0].controls
                rows = objectives.rows + var_type[objectives.extra_parameters["key"]].index[0]
                v_var[rows] = 1.0
                if is_state:
                    acados.Vx = np.vstack((acados.Vx, np.diag(v_var)))
                    acados.Vu = np.vstack((acados.Vu, np.zeros((n_states, n_controls))))
                else:
                    acados.Vx = np.vstack((acados.Vx, np.zeros((n_controls, n_states))))
                    acados.Vu = np.vstack((acados.Vu, np.diag(v_var)))
                acados.W = linalg.block_diag(acados.W, np.diag([objectives.weight] * n_variables))

                node_idx = objectives.node_idx[:-1] if objectives.node[0] == Node.ALL else objectives.node_idx

                y_ref = [np.zeros((n_states if is_state else n_controls, 1)) for _ in node_idx]
                if objectives.target is not None:
                    for idx in node_idx:
                        y_ref[idx][rows] = objectives.target[..., idx].T.reshape((-1, 1))
                acados.y_ref.append(y_ref)

            if objectives.type in allowed_control_objectives:
                add_objective(n_controls, False)
            elif objectives.type in allowed_state_objectives:
                add_objective(n_states, True)
            else:
                raise RuntimeError(
                    f"{objectives[0]['objective'].type.name} is an incompatible objective term with LINEAR_LS cost type"
                )

        def add_linear_ls_mayer(acados, objectives):
            def add_objective(n_variables, is_state):
                def _adjust_dim():
                    v_var = np.zeros(n_variables)
                    var_type = acados.ocp.nlp[0].states if is_state else acados.ocp.nlp[0].controls
                    rows = objectives.rows + var_type[objectives.extra_parameters["key"]].index[0]
                    v_var[rows] = 1.0
                    return v_var, rows

                if objectives.node[0] not in [Node.INTERMEDIATES, Node.PENULTIMATE, Node.END]:
                    v_var, rows = _adjust_dim()
                    if is_state:
                        acados.Vx0 = np.vstack((acados.Vx0, np.diag(v_var)))
                        acados.Vu0 = np.vstack((acados.Vu0, np.zeros((n_states, n_controls))))
                    else:
                        acados.Vx0 = np.vstack((acados.Vx0, np.zeros((n_controls, n_states))))
                        acados.Vu0 = np.vstack((acados.Vu0, np.diag(v_var)))
                    acados.W_0 = linalg.block_diag(acados.W_0, np.diag([objectives.weight] * n_variables))
                    y_ref_start = np.zeros((n_variables, 1))
                    if objectives.target is not None:
                        y_ref_start[rows] = objectives.target[..., 0].T.reshape((-1, 1))
                    acados.y_ref_start.append(y_ref_start)

                if objectives.node[0] in [Node.END, Node.ALL]:
                    v_var, rows = _adjust_dim()
                    if not is_state:
                        raise RuntimeError("Mayer objective at final node for controls is not defined.")
                    acados.Vxe = np.vstack((acados.Vxe, np.diag(v_var)))
                    acados.W_e = linalg.block_diag(acados.W_e, np.diag([objectives.weight] * n_states))
                    y_ref_end = np.zeros((n_states, 1))
                    if objectives.target is not None:
                        y_ref_end[rows] = objectives.target[..., -1].T.reshape((-1, 1))
                    acados.y_ref_end.append(y_ref_end)

            if objectives.type in allowed_control_objectives:
                add_objective(n_controls, False)
            elif objectives.type in allowed_state_objectives:
                add_objective(n_states, True)
            else:
                raise RuntimeError(f"{objectives.type.name} is an incompatible objective term with LINEAR_LS cost type")

        def add_nonlinear_ls_lagrange(acados, objectives, t, dt, x, u, p, a):
            if objectives.function[0].size_in("x")[0] == x.shape[0] * 2:
                x = vertcat(x, x)
            if objectives.function[0].size_in("u")[0] == u.shape[0] * 2:
                u = vertcat(u, u)

            acados.lagrange_costs = vertcat(
                acados.lagrange_costs, objectives.function[0](t, dt, x, u, p, a).reshape((-1, 1))
            )
            acados.W = linalg.block_diag(acados.W, np.diag([objectives.weight] * objectives.function[0].numel_out()))

            node_idx = objectives.node_idx[:-1] if objectives.node[0] == Node.ALL else objectives.node_idx
            if objectives.target is not None:
                acados.y_ref.append([objectives.target[..., idx].T.reshape((-1, 1)) for idx in node_idx])
            else:
                acados.y_ref.append([np.zeros((objectives.function[0].numel_out(), 1)) for _ in node_idx])

        def add_nonlinear_ls_mayer(acados, objectives, t, dt, x, u, p, a, node=None):
            if objectives.node[0] not in [Node.INTERMEDIATES, Node.PENULTIMATE, Node.END]:
                acados.W_0 = linalg.block_diag(
                    acados.W_0, np.diag([objectives.weight] * objectives.function[0].numel_out())
                )

                x_tp = x
                u_tp = u
                if objectives.function[0].size_in("x")[0] == x_tp.shape[0] * 2:
                    x_tp = vertcat(x_tp, x_tp)
                if objectives.function[0].size_in("u")[0] == u_tp.shape[0] * 2:
                    u_tp = vertcat(u_tp, u_tp)

                x_tp = x_tp if objectives.function[0].size_in("x") != (0, 0) else []
                u_tp = u_tp if objectives.function[0].size_in("u") != (0, 0) else []

                acados.mayer_costs = vertcat(
                    acados.mayer_costs, objectives.function[0](t, dt, x_tp, u_tp, p, a).reshape((-1, 1))
                )

                if objectives.target is not None:
                    acados.y_ref_start.append(objectives.target[..., 0].T.reshape((-1, 1)))
                else:
                    acados.y_ref_start.append(np.zeros((objectives.function[0].numel_out(), 1)))

            if objectives.node[0] in [Node.END, Node.ALL]:
                acados.W_e = linalg.block_diag(
                    acados.W_e, np.diag([objectives.weight] * objectives.function[-1].numel_out())
                )
                x_tp = x
                u_tp = u
                if objectives.function[-1].size_in("x")[0] == x_tp.shape[0] * 2:
                    x_tp = vertcat(x_tp, x_tp)
                if objectives.function[-1].size_in("u")[0] == u_tp.shape[0] * 2:
                    u_tp = vertcat(u_tp, u_tp)

                x_tp = x_tp if objectives.function[-1].size_in("x") != (0, 0) else []
                u_tp = u_tp if objectives.function[-1].size_in("u") != (0, 0) else []

                acados.mayer_costs_e = vertcat(
                    acados.mayer_costs_e, objectives.function[-1](t, dt, x_tp, u_tp, p, a).reshape((-1, 1))
                )

                if objectives.target is not None:
                    acados.y_ref_end.append(objectives.target[..., -1].T.reshape((-1, 1)))
                else:
                    acados.y_ref_end.append(np.zeros((objectives.function[-1].numel_out(), 1)))

        if ocp.n_phases != 1:
            raise NotImplementedError("ACADOS with more than one phase is not implemented yet.")
        # costs handling in self.acados_ocp
        self.y_ref = []
        self.y_ref_end = []
        self.y_ref_start = []
        self.lagrange_costs = SX()
        self.mayer_costs_e = SX()
        self.mayer_costs = SX()
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
        self.W_0 = np.zeros((0, 0))
        allowed_control_objectives = [ObjectiveFcn.Lagrange.MINIMIZE_CONTROL]
        allowed_state_objectives = [ObjectiveFcn.Lagrange.MINIMIZE_STATE, ObjectiveFcn.Mayer.TRACK_STATE]

        if self.acados_ocp.cost.cost_type == "LINEAR_LS":
            n_states = ocp.nlp[0].states.shape
            n_controls = ocp.nlp[0].controls.shape
            self.Vu = np.array([], dtype=np.int64).reshape(0, n_controls)
            self.Vx = np.array([], dtype=np.int64).reshape(0, n_states)
            self.Vxe = np.array([], dtype=np.int64).reshape(0, n_states)
            self.Vx0 = np.array([], dtype=np.int64).reshape(0, n_states)
            self.Vu0 = np.array([], dtype=np.int64).reshape(0, n_controls)
            for i in range(ocp.n_phases):
                for J in ocp.nlp[i].J:
                    if not J:
                        continue

                    if J.multi_thread:
                        raise RuntimeError(
                            f"The objective function {J.name} was declared with multi_thread=True, "
                            f"but this is not possible to multi_thread objective function with ACADOS"
                        )

                    if J.type.get_type() == ObjectiveFunction.LagrangeFunction:
                        add_linear_ls_lagrange(self, J)

                        # Deal with first and last node
                        add_linear_ls_mayer(self, J)

                    elif J.type.get_type() == ObjectiveFunction.MayerFunction:
                        add_linear_ls_mayer(self, J)

                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

                if self.nparams:
                    raise RuntimeError("Params not yet handled with LINEAR_LS cost type")

            # Set costs
            self.acados_ocp.cost.Vx = self.Vx if self.Vx.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vu = self.Vu if self.Vu.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vu_0 = self.Vu0 if self.Vu0.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vx_e = self.Vxe if self.Vxe.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vx_0 = self.Vx0 if self.Vx0.shape[0] else np.zeros((0, 0))

            # Set dimensions
            self.acados_ocp.dims.ny = sum([len(data[0]) for data in self.y_ref])
            self.acados_ocp.dims.ny_e = sum([len(data) for data in self.y_ref_end])
            self.acados_ocp.dims.ny_0 = sum([len(data) for data in self.y_ref_start])

            # Set weight
            self.acados_ocp.cost.W = self.W
            self.acados_ocp.cost.W_e = self.W_e
            self.acados_ocp.cost.W_0 = self.W_0

            # Set target shape
            self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.dims.ny,))
            self.acados_ocp.cost.yref_e = np.zeros((self.acados_ocp.dims.ny_e,))
            self.acados_ocp.cost.yref_0 = np.zeros((self.acados_ocp.dims.ny_0,))

        elif self.acados_ocp.cost.cost_type == "NONLINEAR_LS":
            for i, nlp in enumerate(ocp.nlp):
                for j, J in enumerate(nlp.J):
                    if not J:
                        continue

                    if J.multi_thread:
                        raise RuntimeError(
                            f"The objective function {J.name} was declared with multi_thread=True, "
                            f"but this is not possible to multi_thread objective function with ACADOS"
                        )

                    if J.type.get_type() == ObjectiveFunction.LagrangeFunction:
                        add_nonlinear_ls_lagrange(
                            self,
                            J,
                            nlp.time_cx,
                            nlp.dt,
                            nlp.states.scaled.cx_start,
                            nlp.controls.scaled.cx_start,
                            nlp.parameters.scaled.cx,
                            nlp.algebraic_states.scaled.cx_start,
                        )

                        # Deal with first and last node
                        add_nonlinear_ls_mayer(
                            self,
                            J,
                            nlp.time_cx,
                            nlp.dt,
                            nlp.states.scaled.cx_start,
                            nlp.controls.scaled.cx_start,
                            nlp.parameters.scaled.cx,
                            nlp.algebraic_states.scaled.cx_start,
                        )

                    elif J.type.get_type() == ObjectiveFunction.MayerFunction:
                        add_nonlinear_ls_mayer(
                            self,
                            J,
                            nlp.time_cx,
                            nlp.dt,
                            nlp.states.scaled.cx_start,
                            nlp.controls.scaled.cx_start,
                            nlp.parameters.scaled.cx,
                            nlp.algebraic_states.scaled.cx_start,
                        )
                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

            # parameter as mayer function
            # IMPORTANT: it is considered that only parameters are stored in ocp.objectives, for now.
            if self.nparams:
                nlp = ocp.nlp[0]  # Assume 1 phase
                for j, J in enumerate(ocp.J):
                    J.node = [Node.END]
                    add_nonlinear_ls_mayer(
                        self,
                        J,
                        nlp.time_cx,
                        nlp.dt,
                        nlp.states.scaled.cx_start,
                        nlp.controls.scaled.cx_start,
                        nlp.parameters.scaled.cx,
                        nlp.algebraic_states.scaled.cx_start,
                    )

            # Set costs
            self.acados_ocp.model.cost_y_expr = (
                self.lagrange_costs.reshape((-1, 1)) if self.lagrange_costs.numel() else SX(1, 1)
            )
            self.acados_ocp.model.cost_y_expr_e = (
                self.mayer_costs_e.reshape((-1, 1)) if self.mayer_costs_e.numel() else SX(1, 1)
            )
            self.acados_ocp.model.cost_y_expr_0 = (
                self.mayer_costs.reshape((-1, 1)) if self.mayer_costs.numel() else SX(1, 1)
            )

            # Set dimensions
            self.acados_ocp.dims.ny = self.acados_ocp.model.cost_y_expr.shape[0]
            self.acados_ocp.dims.ny_e = self.acados_ocp.model.cost_y_expr_e.shape[0]
            self.acados_ocp.dims.ny_0 = self.acados_ocp.model.cost_y_expr_0.shape[0]

            # Set weight
            self.acados_ocp.cost.W = np.zeros((1, 1)) if self.W.shape == (0, 0) else self.W
            self.acados_ocp.cost.W_e = np.zeros((1, 1)) if self.W_e.shape == (0, 0) else self.W_e
            self.acados_ocp.cost.W_0 = np.zeros((1, 1)) if self.W_0.shape == (0, 0) else self.W_0

            # Set target shape
            self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.cost.W.shape[0],))
            self.acados_ocp.cost.yref_e = np.zeros((self.acados_ocp.cost.W_e.shape[0],))
            self.acados_ocp.cost.yref_0 = np.zeros((self.acados_ocp.cost.W_0.shape[0],))

        elif self.acados_ocp.cost.cost_type == "EXTERNAL":
            raise RuntimeError("EXTERNAL is not interfaced yet, please use NONLINEAR_LS")

        else:
            raise RuntimeError("Available acados cost type: 'LINEAR_LS', 'NONLINEAR_LS' and 'EXTERNAL'.")

    def __update_solver(self):
        """
        Update the ACADOS solver to new values
        """

        param_init = []
        for key in self.ocp.nlp[0].parameters.keys():
            scale_init = self.ocp.parameter_init[key].scale(self.ocp.parameters[key].scaling.scaling)
            param_init = np.concatenate((param_init, scale_init.init[:, 0]))

        for n in range(self.acados_ocp.dims.N):
            if n == 0:
                # Initial node
                if self.y_ref_start:
                    if len(self.y_ref_start) == 1:
                        self.ocp_solver.cost_set(0, "yref", np.array(self.y_ref_start[0])[:, 0])
                    else:
                        self.ocp_solver.cost_set(0, "yref", np.concatenate(self.y_ref_start)[:, 0])
                self.ocp_solver.constraints_set(0, "lbx", self.x_bound_min[:, 0])
                self.ocp_solver.constraints_set(0, "ubx", self.x_bound_max[:, 0])
            else:
                if self.y_ref:  # Target
                    self.ocp_solver.cost_set(n, "yref", np.vstack([data[n] for data in self.y_ref])[:, 0])
                self.ocp_solver.constraints_set(n, "lbx", self.x_bound_min[:, 1])
                self.ocp_solver.constraints_set(n, "ubx", self.x_bound_max[:, 1])

            # Intermediates
            # check following line
            # self.ocp_solver.cost_set(n, "W", self.W)

            # The x_init need to be ordered by index that's why we use a for loop
            x_init = np.ndarray((self.ocp.nlp[0].states.shape))
            for key in self.ocp.nlp[0].states.keys():
                index = self.ocp.nlp[0].states[key].index
                self.ocp.nlp[0].x_init[key].check_and_adjust_dimensions(
                    self.ocp.nlp[0].states[key].shape, self.ocp.nlp[0].ns
                )
                x_init[index] = (
                    self.ocp.nlp[0].x_init[key].init.evaluate_at(n) / self.ocp.nlp[0].x_scaling[key].scaling[:, 0]
                )

            self.ocp_solver.set(n, "x", np.concatenate((param_init, x_init)))

            # The u_init need to be ordered by index that's why we use a for loop
            u_init = np.ndarray((self.acados_ocp.dims.nu, 1))
            for key in self.ocp.nlp[0].controls.keys():
                index = self.ocp.nlp[0].controls[key].index
                self.ocp.nlp[0].u_init[key].check_and_adjust_dimensions(
                    self.ocp.nlp[0].controls[key].shape, self.ocp.nlp[0].ns - 1
                )
                u_init[index, 0] = (
                    self.ocp.nlp[0].u_init[key].init.evaluate_at(n) / self.ocp.nlp[0].u_scaling[key].scaling[:, 0]
                )

            self.ocp_solver.set(n, "u", u_init)

            # The u_bounds need to be ordered by index that's why we use a for loop
            u_bounds_max = np.ndarray(self.acados_ocp.dims.nu)
            u_bounds_min = np.ndarray(self.acados_ocp.dims.nu)
            for key in self.ocp.nlp[0].controls.keys():
                u_tp = self.ocp.nlp[0].u_bounds[key]
                index = self.ocp.nlp[0].controls[key].index
                u_bounds_max[index] = np.array(u_tp.max[:, 0])
                u_bounds_min[index] = np.array(u_tp.min[:, 0])

            self.ocp_solver.constraints_set(n, "lbu", u_bounds_min)
            self.ocp_solver.constraints_set(n, "ubu", u_bounds_max)
            self.ocp_solver.constraints_set(n, "uh", self.all_g_bounds.max[:, 0])
            self.ocp_solver.constraints_set(n, "lh", self.all_g_bounds.min[:, 0])

        # Final
        if self.y_ref_end:
            if len(self.y_ref_end) == 1:
                self.ocp_solver.cost_set(self.acados_ocp.dims.N, "yref", np.array(self.y_ref_end[0])[:, 0])
            else:
                self.ocp_solver.cost_set(self.acados_ocp.dims.N, "yref", np.concatenate(self.y_ref_end)[:, 0])
            # check following line
            # self.ocp_solver.cost_set(self.acados_ocp.dims.N, "W", self.W_e)
        self.ocp_solver.constraints_set(self.acados_ocp.dims.N, "lbx", self.x_bound_min[:, -1])
        self.ocp_solver.constraints_set(self.acados_ocp.dims.N, "ubx", self.x_bound_max[:, -1])

        if len(self.end_g_bounds.max[:, 0]):
            self.ocp_solver.constraints_set(self.acados_ocp.dims.N, "uh", self.end_g_bounds.max[:, 0])
            self.ocp_solver.constraints_set(self.acados_ocp.dims.N, "lh", self.end_g_bounds.min[:, 0])

        # The x_init need to be ordered by index that's why we use a for loop
        x_init = np.ndarray((self.ocp.nlp[0].states.shape,))
        for key in self.ocp.nlp[0].states.keys():
            index = self.ocp.nlp[0].states[key].index
            x_init[index] = self.ocp.nlp[0].x_init[key].init.evaluate_at(self.acados_ocp.dims.N)

        self.ocp_solver.set(self.acados_ocp.dims.N, "x", np.concatenate((param_init, x_init)))

    def online_optim(self, ocp):
        raise NotImplementedError("online_optim is not implemented yet with ACADOS backend")

    def get_optimized_value(self) -> list | dict:
        """
        Get the previously optimized solution

        Returns
        -------
        A solution or a list of solution depending on the number of phases
        """

        ns = self.acados_ocp.dims.N
        n_params = self.ocp.nlp[0].parameters.shape
        acados_x = np.array([self.ocp_solver.get(i, "x") for i in range(ns + 1)]).T
        acados_p = acados_x[:n_params, :]
        acados_x = acados_x[n_params:, :]
        acados_u = np.array([self.ocp_solver.get(i, "u") for i in range(ns)]).T

        out = {
            "x": [],
            "u": acados_u,
            "solver_time_to_optimize": self.ocp_solver.get_stats("time_tot")[0],
            "real_time_to_optimize": self.real_time_to_optimize,
            "iter": self.ocp_solver.get_stats("sqp_iter")[0],
            "status": self.status,
            "solver": SolverType.ACADOS,
        }

        out["x"] = vertcat(out["x"], acados_x.reshape(-1, 1, order="F"))
        out["x"] = vertcat(out["x"], acados_u.reshape(-1, 1, order="F"))
        out["x"] = vertcat(out["x"], acados_p[:, 0])

        # Add dt to solution
        dt_init = self.ocp.dt_parameter_initial_guess.init[0, 0]
        dt = Function("dt", [self.ocp.nlp[0].dt], [self.ocp.nlp[0].dt])(dt_init)
        out["x"] = vertcat(dt, out["x"])

        self.out["sol"] = out
        out = []
        for key in self.out.keys():
            out.append(self.out[key])

        return out[0] if len(out) == 1 else out

    def solve(self, expand_during_shake_tree=False) -> list | dict:
        """
        Solve the prepared ocp

        Parameters
        ----------
        expand_during_shake_tree: bool
            If the casadi graph should be expanded during the shake tree phase. This value is ignored for ACADOS

        Returns
        -------
        A reference to the solution
        """

        tic = perf_counter()
        # Populate costs and constraints vectors
        self.__set_costs(self.ocp)
        self.__set_constraints(self.ocp)

        options = self.opts.as_dict(self)
        if self.ocp_solver is None:
            for key in options:
                setattr(self.acados_ocp.solver_options, key, options[key])
            self.ocp_solver = AcadosOcpSolver(self.acados_ocp, json_file="acados_ocp.json", build=self.opts.c_compile)
            self.opts.set_only_first_options_has_changed(False)
            self.opts.set_has_tolerance_changed(False)

        else:
            if self.opts.only_first_options_has_changed:
                raise RuntimeError(
                    "Some options has been changed the second time acados was run.",
                    "Only " + str(Solver.ACADOS.get_tolerance_keys()) + " can be modified.",
                )

            if self.opts.has_tolerance_changed:
                for key in self.opts.get_tolerance_keys():
                    short_key = key[12:]
                    self.ocp_solver.options_set(short_key, options[key[1:]])
                self.opts.set_has_tolerance_changed(False)

        self.__update_solver()
        self.status = self.ocp_solver.solve()
        self.real_time_to_optimize = perf_counter() - tic
        return self.get_optimized_value()
