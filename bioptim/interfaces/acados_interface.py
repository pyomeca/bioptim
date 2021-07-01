from typing import Union
from datetime import datetime

import numpy as np
from scipy import linalg
from casadi import SX, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from ..misc.enums import Node
from .solver_interface import SolverInterface
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
    configure(self, options: dict)
        Set some ACADOS options
    get_optimized_value(self) -> Union[list[dict], dict]
        Get the previously optimized solution
    solve(self) -> "AcadosInterface"
        Solve the prepared ocp
    """

    def __init__(self, ocp, **solver_options):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        solver_options: dict
            The options to pass to the solver
        """

        if not isinstance(ocp.cx(), SX):
            raise RuntimeError("CasADi graph must be SX to be solved with ACADOS. Please set use_sx to True in OCP")

        super().__init__(ocp)

        # If Acados is installed using the acados_install.sh file, you probably can leave this to unset
        acados_path = ""
        if "acados_dir" in solver_options:
            acados_path = solver_options["acados_dir"]
        self.acados_ocp = AcadosOcp(acados_path=acados_path)
        self.acados_model = AcadosModel()

        if "cost_type" in solver_options:
            self.__set_cost_type(solver_options["cost_type"])
        else:
            self.__set_cost_type()

        if "constr_type" in solver_options:
            self.__set_constr_type(solver_options["constr_type"])
        else:
            self.__set_constr_type()

        self.lagrange_costs = SX()
        self.mayer_costs = SX()
        self.y_ref = []
        self.y_ref_end = []
        self.nparams = 0
        self.params_initial_guess = None
        self.params_bounds = None
        self.__acados_export_model(ocp)
        self.__prepare_acados(ocp)
        self.ocp_solver = None
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
        self.status = None
        self.out = {}

        self.all_constr = None
        self.end_constr = SX()
        self.all_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        self.end_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        self.x_bound_max = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.x_bound_min = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.Vu = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].controls.shape)
        self.Vx = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)
        self.Vxe = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)

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
        x = ocp.nlp[0].states.cx
        u = ocp.nlp[0].controls.cx
        p = ocp.nlp[0].parameters.cx
        if ocp.v.parameters_in_list:
            for param in ocp.v.parameters_in_list:
                if str(param.cx)[:11] == f"time_phase_":
                    raise RuntimeError("Time constraint not implemented yet with Acados.")

        self.nparams = ocp.nlp[0].parameters.shape
        self.params_initial_guess = ocp.v.parameters_in_list.initial_guess
        self.params_initial_guess.check_and_adjust_dimensions(self.nparams, 1)
        self.params_bounds = ocp.v.parameters_in_list.bounds
        self.params_bounds.check_and_adjust_dimensions(self.nparams, 1)
        x = vertcat(p, x)
        x_dot = SX.sym("x_dot", x.shape[0], x.shape[1])

        f_expl = vertcat([0] * self.nparams, ocp.nlp[0].dynamics_func(x[self.nparams :, :], u, p))
        f_impl = x_dot - f_expl

        self.acados_model.f_impl_expr = f_impl
        self.acados_model.f_expl_expr = f_expl
        self.acados_model.x = x
        self.acados_model.xdot = x_dot
        self.acados_model.u = u
        self.acados_model.con_h_expr = np.zeros((0, 0))
        self.acados_model.con_h_expr_e = np.zeros((0, 0))
        self.acados_model.p = []
        now = datetime.now()  # current date and time
        self.acados_model.name = f"model_{now.strftime('%Y_%m_%d_%H%M%S%f')[:-4]}"

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
        self.acados_ocp.solver_options.tf = ocp.nlp[0].tf

        # set dimensions
        self.acados_ocp.dims.nx = ocp.nlp[0].states.shape + ocp.nlp[0].parameters.shape
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
        u_min = np.array(ocp.nlp[0].u_bounds.min)
        u_max = np.array(ocp.nlp[0].u_bounds.max)
        x_min = np.array(ocp.nlp[0].x_bounds.min)
        x_max = np.array(ocp.nlp[0].x_bounds.max)
        self.all_constr = SX()
        self.end_constr = SX()
        # TODO:change for more node flexibility on bounds
        self.all_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        self.end_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for i, nlp in enumerate(ocp.nlp):
            x = nlp.states.cx
            u = nlp.controls.cx
            p = nlp.parameters.cx

            for g, G in enumerate(nlp.g):
                if not G:
                    continue

                if G.node[0] == Node.ALL or G.node[0] == Node.ALL_SHOOTING:
                    self.all_constr = vertcat(self.all_constr, G.function(x, u, p))
                    self.all_g_bounds.concatenate(G.bounds)
                    if G.node[0] == Node.ALL:
                        self.end_constr = vertcat(self.end_constr, G.function(x, u, p))
                        self.end_g_bounds.concatenate(G.bounds)

                elif G.node[0] == Node.END:
                    self.end_constr = vertcat(self.end_constr, G.function(x, u, p))
                    self.end_g_bounds.concatenate(G.bounds)

                else:
                    raise RuntimeError(
                        "Except for states and controls, Acados solver only handles constraints on last or all nodes."
                    )

        self.acados_model.con_h_expr = self.all_constr
        self.acados_model.con_h_expr_e = self.end_constr

        if not np.all(np.all(u_min.T == u_min.T[0, :], axis=0)):
            raise NotImplementedError("u_bounds min must be the same at each shooting point with ACADOS")
        if not np.all(np.all(u_max.T == u_max.T[0, :], axis=0)):
            raise NotImplementedError("u_bounds max must be the same at each shooting point with ACADOS")

        if (
            not np.isfinite(u_min).all()
            or not np.isfinite(x_min).all()
            or not np.isfinite(u_max).all()
            or not np.isfinite(x_max).all()
        ):
            raise NotImplementedError(
                "u_bounds and x_bounds cannot be set to infinity in ACADOS. Consider changing it "
                "to a big value instead."
            )

        # setup state constraints
        # TODO replace all these np.concatenate by proper bound and initial_guess classes
        self.x_bound_max = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.x_bound_min = np.ndarray((self.acados_ocp.dims.nx, 3))
        param_bounds_max = []
        param_bounds_min = []

        if self.nparams:
            param_bounds_max = self.params_bounds.max[:, 0]
            param_bounds_min = self.params_bounds.min[:, 0]

        for i in range(3):
            self.x_bound_max[:, i] = np.concatenate((param_bounds_max, np.array(ocp.nlp[0].x_bounds.max[:, i])))
            self.x_bound_min[:, i] = np.concatenate((param_bounds_min, np.array(ocp.nlp[0].x_bounds.min[:, i])))

        # setup control constraints
        self.acados_ocp.constraints.lbu = np.array(ocp.nlp[0].u_bounds.min[:, 0])
        self.acados_ocp.constraints.ubu = np.array(ocp.nlp[0].u_bounds.max[:, 0])
        self.acados_ocp.constraints.idxbu = np.array(range(self.acados_ocp.dims.nu))
        self.acados_ocp.dims.nbu = self.acados_ocp.dims.nu

        # initial state constraints
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

    def __set_costs(self, ocp):
        """
        Set the cost functions from ocp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        def add_linear_ls_lagrange(acados, J):
            def add_objective(n_variables, is_state):
                v_var = np.zeros(n_variables)
                var_type = acados.ocp.nlp[0].states if is_state else acados.ocp.nlp[0].controls
                rows = J.rows + var_type[J.params["key"]].index[0]
                v_var[rows] = 1.0
                if is_state:
                    acados.Vx = np.vstack((acados.Vx, np.diag(v_var)))
                    acados.Vu = np.vstack((acados.Vu, np.zeros((n_states, n_controls))))
                else:
                    acados.Vx = np.vstack((acados.Vx, np.zeros((n_controls, n_states))))
                    acados.Vu = np.vstack((acados.Vu, np.diag(v_var)))
                acados.W = linalg.block_diag(acados.W, np.diag([J.weight] * n_variables))

                y_ref = [np.zeros((n_states if is_state else n_controls, 1)) for _ in J.node_idx]
                if J.target is not None:
                    for idx in J.node_idx:
                        y_ref[idx][rows] = J.target[:, idx]
                acados.y_ref.append(y_ref)

            if J.type in allowed_control_objectives:
                add_objective(n_controls, False)
            elif J.type in allowed_state_objectives:
                add_objective(n_states, True)
            else:
                raise RuntimeError(
                    f"{J[0]['objective'].type.name} is an incompatible objective term with LINEAR_LS cost type"
                )

        def add_linear_ls_mayer(acados, J):
            if J.type in allowed_state_objectives:
                vxe = np.zeros(n_states)
                rows = J.rows + acados.ocp.nlp[0].states[J.params["key"]].index[0]
                vxe[rows] = 1.0
                acados.Vxe = np.vstack((acados.Vxe, np.diag(vxe)))
                acados.W_e = linalg.block_diag(acados.W_e, np.diag([J.weight] * n_states))

                y_ref_end = np.zeros((n_states, 1))
                if J.target is not None:
                    y_ref_end[rows] = J.target[:, -1][:, np.newaxis]
                acados.y_ref_end.append(y_ref_end)

            else:
                raise RuntimeError(f"{J.type.name} is an incompatible objective term with LINEAR_LS cost type")

        def add_nonlinear_ls_mayer(acados, J, x, u, p):
            acados.W_e = linalg.block_diag(acados.W_e, np.diag([J.weight] * J.function.size1_out("o0")))
            x = x if J.function.sparsity_in("i0").shape != (0, 0) else []
            u = u if J.function.sparsity_in("i1").shape != (0, 0) else []
            acados.mayer_costs = vertcat(acados.mayer_costs, J.function(x, u, p))

            if J.target is not None:
                acados.y_ref_end.append(J.target[:, -1][:, np.newaxis])
            else:
                acados.y_ref_end.append(np.zeros(J.function.size_out("o0"))[:, np.newaxis])

        def add_nonlinear_ls_lagrange(acados, J, x, u, p):
            acados.lagrange_costs = vertcat(acados.lagrange_costs, J.function(x, u, p))
            acados.W = linalg.block_diag(acados.W, np.diag([J.weight] * J.function.size_out("o0")[0]))
            if J.target is not None:
                acados.y_ref.append([J.target[:, idx:idx+1] for idx in J.node_idx])
            else:
                acados.y_ref.append([np.zeros(J.function.size_out("o0")) for _ in J.node_idx])

        if ocp.n_phases != 1:
            raise NotImplementedError("ACADOS with more than one phase is not implemented yet.")
        # costs handling in self.acados_ocp
        self.y_ref = []
        self.y_ref_end = []
        self.lagrange_costs = SX()
        self.mayer_costs = SX()
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
        allowed_control_objectives = [ObjectiveFcn.Lagrange.MINIMIZE_CONTROL]
        allowed_state_objectives = [ObjectiveFcn.Lagrange.MINIMIZE_STATE, ObjectiveFcn.Mayer.TRACK_STATE]

        if self.acados_ocp.cost.cost_type == "LINEAR_LS":
            n_states = ocp.nlp[0].states.shape
            n_controls = ocp.nlp[0].controls.shape
            self.Vu = np.array([], dtype=np.int64).reshape(0, n_controls)
            self.Vx = np.array([], dtype=np.int64).reshape(0, n_states)
            self.Vxe = np.array([], dtype=np.int64).reshape(0, n_states)
            for i in range(ocp.n_phases):
                for J in ocp.nlp[i].J:
                    if not J:
                        continue

                    if J.type.get_type() == ObjectiveFunction.LagrangeFunction:
                        add_linear_ls_lagrange(self, J)

                        # Deal with last node to match ipopt formulation
                        if J.node[0] == Node.ALL:
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
            self.acados_ocp.cost.Vx_e = self.Vxe if self.Vxe.shape[0] else np.zeros((0, 0))

            # Set dimensions
            self.acados_ocp.dims.ny = sum([len(data[0]) for data in self.y_ref])
            self.acados_ocp.dims.ny_e = sum([len(data) for data in self.y_ref_end])

            # Set weight
            self.acados_ocp.cost.W = self.W
            self.acados_ocp.cost.W_e = self.W_e

            # Set target shape
            self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.cost.W.shape[0],))
            self.acados_ocp.cost.yref_e = np.zeros((self.acados_ocp.cost.W_e.shape[0],))

        elif self.acados_ocp.cost.cost_type == "NONLINEAR_LS":
            for i, nlp in enumerate(ocp.nlp):
                for j, J in enumerate(nlp.J):
                    if not J:
                        continue
                    if J.type.get_type() == ObjectiveFunction.LagrangeFunction:
                        add_nonlinear_ls_lagrange(self, J, nlp.states.cx, nlp.controls.cx, nlp.parameters.cx)

                        # Deal with last node to match ipopt formulation
                        if J.node[0] == Node.ALL:
                            add_nonlinear_ls_mayer(self, J, nlp.states.cx, nlp.controls.cx, nlp.parameters.cx)

                    elif J.type.get_type() == ObjectiveFunction.MayerFunction:
                        add_nonlinear_ls_mayer(self, J, nlp.states.cx, nlp.controls.cx, nlp.parameters.cx)

                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

            # parameter as mayer function
            # IMPORTANT: it is considered that only parameters are stored in ocp.J, for now.
            if self.nparams:
                nlp = ocp.nlp[0]  # Assume 1 phase
                for j, J in enumerate(ocp.J):
                    add_nonlinear_ls_mayer(self, J, nlp.states.cx, nlp.controls.cx, nlp.parameters.cx)

            # Set costs
            self.acados_ocp.model.cost_y_expr = self.lagrange_costs if self.lagrange_costs.numel() else SX(1, 1)
            self.acados_ocp.model.cost_y_expr_e = self.mayer_costs if self.mayer_costs.numel() else SX(1, 1)

            # Set dimensions
            self.acados_ocp.dims.ny = self.acados_ocp.model.cost_y_expr.shape[0]
            self.acados_ocp.dims.ny_e = self.acados_ocp.model.cost_y_expr_e.shape[0]

            # Set weight
            self.acados_ocp.cost.W = np.zeros((1, 1)) if self.W.shape == (0, 0) else self.W
            self.acados_ocp.cost.W_e = np.zeros((1, 1)) if self.W_e.shape == (0, 0) else self.W_e

            # Set target shape
            self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.cost.W.shape[0],))
            self.acados_ocp.cost.yref_e = np.zeros((self.acados_ocp.cost.W_e.shape[0],))

        elif self.acados_ocp.cost.cost_type == "EXTERNAL":
            raise RuntimeError("EXTERNAL is not interfaced yet, please use NONLINEAR_LS")

        else:
            raise RuntimeError("Available acados cost type: 'LINEAR_LS', 'NONLINEAR_LS' and 'EXTERNAL'.")

    def __update_solver(self):
        """
        Update the ACADOS solver to new values
        """

        param_init = []
        for n in range(self.acados_ocp.dims.N):
            if self.y_ref:  # Target
                self.ocp_solver.cost_set(n, "yref", np.vstack([data[n] for data in self.y_ref])[:, 0])
            # check following line
            # self.ocp_solver.cost_set(n, "W", self.W)

            if self.nparams:
                param_init = self.params_initial_guess.init.evaluate_at(n)

            self.ocp_solver.set(n, "x", np.concatenate((param_init, self.ocp.nlp[0].x_init.init.evaluate_at(n))))
            self.ocp_solver.set(n, "u", self.ocp.nlp[0].u_init.init.evaluate_at(n))
            self.ocp_solver.constraints_set(n, "lbu", self.ocp.nlp[0].u_bounds.min[:, 0])
            self.ocp_solver.constraints_set(n, "ubu", self.ocp.nlp[0].u_bounds.max[:, 0])
            self.ocp_solver.constraints_set(n, "uh", self.all_g_bounds.max[:, 0])
            self.ocp_solver.constraints_set(n, "lh", self.all_g_bounds.min[:, 0])

            if n == 0:
                self.ocp_solver.constraints_set(n, "lbx", self.x_bound_min[:, 0])
                self.ocp_solver.constraints_set(n, "ubx", self.x_bound_max[:, 0])
            else:
                self.ocp_solver.constraints_set(n, "lbx", self.x_bound_min[:, 1])
                self.ocp_solver.constraints_set(n, "ubx", self.x_bound_max[:, 1])

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

        if self.ocp.nlp[0].x_init.init.shape[1] == self.acados_ocp.dims.N + 1:
            if self.nparams:
                self.ocp_solver.set(
                    self.acados_ocp.dims.N,
                    "x",
                    np.concatenate(
                        (
                            self.params_initial_guess.init[:, 0],
                            self.ocp.nlp[0].x_init.init[:, self.acados_ocp.dims.N],
                        )
                    ),
                )
            else:
                self.ocp_solver.set(self.acados_ocp.dims.N, "x", self.ocp.nlp[0].x_init.init[:, self.acados_ocp.dims.N])

    def configure(self, options: dict):
        """
        Set some ACADOS options

        Parameters
        ----------
        options: dict
            The dictionary of options
        """
        if options is None:
            options = {}

        if "acados_dir" in options:
            del options["acados_dir"]
        if "cost_type" in options:
            del options["cost_type"]

        if self.ocp_solver is None:
            self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
            self.acados_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            self.acados_ocp.solver_options.integrator_type = "IRK"
            self.acados_ocp.solver_options.nlp_solver_type = "SQP"

            self.acados_ocp.solver_options.nlp_solver_tol_comp = 1e-06
            self.acados_ocp.solver_options.nlp_solver_tol_eq = 1e-06
            self.acados_ocp.solver_options.nlp_solver_tol_ineq = 1e-06
            self.acados_ocp.solver_options.nlp_solver_tol_stat = 1e-06
            self.acados_ocp.solver_options.nlp_solver_max_iter = 200
            self.acados_ocp.solver_options.sim_method_newton_iter = 5
            self.acados_ocp.solver_options.sim_method_num_stages = 4
            self.acados_ocp.solver_options.sim_method_num_steps = 1
            self.acados_ocp.solver_options.print_level = 1
            for key in options:
                setattr(self.acados_ocp.solver_options, key, options[key])
        else:
            available_options = [
                "nlp_solver_tol_comp",
                "nlp_solver_tol_eq",
                "nlp_solver_tol_ineq",
                "nlp_solver_tol_stat",
            ]
            for key in options:
                if key in available_options:
                    short_key = key[11:]
                    self.ocp_solver.options_set(short_key, options[key])
                else:
                    raise RuntimeError(
                        f"[ACADOS] Only editable solver options after solver creation are :\n {available_options}"
                    )

    def online_optim(self, ocp):
        raise NotImplementedError("online_optim is not implemented yet with ACADOS backend")

    def get_optimized_value(self) -> Union[list, dict]:
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
            "time_tot": self.ocp_solver.get_stats("time_tot")[0],
            "iter": self.ocp_solver.get_stats("sqp_iter")[0],
            "status": self.status,
        }

        out["x"] = vertcat(out["x"], acados_x.reshape(-1, 1, order="F"))
        out["x"] = vertcat(out["x"], acados_u.reshape(-1, 1, order="F"))
        out["x"] = vertcat(out["x"], acados_p[:, 0])

        self.out["sol"] = out
        out = []
        for key in self.out.keys():
            out.append(self.out[key])
        return out[0] if len(out) == 1 else out

    def solve(self) -> "AcadosInterface":
        """
        Solve the prepared ocp

        Returns
        -------
        A reference to the solution
        """

        # Populate costs and constraints vectors
        self.__set_costs(self.ocp)
        self.__set_constraints(self.ocp)
        if self.ocp_solver is None:
            self.ocp_solver = AcadosOcpSolver(self.acados_ocp, json_file="acados_ocp.json")
        self.__update_solver()

        self.status = self.ocp_solver.solve()
        self.get_optimized_value()
        return self
