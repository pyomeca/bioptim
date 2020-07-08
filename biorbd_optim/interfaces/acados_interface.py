import os

import numpy as np
from scipy import linalg
from casadi import SX, vertcat, sum1, Function
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from .solver_interface import SolverInterface
from ..limits.objective_functions import ObjectiveFunction


class AcadosInterface(SolverInterface):
    def __init__(self, ocp, **solver_options):
        if not isinstance(ocp.CX(), SX):
            raise RuntimeError("CasADi graph must be SX to be solved with ACADOS")
        super().__init__()

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

        self.lagrange_costs = SX()
        self.mayer_costs = SX()
        self.dtts = []
        self.dtts_e = []
        self.__acados_export_model(ocp)
        self.__prepare_acados(ocp)
        self.ocp_solver = None

    def __acados_export_model(self, ocp):
        # Declare model variables
        x = ocp.nlp[0]["X"][0]
        u = ocp.nlp[0]["U"][0]
        p = ocp.nlp[0]["p"]
        mod = ocp.nlp[0]["model"]
        x_dot = SX.sym("x_dot", mod.nbQdot() * 2, 1)

        f_expl = ocp.nlp[0]["dynamics_func"](x, u, p)
        f_impl = x_dot - f_expl

        self.acados_model.f_impl_expr = f_impl
        self.acados_model.f_expl_expr = f_expl
        self.acados_model.x = x
        self.acados_model.xdot = x_dot
        self.acados_model.u = u
        self.acados_model.p = []
        self.acados_model.name = "model_name"

    def __prepare_acados(self, ocp):
        if ocp.nb_phases > 1:
            raise NotImplementedError("more than 1 phase is not implemented yet with ACADOS backend")

        # set model
        self.acados_ocp.model = self.acados_model

        # set dimensions
        for i in range(ocp.nb_phases):
            # set time
            self.acados_ocp.solver_options.tf = ocp.nlp[i]["tf"]
            # set dimensions
            self.acados_ocp.dims.nx = ocp.nlp[i]["nx"]
            self.acados_ocp.dims.nu = ocp.nlp[i]["nu"]
            self.acados_ocp.dims.ny = self.acados_ocp.dims.nx + self.acados_ocp.dims.nu
            self.acados_ocp.dims.ny_e = ocp.nlp[i]["nx"]
            self.acados_ocp.dims.N = ocp.nlp[i]["ns"]

        for i in range(ocp.nb_phases):
            # set constraints
            for j in range(ocp.nlp[i]["nx"]):
                if ocp.nlp[i]["X_bounds"].min[j, 0] == -np.inf and ocp.nlp[i]["X_bounds"].max[j, 0] == np.inf:
                    pass
                elif ocp.nlp[i]["X_bounds"].min[j, 0] != ocp.nlp[i]["X_bounds"].max[j, 0]:
                    raise RuntimeError(
                        "Initial constraint on state must be hard. Hint: you can pass it as an objective"
                    )
                else:
                    self.acados_ocp.constraints.x0 = np.array(ocp.nlp[i]["X_bounds"].min[:, 0])
                    self.acados_ocp.dims.nbx_0 = self.acados_ocp.dims.nx
            self.acados_ocp.constraints.constr_type = "BGH"
            self.acados_ocp.constraints.lbu = np.array(ocp.nlp[i]["U_bounds"].min[:, 0])
            self.acados_ocp.constraints.ubu = np.array(ocp.nlp[i]["U_bounds"].max[:, 0])
            self.acados_ocp.constraints.idxbu = np.array(range(self.acados_ocp.dims.nu))
            self.acados_ocp.dims.nbu = self.acados_ocp.dims.nu

            # set control constraints
            # self.acados_ocp.constraints.Jbx_e = np.eye(self.acados_ocp.dims.nx)
            # self.acados_ocp.constraints.ubx_e = np.array(ocp.nlp[i]["X_bounds"].max[:, -1])
            # self.acados_ocp.constraints.lbx_e = np.array(ocp.nlp[i]["X_bounds"].min[:, -1])
            # self.acados_ocp.constraints.idxbx_e = np.array(range(self.acados_ocp.dims.nx))
            # self.acados_ocp.dims.nbx_e = self.acados_ocp.dims.nx

        return self.acados_ocp

    def __set_cost_type(self, cost_type="NONLINEAR_LS"):
        self.acados_ocp.cost.cost_type = cost_type
        self.acados_ocp.cost.cost_type_e = cost_type

    def __set_costs(self, ocp):
        # set weight for states and controls (default: 1.00)
        # Q = 1.00 * np.eye(self.acados_ocp.dims.nx)
        # R = 1.00 * np.eye(self.acados_ocp.dims.nu)
        # self.acados_ocp.cost.W = linalg.block_diag(Q, R)
        # self.acados_ocp.cost.W_e = Q
        self.dtts = []
        self.dtts_e = []
        if self.acados_ocp.cost.cost_type == "LINEAR_LS":
            # set Lagrange terms
            self.acados_ocp.cost.Vx = np.zeros((self.acados_ocp.dims.ny, self.acados_ocp.dims.nx))
            self.acados_ocp.cost.Vx[: self.acados_ocp.dims.nx, :] = np.eye(self.acados_ocp.dims.nx)

            Vu = np.zeros((self.acados_ocp.dims.ny, self.acados_ocp.dims.nu))
            Vu[self.acados_ocp.dims.nx :, :] = np.eye(self.acados_ocp.dims.nu)
            self.acados_ocp.cost.Vu = Vu

            # set Mayer term
            self.acados_ocp.cost.Vx_e = np.zeros((self.acados_ocp.dims.nx, self.acados_ocp.dims.nx))

        elif self.acados_ocp.cost.cost_type == "NONLINEAR_LS":
            for i in range(ocp.nb_phases):
                for j in range(len(ocp.nlp[i]["J"])):
                    if (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.LagrangeFunction
                    ):
                        if "data_to_track" in ocp.original_values["objective_functions"][i][j]:
                            dtt = ocp.original_values["objective_functions"][i][j]["data_to_track"]
                            cost_numel = ocp.nlp[i]["J_wt_dtt"][j][0].numel()
                            self.lagrange_costs = vertcat(
                                self.lagrange_costs, ocp.nlp[i]["J_wt_dtt"][j][0].reshape((cost_numel, 1))
                            )
                            self.dtts.append(dtt.reshape(np.prod(dtt.shape[:-1]), dtt.shape[-1], order="F"))
                    elif (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.MayerFunction
                    ):
                        if "data_to_track" in ocp.original_values["objective_functions"][i][j]:
                            dtt = ocp.original_values["objective_functions"][i][j]["data_to_track"]
                            cost_numel = ocp.nlp[i]["J_wt_dtt"][j][0].numel()
                            tmp_mayer_func = Function(
                                f"cas_mayer_func_{j}",
                                [ocp.nlp[i]["X"][-1]],
                                [ocp.nlp[i]["J_wt_dtt"][j][0].reshape((cost_numel, 1))],
                            )
                            self.mayer_costs = vertcat(self.mayer_costs, tmp_mayer_func(ocp.nlp[i]["X"][0]))
                            dtts_e = np.vstack([dtts_e, dtt.reshape(cost_numel, 1)])
                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

            if self.lagrange_costs.numel():
                self.acados_ocp.model.cost_y_expr = self.lagrange_costs
            else:
                self.acados_ocp.model.cost_y_expr = SX(1, 1)
            if self.mayer_costs.numel():
                self.acados_ocp.model.cost_y_expr_e = self.mayer_costs
            else:
                self.acados_ocp.model.cost_y_expr_e = SX(1, 1)
            self.acados_ocp.dims.ny = self.acados_ocp.model.cost_y_expr.shape[0]
            self.acados_ocp.dims.ny_e = self.acados_ocp.model.cost_y_expr_e.shape[0]
            self.acados_ocp.cost.yref = np.zeros((max(self.acados_ocp.dims.ny, 1),))
            self.acados_ocp.cost.yref_e = np.zeros((max(self.acados_ocp.dims.ny_e, 1),))
            Q_ocp = np.zeros((15, 15))
            np.fill_diagonal(Q_ocp, 1000)
            R_ocp = np.zeros((4, 4))
            np.fill_diagonal(R_ocp, 100)
            self.acados_ocp.cost.W = linalg.block_diag(Q_ocp, R_ocp)
            self.acados_ocp.cost.W_e = np.zeros((1, 1))
            if len(self.dtts):
                self.dtts = np.vstack(self.dtts)
            else:
                self.dtts = np.zeros((1, 1))
            if len(self.dtts_e):
                self.dtts_e = np.vstack(self.dtts_e)
            else:
                self.dtts_e = np.zeros((1, 1))

        elif self.acados_ocp.cost.cost_type == "EXTERNAL":
            for i in range(ocp.nb_phases):
                for j in range(len(ocp.nlp[i]["J"])):
                    if (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.LagrangeFunction
                    ):
                        self.lagrange_costs = vertcat(self.lagrange_costs, ocp.nlp[i]["J"][j][0])
                    elif (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.MayerFunction
                    ):
                        tmp_mayer_func = Function(f"cas_mayer_func_{j}", [ocp.nlp[i]["X"][-1]], [ocp.nlp[i]["J"][j][0]])
                        self.mayer_costs = vertcat(self.mayer_costs, tmp_mayer_func(ocp.nlp[i]["X"][0]))
                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")
            self.acados_ocp.model.cost_expr_ext_cost = sum1(self.lagrange_costs)
            self.acados_ocp.model.cost_expr_ext_cost_e = sum1(self.mayer_costs)

        else:
            raise RuntimeError("Available acados cost type: 'LINEAR_LS', 'NONLINEAR_LS' and 'EXTERNAL'.")

        # set y values
        # self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.dims.ny,))
        # self.acados_ocp.cost.yref_e = np.ones((self.acados_ocp.dims.ny_e,))

    def configure(self, options):
        if "acados_dir" in options:
            del options["acados_dir"]
        if "cost_type" in options:
            del options["cost_type"]

        self.acados_ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"  # FULL_CONDENSING_QPOASES
        self.acados_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.acados_ocp.solver_options.integrator_type = "ERK"
        self.acados_ocp.solver_options.nlp_solver_type = "SQP"

        self.acados_ocp.solver_options.nlp_solver_tol_comp = 1e-02
        self.acados_ocp.solver_options.nlp_solver_tol_eq = 1e-02
        self.acados_ocp.solver_options.nlp_solver_tol_ineq = 1e-02
        self.acados_ocp.solver_options.nlp_solver_tol_stat = 1e-02
        self.acados_ocp.solver_options.sim_method_newton_iter = 5
        self.acados_ocp.solver_options.sim_method_num_stages = 4
        self.acados_ocp.solver_options.sim_method_num_steps = 1
        self.acados_ocp.solver_options.print_level = 1

        for key in options:
            setattr(self.acados_ocp.solver_options, key, options[key])

    def get_iterations(self):
        raise NotImplementedError("return_iterations is not implemented yet with ACADOS backend")

    def online_optim(self, ocp):
        raise NotImplementedError("online_optim is not implemented yet with ACADOS backend")

    def get_optimized_value(self, ocp):
        acados_x = np.array([self.ocp_solver.get(i, "x") for i in range(ocp.nlp[0]["ns"] + 1)]).T
        acados_q = acados_x[: ocp.nlp[0]["nu"], :]
        acados_qdot = acados_x[ocp.nlp[0]["nu"] :, :]
        acados_u = np.array([self.ocp_solver.get(i, "u") for i in range(ocp.nlp[0]["ns"])]).T

        out = {
            "qqdot": acados_x,
            "x": [],
            "u": acados_u,
            "time_tot": self.ocp_solver.get_stats("time_tot")[0],
        }
        for i in range(ocp.nlp[0]["ns"]):
            out["x"] = vertcat(out["x"], acados_q[:, i])
            out["x"] = vertcat(out["x"], acados_qdot[:, i])
            out["x"] = vertcat(out["x"], acados_u[:, i])

        out["x"] = vertcat(out["x"], acados_q[:, ocp.nlp[0]["ns"]])
        out["x"] = vertcat(out["x"], acados_qdot[:, ocp.nlp[0]["ns"]])

        return out

    def solve(self, ocp):
        # populate costs vectors
        self.__set_costs(ocp)
        if self.ocp_solver is None:
            self.ocp_solver = AcadosOcpSolver(self.acados_ocp, json_file="acados_ocp.json")
        for n in range(self.acados_ocp.dims.N):
            self.ocp_solver.cost_set(n, "y_ref", self.dtts[:, n].squeeze())
        self.ocp_solver.solve()
        return self
