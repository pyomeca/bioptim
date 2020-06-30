import os

import numpy as np
from scipy import linalg
from casadi import SX, vertcat
from acados_template import AcadosModel, AcadosOcp

from .objective_functions import ObjectiveFunction
from .solver_interface import SolverInterface


class AcadosInterface(SolverInterface):
    def __init__(self, ocp, **solver_options):
        raise NotImplementedError("ACADOS backend is not implemented yet")

        if not isinstance(ocp.CX(), SX):
            raise RuntimeError("CasADi graph must be SX to be solved with ACADOS")

        # TODO: Remove this part when it is solved
        if "acados_dir" in solver_options:
            os.environ["ACADOS_SOURCE_DIR"] = solver_options["acados_dir"]

        super().__init__()

        self.acados_ocp = AcadosOcp()
        self.acados_model = AcadosModel()
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

        # set cost module
        self.acados_ocp.cost.cost_type = "EXTERNAL"
        self.acados_ocp.cost.cost_type_e = "EXTERNAL"

        if self.acados_ocp.cost.cost_type != self.acados_ocp.cost.cost_type_e:
            raise NotImplementedError(
                "Different cost types for Lagrange and Mayer terms in Acados not implemented yet."
            )

        # set weight for states and controls (default: 1.00)
        Q = 1.00 * np.eye(self.acados_ocp.dims.nx)
        R = 1.00 * np.eye(self.acados_ocp.dims.nu)

        self.acados_ocp.cost.W = linalg.block_diag(Q, R)

        self.acados_ocp.cost.W_e = Q

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
            # TODO: change cost_expr_ext_cost and cost_expr_ext_cost_e
            raise NotImplementedError("NONLINEAR_LS cost type not implemented yet in acados.")

            self.acados_ocp.model.cost_expr_ext_cost = SX(0, 0)
            self.acados_ocp.model.cost_expr_ext_cost_e = SX(0, 0)

            k = 0
            for i in range(ocp.nb_phases):
                for j in range(len(ocp.nlp[i]["J"])):
                    if (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.LagrangeFunction
                    ):
                        # set Lagrange term
                        if self.acados_ocp.model.cost_expr_ext_cost.shape == (0, 0):
                            self.acados_ocp.model.cost_expr_ext_cost = ocp.nlp[i]["J"][j][0]
                        else:
                            self.acados_ocp.model.cost_expr_ext_cost += ocp.nlp[i]["J"][j][0]
                    elif (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.MayerFunction
                    ):
                        # set Mayer term
                        if self.acados_ocp.model.cost_expr_ext_cost_e.shape == (0, 0):
                            self.acados_ocp.model.cost_expr_ext_cost_e = ocp.nlp[i]["J_acados_mayer"][k][0]
                            k += 1
                        else:
                            self.acados_ocp.model.cost_expr_ext_cost_e += ocp.nlp[i]["J_acados_mayer"][k][0]
                            k += 1
                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

        elif self.acados_ocp.cost.cost_type == "EXTERNAL":
            self.acados_ocp.model.cost_expr_ext_cost = SX(0, 0)
            self.acados_ocp.model.cost_expr_ext_cost_e = SX(0, 0)

            k = 0
            for i in range(ocp.nb_phases):
                for j in range(len(ocp.nlp[i]["J"])):
                    if (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.LagrangeFunction
                    ):
                        # set Lagrange term
                        if self.acados_ocp.model.cost_expr_ext_cost.shape == (0, 0):
                            self.acados_ocp.model.cost_expr_ext_cost = ocp.nlp[i]["J"][j][0]
                        else:
                            self.acados_ocp.model.cost_expr_ext_cost += ocp.nlp[i]["J"][j][0]
                    elif (
                        ocp.original_values["objective_functions"][i][j]["type"]._get_type()
                        == ObjectiveFunction.MayerFunction
                    ):
                        # set Mayer term
                        if self.acados_ocp.model.cost_expr_ext_cost_e.shape == (0, 0):
                            self.acados_ocp.model.cost_expr_ext_cost_e = ocp.nlp[i]["J_acados_mayer"][k][0]
                            k += 1
                        else:
                            self.acados_ocp.model.cost_expr_ext_cost_e += ocp.nlp[i]["J_acados_mayer"][k][0]
                            k += 1
                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

        else:
            raise RuntimeError("Available acados cost type: 'LINEAR_LS' and 'EXTERNAL'.")

        # set y values
        self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.dims.ny,))
        self.acados_ocp.cost.yref_e = np.ones((self.acados_ocp.dims.ny_e,))

        for i in range(ocp.nb_phases):
            # set constraints
            for j in range(-1, 0):
                for k in range(ocp.nlp[i]["nx"]):
                    if ocp.nlp[i]["X_bounds"].min[k, j] != ocp.nlp[i]["X_bounds"].max[k, j]:
                        raise RuntimeError("The initial values must be set and fixed.")

            self.acados_ocp.constraints.x0 = np.array(ocp.nlp[i]["X_bounds"].min[:, 0])
            self.acados_ocp.dims.nbx_0 = self.acados_ocp.dims.nx
            self.acados_ocp.constraints.constr_type = "BGH"
            self.acados_ocp.constraints.lbu = np.array(ocp.nlp[i]["U_bounds"].min[:, 0])
            self.acados_ocp.constraints.ubu = np.array(ocp.nlp[i]["U_bounds"].max[:, 0])
            self.acados_ocp.constraints.idxbu = np.array(range(self.acados_ocp.dims.nu))
            self.acados_ocp.dims.nbu = self.acados_ocp.dims.nu

            # set control constraints
            self.acados_ocp.constraints.Jbx_e = np.eye(self.acados_ocp.dims.nx)
            self.acados_ocp.constraints.ubx_e = np.array(ocp.nlp[i]["X_bounds"].max[:, -1])
            self.acados_ocp.constraints.lbx_e = np.array(ocp.nlp[i]["X_bounds"].min[:, -1])
            self.acados_ocp.constraints.idxbx_e = np.array(range(self.acados_ocp.dims.nx))
            self.acados_ocp.dims.nbx_e = self.acados_ocp.dims.nx

        return self.acados_ocp

    def configure(self, options):
        # TODO: Removed this when it is managed properly
        if "acados_dir" in options:
            del options["acados_dir"]

        self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        self.acados_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.acados_ocp.solver_options.integrator_type = "ERK"
        self.acados_ocp.solver_options.nlp_solver_type = "SQP"

        self.acados_ocp.solver_options.nlp_solver_tol_comp = 1e-02
        self.acados_ocp.solver_options.nlp_solver_tol_eq = 1e-02
        self.acados_ocp.solver_options.nlp_solver_tol_ineq = 1e-02
        self.acados_ocp.solver_options.nlp_solver_tol_stat = 1e-02
        self.acados_ocp.solver_options.sim_method_newton_iter = 5
        self.acados_ocp.solver_options.sim_method_num_stages = 4
        self.acados_ocp.solver_options.sim_method_num_steps = 10
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

    def solve(self):
        from acados_template import AcadosOcpSolver

        self.ocp_solver = AcadosOcpSolver(self.acados_ocp, json_file="acados_ocp.json")
        self.ocp_solver.solve()
