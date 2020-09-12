import numpy as np
from scipy import linalg
from casadi import SX, vertcat, sum1, Function
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from .solver_interface import SolverInterface
from ..limits.objective_functions import ObjectiveFunction


class AcadosInterface(SolverInterface):
    def __init__(self, ocp, **solver_options):
        if not isinstance(ocp.CX(), SX):
            raise RuntimeError("CasADi graph must be SX to be solved with ACADOS. Please set use_SX to True in OCP")
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

        self.lagrange_costs = SX()
        self.mayer_costs = SX()
        self.y_ref = []
        self.y_ref_end = []
        self.__acados_export_model(ocp)
        self.__prepare_acados(ocp)
        self.ocp_solver = None
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
        self.out = {}

    def __acados_export_model(self, ocp):
        # Declare model variables
        x = ocp.nlp[0]["X"][0]
        u = ocp.nlp[0]["U"][0]
        p = ocp.nlp[0]["p"]
        x_dot = SX.sym("x_dot", x.shape[0], x.shape[1])

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
            raise NotImplementedError("More than 1 phase is not implemented yet with ACADOS backend")
        if ocp.param_to_optimize:
            raise NotImplementedError("Parameters optimization is not implemented yet with ACADOS")

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

            # control constraints
            self.acados_ocp.constraints.constr_type = "BGH"
            self.acados_ocp.constraints.lbu = np.array(ocp.nlp[i]["U_bounds"].min[:, 0])
            self.acados_ocp.constraints.ubu = np.array(ocp.nlp[i]["U_bounds"].max[:, 0])
            self.acados_ocp.constraints.idxbu = np.array(range(self.acados_ocp.dims.nu))
            self.acados_ocp.dims.nbu = self.acados_ocp.dims.nu

            # path constraints
            self.acados_ocp.constraints.Jbx = np.eye(self.acados_ocp.dims.nx)
            self.acados_ocp.constraints.ubx = np.array(ocp.nlp[i]["X_bounds"].max[:, 1])
            self.acados_ocp.constraints.lbx = np.array(ocp.nlp[i]["X_bounds"].min[:, 1])
            self.acados_ocp.constraints.idxbx = np.array(range(self.acados_ocp.dims.nx))
            self.acados_ocp.dims.nbx = self.acados_ocp.dims.nx

            # terminal constraints
            self.acados_ocp.constraints.Jbx_e = np.eye(self.acados_ocp.dims.nx)
            self.acados_ocp.constraints.ubx_e = np.array(ocp.nlp[i]["X_bounds"].max[:, -1])
            self.acados_ocp.constraints.lbx_e = np.array(ocp.nlp[i]["X_bounds"].min[:, -1])
            self.acados_ocp.constraints.idxbx_e = np.array(range(self.acados_ocp.dims.nx))
            self.acados_ocp.dims.nbx_e = self.acados_ocp.dims.nx

        return self.acados_ocp

    def __set_cost_type(self, cost_type="NONLINEAR_LS"):
        self.acados_ocp.cost.cost_type = cost_type
        self.acados_ocp.cost.cost_type_e = cost_type

    def __set_costs(self, ocp):
        self.y_ref = []
        self.y_ref_end = []
        self.lagrange_costs = SX()
        self.mayer_costs = SX()
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
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
            if ocp.nb_phases != 1:
                raise NotImplementedError("ACADOS with more than one phase is not implemented yet")

            for i in range(ocp.nb_phases):
                # TODO: I think ocp.J is missing here (the parameters would be stored there)
                # TODO: Yes the objectives in ocp are not dealt with,
                #  does that makes sense since we only work with 1 phase ?
                # TODO: If you use parameter, it is added to the main J
                for j, J in enumerate(ocp.nlp[i]["J"]):
                    if J[0]["objective"].type.get_type() == ObjectiveFunction.LagrangeFunction:
                        self.lagrange_costs = vertcat(self.lagrange_costs, J[0]["val"].reshape((-1, 1)))
                        self.W = linalg.block_diag(self.W, np.diag([J[0]["objective"].weight] * J[0]["val"].numel()))
                        if J[0]["target"] is not None:
                            self.y_ref.append([J_tp["target"].T.reshape((-1, 1)) for J_tp in J])
                        else:
                            self.y_ref.append([np.zeros((J_tp["val"].numel(), 1)) for J_tp in J])

                    elif J[0]["objective"].type.get_type() == ObjectiveFunction.MayerFunction:
                        mayer_func_tp = Function(f"cas_mayer_func_{i}_{j}", [ocp.nlp[i]["X"][-1]], [J[0]["val"]])
                        self.W_e = linalg.block_diag(
                            self.W_e, np.diag([J[0]["objective"].weight] * J[0]["val"].numel())
                        )
                        self.mayer_costs = vertcat(self.mayer_costs, mayer_func_tp(ocp.nlp[i]["X"][0]))
                        if J[0]["target"] is not None:
                            self.y_ref_end.append([J[0]["target"]])
                        else:
                            self.y_ref_end.append([np.zeros((J[0]["val"].numel(), 1))])

                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

            # Set costs
            self.acados_ocp.model.cost_y_expr = self.lagrange_costs if self.lagrange_costs.numel() else SX(1, 1)
            self.acados_ocp.model.cost_y_expr_e = self.mayer_costs if self.mayer_costs.numel() else SX(1, 1)

            # Set dimensions
            self.acados_ocp.dims.ny = self.acados_ocp.model.cost_y_expr.shape[0]
            self.acados_ocp.dims.ny_e = self.acados_ocp.model.cost_y_expr_e.shape[0]
            self.acados_ocp.cost.yref = np.zeros((max(self.acados_ocp.dims.ny, 1),))
            self.acados_ocp.cost.yref_e = (
                np.concatenate(self.y_ref_end, -1).T.squeeze() if len(self.y_ref_end) else np.zeros((1,))
            )

            # Set weight
            self.acados_ocp.cost.W = np.zeros((1, 1)) if self.W.shape == (0, 0) else self.W
            self.acados_ocp.cost.W_e = np.zeros((1, 1)) if self.W_e.shape == (0, 0) else self.W_e

        elif self.acados_ocp.cost.cost_type == "EXTERNAL":
            for i in range(ocp.nb_phases):
                for j in range(len(ocp.nlp[i]["J"])):
                    J = ocp.nlp[i]["J"][j][0]

                    raise RuntimeError("TODO: The target for EXTERNAL is not right currently")
                    if J["type"] == ObjectiveFunction.LagrangeFunction:
                        self.lagrange_costs = vertcat(self.lagrange_costs, J["val"][0] - J["target"][0])
                    elif J["type"] == ObjectiveFunction.MayerFunction:
                        raise RuntimeError("TODO: I may have broken this (is this the right J?)")
                        mayer_func_tp = Function(f"cas_mayer_func_{i}_{j}", [ocp.nlp[i]["X"][-1]], [J["val"]])
                        self.mayer_costs = vertcat(self.mayer_costs, mayer_func_tp(ocp.nlp[i]["X"][0]))
                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")
            self.acados_ocp.model.cost_expr_ext_cost = sum1(self.lagrange_costs)
            self.acados_ocp.model.cost_expr_ext_cost_e = sum1(self.mayer_costs)

        else:
            raise RuntimeError("Available acados cost type: 'LINEAR_LS', 'NONLINEAR_LS' and 'EXTERNAL'.")

    def configure(self, options):
        if "acados_dir" in options:
            del options["acados_dir"]
        if "cost_type" in options:
            del options["cost_type"]

        self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        self.acados_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.acados_ocp.solver_options.integrator_type = "ERK"
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

    def get_iterations(self):
        raise NotImplementedError("return_iterations is not implemented yet with ACADOS backend")

    def online_optim(self, ocp):
        raise NotImplementedError("online_optim is not implemented yet with ACADOS backend")

    def get_optimized_value(self):
        ns = self.acados_ocp.dims.N
        nu = self.acados_ocp.dims.nu
        acados_x = np.array([self.ocp_solver.get(i, "x") for i in range(ns + 1)]).T
        acados_q = acados_x[:nu, :]
        acados_qdot = acados_x[nu:, :]
        acados_u = np.array([self.ocp_solver.get(i, "u") for i in range(ns)]).T

        out = {
            "qqdot": acados_x,
            "x": [],
            "u": acados_u,
            "time_tot": self.ocp_solver.get_stats("time_tot")[0],
        }
        for i in range(ns):
            out["x"] = vertcat(out["x"], acados_q[:, i])
            out["x"] = vertcat(out["x"], acados_qdot[:, i])
            out["x"] = vertcat(out["x"], acados_u[:, i])

        out["x"] = vertcat(out["x"], acados_q[:, ns])
        out["x"] = vertcat(out["x"], acados_qdot[:, ns])
        self.out["sol"] = out
        out = []
        for key in self.out.keys():
            out.append(self.out[key])
        return out[0] if len(out) == 1 else out

    def solve(self):
        # Populate costs vectors
        self.__set_costs(self.ocp)
        if self.ocp_solver is None:
            self.ocp_solver = AcadosOcpSolver(self.acados_ocp, json_file="acados_ocp.json")
        for n in range(self.acados_ocp.dims.N):
            self.ocp_solver.cost_set(n, "yref", np.concatenate([data[n] for data in self.y_ref])[:, 0])
        self.ocp_solver.solve()
        self.get_optimized_value()
        return self
