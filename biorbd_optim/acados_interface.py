import os
import biorbd
from acados_template import AcadosModel, AcadosOcp
from casadi import MX, Function, SX, vertcat
import numpy as np
import scipy.linalg
from .objective_functions import ObjectiveFunction


class SolverInterface:
    def __init__(self):
        self.solver = None
    u = self.nlp[0]['U'][0]
    p = self.nlp[0]['p_SX']
    mod = self.nlp[0]['model']
    x_dot = SX.sym("x_dot", mod.nbQdot() * 2, 1)

    def configure(self, **options):
        raise RuntimeError("SolverInterface is an abstract class")
    expl_ode_fun = Function('myFunName', [x, u, p], [f_expl]).expand()
    acados_model = AcadosModel()
    acados_model.f_impl_expr = f_impl
    acados_model.f_expl_expr = f_expl
    acados_model.x = x
    acados_model.xdot = x_dot
    acados_model.u = u
    acados_model.p = []
    acados_model.name = "model_name"

    def solve(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_iterations(self):
        raise RuntimeError("SolverInterface is an abstract class")

    def get_optimized_value(self):
        raise RuntimeError("SolverInterface is an abstract class")
    acados_ocp = AcadosOcp()

    # # set model
    acados_model = acados_export_model(self)
    acados_ocp.model = acados_model

class AcadosInterface(SolverInterface):
    def __init__(self, ocp):
        super().__init__()

        self.acados_ocp = AcadosOcp()

        self.acados_model = AcadosModel()
        self.acados_export_model(ocp)

        self.ocp_solver = None

    def acados_export_model(self, ocp):
        # Declare model variables
        x = ocp.nlp[0]['X'][0]
        u = ocp.nlp[0]['U'][0]
        p = ocp.nlp[0]['p_SX']
        mod = ocp.nlp[0]['model']
        x_dot = SX.sym("x_dot", mod.nbQdot() * 2, 1)

        f_expl = ocp.nlp[0]['dynamics_func'](x, u, p)
        f_impl = x_dot - f_expl
        # expl_ode_fun = Function('myFunName', [x, u, p], [f_expl]).expand()

        self.acados_model.f_impl_expr = f_impl
        self.acados_model.f_expl_expr = f_expl
        self.acados_model.x = x
        self.acados_model.xdot = x_dot
        self.acados_model.u = u
        self.acados_model.p = []
        self.acados_model.name = "model_name"

    def prepare_acados(self, ocp):
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
        self.acados_ocp.cost.cost_type = 'EXTERNAL'
        self.acados_ocp.cost.cost_type_e = 'EXTERNAL'

        if self.acados_ocp.cost.cost_type != self.acados_ocp.cost.cost_type_e:
            raise NotImplementedError(
                "Different cost types for Lagrange and Mayer terms in Acados not implemented yet.")

        # set weight for states and controls (default: 1.00)
        Q = 1.00 * np.eye(self.acados_ocp.dims.nx)
        R = 1.00 * np.eye(self.acados_ocp.dims.nu)

        self.acados_ocp.cost.W = scipy.linalg.block_diag(Q, R)

        self.acados_ocp.cost.W_e = Q

        if self.acados_ocp.cost.cost_type == 'LINEAR_LS':

            # set Lagrange terms
            self.acados_ocp.cost.Vx = np.zeros((self.acados_ocp.dims.ny, self.acados_ocp.dims.nx))
            self.acados_ocp.cost.Vx[:self.acados_ocp.dims.nx, :] = np.eye(self.acados_ocp.dims.nx)

            Vu = np.zeros((self.acados_ocp.dims.ny, self.acados_ocp.dims.nu))
            Vu[self.acados_ocp.dims.nx:, :] = np.eye(self.acados_ocp.dims.nu)
            self.acados_ocp.cost.Vu = Vu

            # set Mayer term
            self.acados_ocp.cost.Vx_e = np.zeros((self.acados_ocp.dims.nx, self.acados_ocp.dims.nx))

        elif self.acados_ocp.cost.cost_type == 'EXTERNAL':
            self.acados_ocp.model.cost_expr_ext_cost = SX(0, 0)
            self.acados_ocp.model.cost_expr_ext_cost_e = SX(0, 0)

            k = 0
            for i in range(ocp.nb_phases):
                for j in range(len(ocp.nlp[i]['J'])):
                    if ocp.original_values['objective_functions'][i][j][
                        'type']._get_type() == ObjectiveFunction.LagrangeFunction:
                        # set Lagrange term
                        if self.acados_ocp.model.cost_expr_ext_cost.shape == (0, 0):
                            self.acados_ocp.model.cost_expr_ext_cost = ocp.nlp[i]['J'][j][0]
                        else:
                            self.acados_ocp.model.cost_expr_ext_cost += ocp.nlp[i]['J'][j][0]
                    elif ocp.original_values['objective_functions'][i][j][
                        'type']._get_type() == ObjectiveFunction.MayerFunction:
                        # set Mayer term
                        if self.acados_ocp.model.cost_expr_ext_cost_e.shape == (0, 0):
                            self.acados_ocp.model.cost_expr_ext_cost_e = ocp.nlp[i]['J_acados_mayer'][k][0]
                            k += 1
                        else:
                            self.acados_ocp.model.cost_expr_ext_cost_e += ocp.nlp[i]['J_acados_mayer'][k][0]
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
                for k in range(ocp.nlp[i]['nx']):
                    if ocp.nlp[i]["X_bounds"].min[k, j] != ocp.nlp[i]["X_bounds"].max[k, j]:
                        raise RuntimeError("The initial values must be set and fixed.")

            self.acados_ocp.constraints.x0 = np.array(ocp.nlp[i]["X_bounds"].min[:, 0])
            self.acados_ocp.dims.nbx_0 = self.acados_ocp.dims.nx
            self.acados_ocp.constraints.constr_type = 'BGH'
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
