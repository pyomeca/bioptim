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

    for i in range(self.nb_phases):
        # set time
        acados_ocp.solver_options.tf = self.nlp[i]["tf"]
        # set dimensions
        acados_ocp.dims.nx = self.nlp[i]["nx"]
        acados_ocp.dims.nu = self.nlp[i]["nu"]
        acados_ocp.dims.ny = acados_ocp.dims.nx + acados_ocp.dims.nu
        acados_ocp.dims.ny_e = self.nlp[i]["nx"]
        acados_ocp.dims.N = self.nlp[i]["ns"]

    # set cost module
    acados_ocp.cost.cost_type = 'EXTERNAL'
    acados_ocp.cost.cost_type_e = 'EXTERNAL'

    if acados_ocp.cost.cost_type != acados_ocp.cost.cost_type_e:
        raise NotImplementedError("Different cost types for Lagrange and Mayer terms in Acados not implemented yet.")

    # set weight for states and controls (default: 1.00)
    Q = 1.00 * np.eye(acados_ocp.dims.nx)
    R = 1.00 * np.eye(acados_ocp.dims.nu)

    acados_ocp.cost.W = scipy.linalg.block_diag(Q, R)

    acados_ocp.cost.W_e = Q

    if acados_ocp.cost.cost_type == 'LINEAR_LS':

        # set Lagrange terms
        acados_ocp.cost.Vx = np.zeros((acados_ocp.dims.ny, acados_ocp.dims.nx))
        acados_ocp.cost.Vx[:acados_ocp.dims.nx, :] = np.eye(acados_ocp.dims.nx)

        Vu = np.zeros((acados_ocp.dims.ny, acados_ocp.dims.nu))
        Vu[acados_ocp.dims.nx:, :] = np.eye(acados_ocp.dims.nu)
        acados_ocp.cost.Vu = Vu

        # set Mayer term
        acados_ocp.cost.Vx_e = np.zeros((acados_ocp.dims.nx, acados_ocp.dims.nx))

    elif acados_ocp.cost.cost_type == 'EXTERNAL':
        acados_ocp.model.cost_expr_ext_cost = SX(0,0)
        acados_ocp.model.cost_expr_ext_cost_e = SX(0,0)

        k = 0;
        for i in range(self.nb_phases):
            for j in range(len(self.nlp[i]['J'])):
                if self.original_values['objective_functions'][i][j]['type']._get_type() == ObjectiveFunction.LagrangeFunction:
                    # set Lagrange term
                    if acados_ocp.model.cost_expr_ext_cost.shape == (0,0):
                        acados_ocp.model.cost_expr_ext_cost = self.nlp[i]['J'][j][0]
                    else:
                        acados_ocp.model.cost_expr_ext_cost += self.nlp[i]['J'][j][0]
                elif self.original_values['objective_functions'][i][j]['type']._get_type() == ObjectiveFunction.MayerFunction:
                    # set Mayer term
                    if acados_ocp.model.cost_expr_ext_cost_e.shape == (0,0):
                        acados_ocp.model.cost_expr_ext_cost_e = self.nlp[i]['J_acados_mayer'][k][0]
                        k +=1
                    else:
                        acados_ocp.model.cost_expr_ext_cost_e += self.nlp[i]['J_acados_mayer'][k][0]
                        k += 1
                else:
                    raise RuntimeError("The objective function is not Lagrange nor Mayer.")

    else:
        raise RuntimeError("Available acados cost type: 'LINEAR_LS' and 'EXTERNAL'.")


    # set y values
    acados_ocp.cost.yref = np.zeros((acados_ocp.dims.ny,))
    acados_ocp.cost.yref_e = np.ones((acados_ocp.dims.ny_e,))

    for i in range(self.nb_phases):
        # set constraints
        for j in range(-1,0):
            for k in range(self.nlp[i]['nx']):
                if self.nlp[i]["X_bounds"].min[k, j] != self.nlp[i]["X_bounds"].max[k,j]:
                    raise RuntimeError("The initial values must be set and fixed.")

        acados_ocp.constraints.x0 = np.array(self.nlp[i]["X_bounds"].min[:, 0])
        acados_ocp.dims.nbx_0 = acados_ocp.dims.nx
        acados_ocp.constraints.constr_type = 'BGH'  # TODO: put as an option in ocp?
        acados_ocp.constraints.lbu = np.array(self.nlp[i]["U_bounds"].min[:, 0])
        acados_ocp.constraints.ubu = np.array(self.nlp[i]["U_bounds"].max[:, 0])
        acados_ocp.constraints.idxbu = np.array(range(acados_ocp.dims.nu))
        acados_ocp.dims.nbu = acados_ocp.dims.nu

        # set control constraints
        acados_ocp.constraints.Jbx_e = np.eye(acados_ocp.dims.nx)
        acados_ocp.constraints.ubx_e = np.array(self.nlp[i]["X_bounds"].max[:, -1])
        acados_ocp.constraints.lbx_e = np.array(self.nlp[i]["X_bounds"].min[:, -1])
        acados_ocp.constraints.idxbx_e = np.array(range(acados_ocp.dims.nx))
        acados_ocp.dims.nbx_e = acados_ocp.dims.nx

    return acados_ocp
