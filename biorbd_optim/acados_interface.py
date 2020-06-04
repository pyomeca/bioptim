import os
import biorbd
from acados_template import AcadosModel,  AcadosOcp
from casadi import MX, Function, SX, vertcat
import numpy as np
import scipy.linalg


#TODO: change the name of the function
class PrepareAcados:
    def export_eocar_ode_model(self):
        for i in range(self.nb_phases):
            m = biorbd.Model(self.nlp[i]['model'].path().relativePath().to_string())
        model_name = "model_name"

        # Declare model variables
        x = MX.sym('x', m.nbQ() * 2)
        u = MX.sym('u', m.nbQ())
        xdot = MX.sym('dx', m.nbQ() * 2)
        #TODO: get data directly from nlp['dynamics']
        f = Function('f', [x, u],
                     [vertcat(x[m.nbQ():], m.ForwardDynamics(x[:m.nbQ()], x[m.nbQ():], u).to_mx())]).expand()
        x = SX.sym('x', m.nbQ() * 2)
        u = SX.sym('u', m.nbQ())
        xdot = SX.sym('dx', m.nbQ() * 2)
    f_expl = self.nlp[0]['dynamics'][0](x,u)
        f_impl = xdot - f_expl

        acados_model = AcadosModel()
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl
        acados_model.x = x
        acados_model.xdot = xdot
        acados_model.u = u
        # model.z = z
        acados_model.p = []
        acados_model.name = model_name

        return acados_model


    def prepare_acados(self):
        #TODO: Allow user to define source_dir directly in example file
        os.environ["ACADOS_SOURCE_DIR"] = "/home/dangzilla/Documents/Programmation/acados"
        # create ocp object to formulate the OCP
        acados_ocp = AcadosOcp()

        # # set model
        acados_model = self.export_eocar_ode_model
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
        #TODO: test external cost_type
        acados_ocp.cost.cost_type = 'LINEAR_LS'
        acados_ocp.cost.cost_type_e = 'LINEAR_LS'

        # set weight à modifier avec objective functions (user needs to define this TODO)
        Q = 0.00 * np.eye(acados_ocp.dims.nx)
        R = 5 * np.eye(acados_ocp.dims.nu)

        acados_ocp.cost.W = scipy.linalg.block_diag(Q, R)

        acados_ocp.cost.W_e = Q

        # set Lagrange term matrices
        acados_ocp.cost.Vx = np.zeros((acados_ocp.dims.ny, acados_ocp.dims.nx))
        acados_ocp.cost.Vx[:acados_ocp.dims.nx, :] = np.eye(acados_ocp.dims.nx)

        # Vu = np.zeros((ny, nu))
        # Vu[nx:, :] = np.eye(nu)
        # ocp.cost.Vu = Vu

        Vu = np.zeros((acados_ocp.dims.ny, acados_ocp.dims.nu))
        Vu[acados_ocp.dims.nx:, :] = np.eye(acados_ocp.dims.nu)
        acados_ocp.cost.Vu = Vu

        # set Mayer term matrices
        acados_ocp.cost.Vx_e = np.zeros((acados_ocp.dims.nx, acados_ocp.dims.nx))

        acados_ocp.cost.yref = np.zeros((acados_ocp.dims.ny,))
        acados_ocp.cost.yref_e = np.ones((acados_ocp.dims.ny_e,))

        for i in range(self.nb_phases):
            # set constraints
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


