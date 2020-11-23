from casadi import Function, vertcat, norm_fro, collocation_points, tangent, vertsplit, rootfinder
from ..misc.enums import ControlType


def RK4(ode, ode_opt):
    """
    Numerical integration using fourth order Runge-Kutta method.
    :param ode: ode["x"] -> States. ode["p"] -> Controls. ode["ode"] -> Ordinary differential equation function
    (dynamics of the system).
    :param ode_opt: ode_opt["t0"] -> Initial time of the integration. ode_opt["tf"] -> Final time of the integration.
    ode_opt["number_of_finite_elements"] -> Number of steps between nodes. ode_opt["idx"] -> Index of ??. (integer)
    :return: Integration function. (CasADi function)
    """
    t_span = ode_opt["t0"], ode_opt["tf"]
    n_step = ode_opt["number_of_finite_elements"]
    idx = ode_opt["idx"]
    CX = ode_opt["CX"]
    x_sym = ode["x"]
    u_sym = ode["p"]
    param_sym = ode_opt["param"]
    fun = ode["ode"]
    model = ode_opt["model"]
    step_time = t_span[1] - t_span[0]
    h_norm = 1 / n_step
    h = step_time * h_norm  # Length of steps
    control_type = ode_opt["control_type"]

    def get_u(u, dt_norm):
        if control_type == ControlType.CONSTANT:
            return u
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt_norm
        else:
            raise RuntimeError(f"{control_type} ControlType not implemented yet")

    def dxdt(h, states, controls, params):
        u = controls
        x = CX(states.shape[0], n_step + 1)
        p = params
        x[:, 0] = states

        nb_dof = 0
        quat_idx = []
        quat_number = 0
        for j in range(model.nbSegment()):
            if model.segment(j).isRotationAQuaternion():
                quat_idx.append([nb_dof, nb_dof + 1, nb_dof + 2, model.nbDof() + quat_number])
                quat_number += 1
            nb_dof += model.segment(j).nbDof()

        for i in range(1, n_step + 1):
            t_norm_init = (i - 1) / n_step  # normalized time
            k1 = fun(x[:, i - 1], get_u(u, t_norm_init), p)[:, idx]
            k2 = fun(x[:, i - 1] + h / 2 * k1, get_u(u, t_norm_init + h_norm / 2), p)[:, idx]
            k3 = fun(x[:, i - 1] + h / 2 * k2, get_u(u, t_norm_init + h_norm / 2), p)[:, idx]
            k4 = fun(x[:, i - 1] + h * k3, get_u(u, t_norm_init + h_norm), p)[:, idx]
            x[:, i] = x[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            for j in range(model.nbQuat()):
                quaternion = vertcat(
                    x[quat_idx[j][3], i], x[quat_idx[j][0], i], x[quat_idx[j][1], i], x[quat_idx[j][2], i]
                )
                quaternion /= norm_fro(quaternion)
                x[quat_idx[j][0] : quat_idx[j][2] + 1, i] = quaternion[1:4]
                x[quat_idx[j][3], i] = quaternion[0]

        return x[:, -1], x

    return Function(
        "integrator", [x_sym, u_sym, param_sym], dxdt(h, x_sym, u_sym, param_sym), ["x0", "p", "params"], ["xf", "xall"]
    )


def IRK(ode, ode_opt):
    """
    Numerical integration using implicit Runge-Kutta method.
    :param ode: ode["x"] -> States. ode["p"] -> Controls. ode["ode"] -> Ordinary differential equation function
    (dynamics of the system).
    :param ode_opt: ode_opt["t0"] -> Initial time of the integration. ode_opt["tf"] -> Final time of the integration.
    ode_opt["number_of_finite_elements"] -> Number of steps between nodes. ode_opt["idx"] -> Index of ??. (integer)
    :return: Integration function. (CasADi function)
    """
    t_span = ode_opt["t0"], ode_opt["tf"]
    degree = ode_opt["degree_of_interpolating_polynomial"]
    idx = ode_opt["idx"]
    CX = ode_opt["CX"]
    x_sym = ode["x"]
    u_sym = ode["p"]
    param_sym = ode_opt["param"]
    fun = ode["ode"]
    model = ode_opt["model"]
    step_time = t_span[1] - t_span[0]
    h = step_time  # Length between two nodes
    control_type = ode_opt["control_type"]

    def get_u(u, dt_norm):
        if control_type == ControlType.CONSTANT:
            return u
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt_norm
        else:
            raise RuntimeError(f"{control_type} ControlType not implemented yet")

    def dxdt(h, states, controls, params):
        nu = controls.shape[0]
        nx = states.shape[0]

        nb_dof = 0
        quat_idx = []
        quat_number = 0
        for j in range(model.nbSegment()):
            if model.segment(j).isRotationAQuaternion():
                quat_idx.append([nb_dof, nb_dof + 1, nb_dof + 2, model.nbDof() + quat_number])
                quat_number += 1
            nb_dof += model.segment(j).nbDof()

        # Choose collocation points
        time_points = [0] + collocation_points(degree, "legendre")

        # Coefficients of the collocation equation
        C = CX.zeros((degree + 1, degree + 1))

        # Coefficients of the continuity equation
        D = CX.zeros(degree + 1)

        # Dimensionless time inside one control interval
        time_control_interval = CX.sym("time_control_interval")

        # For all collocation points
        for j in range(degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            L = 1
            for r in range(degree + 1):
                if r != j:
                    L *= (time_control_interval - time_points[r]) / (time_points[j] - time_points[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            lfcn = Function("lfcn", [time_control_interval], [L])
            D[j] = lfcn(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            tfcn = Function("tfcn", [time_control_interval], [tangent(L, time_control_interval)])
            for r in range(degree + 1):
                C[j, r] = tfcn(time_points[r])

        # Total number of variables for one finite element
        X0 = states
        U = controls
        P = params

        # Get the state at each collocation point
        # X = [X0] + vertsplit(V, [r * nx for r in range(degree + 1)])

        # Get the collocation equations (that define V)
        V = [CX.sym(f"X_irk_{j}", nx, 1) for j in range(1, degree + 1)]
        X = [X0] + V

        V_eq = []
        for j in range(1, degree + 1):

            t_norm_init = (j - 1) / degree  # normalized time
            # Expression for the state derivative at the collocation point
            xp_j = 0
            for r in range(degree + 1):
                xp_j += C[r, j] * X[r]

            # Append collocation equations
            f_j = fun(X[j], get_u(U, t_norm_init), P)[:, idx]
            V_eq.append(h * f_j - xp_j)

        # Concatenate constraints
        V = vertcat(*V)
        V_eq = vertcat(*V_eq)

        # Root-finding function, implicitly defines V as a function of X0 and P
        vfcn = Function("vfcn", [V, X0, U, P], [V_eq]).expand()

        # Create a implicit function instance to solve the system of equations
        ifcn = rootfinder("ifcn", "newton", vfcn)
        V = ifcn(CX(), X0, U, P)
        X = [X0 if r == 0 else V[(r - 1) * nx : r * nx] for r in range(degree + 1)]

        # Get an expression for the state at the end of the finite element
        XF = CX.zeros(nx, degree + 1)  # 0 #
        for r in range(degree + 1):
            XF[:, r] = XF[:, r - 1] + D[r] * X[r]

            # Dont know if its better to renormalize at each collocation point or only the last one
            for j in range(model.nbQuat()):
                quaternion = vertcat(
                    XF[quat_idx[j][3], r], XF[quat_idx[j][0], r], XF[quat_idx[j][1], r], XF[quat_idx[j][2], r]
                )
                quaternion /= norm_fro(quaternion)
                XF[quat_idx[j][0] : quat_idx[j][2] + 1, r] = quaternion[1:4]
                XF[quat_idx[j][3], r] = quaternion[0]

        return XF[:, -1], XF

    return Function(
        "integrator", [x_sym, u_sym, param_sym], dxdt(h, x_sym, u_sym, param_sym), ["x0", "p", "params"], ["xf", "xall"]
    )
