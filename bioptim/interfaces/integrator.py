from casadi import Function, vertcat, horzcat, norm_fro, collocation_points, tangent, rootfinder

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
    degree = ode_opt["irk_polynomial_interpolation_degree"]
    idx = ode_opt["idx"]
    CX = ode_opt["CX"]
    x_sym = ode["x"]
    u_sym = ode["p"]
    param_sym = ode_opt["param"]
    fun = ode["ode"]
    step_time = t_span[1] - t_span[0]
    h = step_time
    control_type = ode_opt["control_type"]

    def get_u(u, dt_norm):
        if control_type == ControlType.CONSTANT:
            return u
        else:
            raise NotImplementedError(f"{control_type} ControlType not implemented yet")

    def dxdt(h, states, controls, params):
        nu = controls.shape[0]
        nx = states.shape[0]

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

            # Evaluate the time derivative of the polynomial at all collocation points to get
            # the coefficients of the continuity equation
            tfcn = Function("tfcn", [time_control_interval], [tangent(L, time_control_interval)])
            for r in range(degree + 1):
                C[j, r] = tfcn(time_points[r])

        # Total number of variables for one finite element
        x0 = states
        u = controls

        x_irk_points = [CX.sym(f"X_irk_{j}", nx, 1) for j in range(1, degree + 1)]
        x = [x0] + x_irk_points

        x_irk_points_eq = []
        for j in range(1, degree + 1):

            t_norm_init = (j - 1) / degree  # normalized time
            # Expression for the state derivative at the collocation point
            xp_j = 0
            for r in range(degree + 1):
                xp_j += C[r, j] * x[r]

            # Append collocation equations
            f_j = fun(x[j], get_u(u, t_norm_init), params)[:, idx]
            x_irk_points_eq.append(h * f_j - xp_j)

        # Concatenate constraints
        x_irk_points = vertcat(*x_irk_points)
        x_irk_points_eq = vertcat(*x_irk_points_eq)

        # Root-finding function, implicitly defines x_irk_points as a function of x0 and p
        vfcn = Function("vfcn", [x_irk_points, x0, u, params], [x_irk_points_eq]).expand()

        # Create a implicit function instance to solve the system of equations
        ifcn = rootfinder("ifcn", "newton", vfcn)
        x_irk_points = ifcn(CX(), x0, u, params)
        x = [x0 if r == 0 else x_irk_points[(r - 1) * nx : r * nx] for r in range(degree + 1)]

        # Get an expression for the state at the end of the finite element
        xf = CX.zeros(nx, degree + 1)  # 0 #
        for r in range(degree + 1):
            xf[:, r] = xf[:, r - 1] + D[r] * x[r]

        return xf[:, -1], horzcat(x0, xf[:, -1])

    return Function(
        "integrator", [x_sym, u_sym, param_sym], dxdt(h, x_sym, u_sym, param_sym), ["x0", "p", "params"], ["xf", "xall"]
    )
