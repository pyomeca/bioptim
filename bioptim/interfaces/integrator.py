from casadi import Function, vertcat, horzcat, norm_fro, collocation_points, tangent, rootfinder

from ..misc.enums import ControlType


class Integrator:
    def __init__(self, ode, ode_opt):
        self.model = ode_opt["model"]
        self.t_span = ode_opt["t0"], ode_opt["tf"]
        self.idx = ode_opt["idx"]
        self.CX = ode_opt["CX"]
        self.x_sym = ode["x"]
        self.u_sym = ode["p"]
        self.param_sym = ode_opt["param"]
        self.fun = ode["ode"]
        self.control_type = ode_opt["control_type"]
        self.step_time = self.t_span[1] - self.t_span[0]
        self.h = self.step_time
        self.function = None

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def map(self, *args, **kwargs):
        return self.function.map(*args, **kwargs)

    def get_u(self, u, dt_norm):
        if self.control_type == ControlType.CONSTANT:
            return u
        elif self.control_type == ControlType.LINEAR_CONTINUOUS:
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt_norm
        else:
            raise RuntimeError(f"{self.control_type} ControlType not implemented yet")

    def dxdt(self, h, states, controls, params):
        raise RuntimeError("Integrator is abstract, please specify a proper one")

    def _finish_init(self):
        self.function = Function(
            "integrator",
            [self.x_sym, self.u_sym, self.param_sym],
            self.dxdt(self.h, self.x_sym, self.u_sym, self.param_sym),
            ["x0", "p", "params"],
            ["xf", "xall"],
        )


class RK(Integrator):
    """
    Numerical integration using fourth order Runge-Kutta method.
    :param ode: ode["x"] -> States. ode["p"] -> Controls. ode["ode"] -> Ordinary differential equation function
    (dynamics of the system).
    :param ode_opt: ode_opt["t0"] -> Initial time of the integration. ode_opt["tf"] -> Final time of the integration.
    ode_opt["number_of_finite_elements"] -> Number of steps between nodes. ode_opt["idx"] -> Index of ??. (integer)
    :return: Integration function. (CasADi function)
    """

    def __init__(self, ode, ode_opt):
        super(RK, self).__init__(ode, ode_opt)
        self.n_step = ode_opt["number_of_finite_elements"]
        self.h_norm = 1 / self.n_step
        self.h = self.step_time * self.h_norm  # Length of steps

    def next_x(self, h, t, x_prev, u, p):
        raise RuntimeError("RK is abstract, please select a specific RK")

    def dxdt(self, h, states, controls, params):
        u = controls
        x = self.CX(states.shape[0], self.n_step + 1)
        p = params
        x[:, 0] = states

        nb_dof = 0
        quat_idx = []
        quat_number = 0
        for j in range(self.model.nbSegment()):
            if self.model.segment(j).isRotationAQuaternion():
                quat_idx.append([nb_dof, nb_dof + 1, nb_dof + 2, self.model.nbDof() + quat_number])
                quat_number += 1
            nb_dof += self.model.segment(j).nbDof()

        for i in range(1, self.n_step + 1):
            t_norm_init = (i - 1) / self.n_step  # normalized time
            x[:, i] = self.next_x(h, t_norm_init, x[:, i - 1], u, p)

            for j in range(self.model.nbQuat()):
                quaternion = vertcat(
                    x[quat_idx[j][3], i], x[quat_idx[j][0], i], x[quat_idx[j][1], i], x[quat_idx[j][2], i]
                )
                quaternion /= norm_fro(quaternion)
                x[quat_idx[j][0] : quat_idx[j][2] + 1, i] = quaternion[1:4]
                x[quat_idx[j][3], i] = quaternion[0]

        return x[:, -1], x


class RK4(RK):
    def __init__(self, ode, ode_opt):
        super(RK4, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h, t, x_prev, u, p):
        k1 = self.fun(x_prev, self.get_u(u, t), p)[:, self.idx]
        k2 = self.fun(x_prev + h / 2 * k1, self.get_u(u, t + self.h_norm / 2), p)[:, self.idx]
        k3 = self.fun(x_prev + h / 2 * k2, self.get_u(u, t + self.h_norm / 2), p)[:, self.idx]
        k4 = self.fun(x_prev + h * k3, self.get_u(u, t + self.h_norm), p)[:, self.idx]
        return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class RK8(RK4):
    def __init__(self, ode, ode_opt):
        super(RK8, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h, t, x_prev, u, p):
        k1 = self.fun(x_prev, self.get_u(u, t), p)[:, self.idx]
        k2 = self.fun(x_prev + (h * 4 / 27) * k1, self.get_u(u, t + self.h_norm * (4 / 27)), p)[:, self.idx]
        k3 = self.fun(x_prev + (h / 18) * (k1 + 3 * k2), self.get_u(u, t + self.h_norm * (2 / 9)), p)[:, self.idx]
        k4 = self.fun(x_prev + (h / 12) * (k1 + 3 * k3), self.get_u(u, t + self.h_norm * (1 / 3)), p)[:, self.idx]
        k5 = self.fun(x_prev + (h / 8) * (k1 + 3 * k4), self.get_u(u, t + self.h_norm * (1 / 2)), p)[:, self.idx]
        k6 = self.fun(
            x_prev + (h / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5), self.get_u(u, t + self.h_norm * (2 / 3)), p
        )[:, self.idx]
        k7 = self.fun(
            x_prev + (h / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6),
            self.get_u(u, t + self.h_norm * (1 / 6)),
            p,
        )[:, self.idx]
        k8 = self.fun(
            x_prev + (h / 20) * (-234 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7),
            self.get_u(u, t + self.h_norm),
            p,
        )[:, self.idx]
        k9 = self.fun(
            x_prev + (h / 288) * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8),
            self.get_u(u, t + self.h_norm * (5 / 6)),
            p,
        )[:, self.idx]
        k10 = self.fun(
            x_prev
            + (h / 820) * (1481 * k1 - 81 * k3 + 7104 * k4 - 3376 * k5 + 72 * k6 - 5040 * k7 - 60 * k8 + 720 * k9),
            self.get_u(u, t + self.h_norm),
            p,
        )[:, self.idx]

        return x_prev + h / 840 * (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10)


class IRK(Integrator):
    """
    Numerical integration using implicit Runge-Kutta method.
    :param ode: ode["x"] -> States. ode["p"] -> Controls. ode["ode"] -> Ordinary differential equation function
    (dynamics of the system).
    :param ode_opt: ode_opt["t0"] -> Initial time of the integration. ode_opt["tf"] -> Final time of the integration.
    ode_opt["number_of_finite_elements"] -> Number of steps between nodes. ode_opt["idx"] -> Index of ??. (integer)
    :return: Integration function. (CasADi function)
    """

    def __init__(self, ode, ode_opt):
        super(IRK, self).__init__(ode, ode_opt)
        self.degree = ode_opt["irk_polynomial_interpolation_degree"]
        self._finish_init()

    def get_u(self, u, dt_norm):
        if self.control_type == ControlType.CONSTANT:
            return super(IRK, self).get_u(u, dt_norm)
        else:
            raise NotImplementedError(f"{self.control_type} ControlType not implemented yet with IRK")

    def dxdt(self, h, states, controls, params):
        nu = controls.shape[0]
        nx = states.shape[0]

        # Choose collocation points
        time_points = [0] + collocation_points(self.degree, "legendre")

        # Coefficients of the collocation equation
        C = self.CX.zeros((self.degree + 1, self.degree + 1))

        # Coefficients of the continuity equation
        D = self.CX.zeros(self.degree + 1)

        # Dimensionless time inside one control interval
        time_control_interval = self.CX.sym("time_control_interval")

        # For all collocation points
        for j in range(self.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            L = 1
            for r in range(self.degree + 1):
                if r != j:
                    L *= (time_control_interval - time_points[r]) / (time_points[j] - time_points[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            lfcn = Function("lfcn", [time_control_interval], [L])
            D[j] = lfcn(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get
            # the coefficients of the continuity equation
            tfcn = Function("tfcn", [time_control_interval], [tangent(L, time_control_interval)])
            for r in range(self.degree + 1):
                C[j, r] = tfcn(time_points[r])

        # Total number of variables for one finite element
        x0 = states
        u = controls

        x_irk_points = [self.CX.sym(f"X_irk_{j}", nx, 1) for j in range(1, self.degree + 1)]
        x = [x0] + x_irk_points

        x_irk_points_eq = []
        for j in range(1, self.degree + 1):

            t_norm_init = (j - 1) / self.degree  # normalized time
            # Expression for the state derivative at the collocation point
            xp_j = 0
            for r in range(self.degree + 1):
                xp_j += C[r, j] * x[r]

            # Append collocation equations
            f_j = self.fun(x[j], self.get_u(u, t_norm_init), params)[:, self.idx]
            x_irk_points_eq.append(h * f_j - xp_j)

        # Concatenate constraints
        x_irk_points = vertcat(*x_irk_points)
        x_irk_points_eq = vertcat(*x_irk_points_eq)

        # Root-finding function, implicitly defines x_irk_points as a function of x0 and p
        vfcn = Function("vfcn", [x_irk_points, x0, u, params], [x_irk_points_eq]).expand()

        # Create a implicit function instance to solve the system of equations
        ifcn = rootfinder("ifcn", "newton", vfcn)
        x_irk_points = ifcn(self.CX(), x0, u, params)
        x = [x0 if r == 0 else x_irk_points[(r - 1) * nx : r * nx] for r in range(self.degree + 1)]

        # Get an expression for the state at the end of the finite element
        xf = self.CX.zeros(nx, self.degree + 1)  # 0 #
        for r in range(self.degree + 1):
            xf[:, r] = xf[:, r - 1] + D[r] * x[r]

        return xf[:, -1], horzcat(x0, xf[:, -1])
