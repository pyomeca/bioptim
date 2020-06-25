from casadi import MX, Function, vertcat, norm_fro


def RK4(Model, ode, ode_opt):
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
    x_sym = ode["x"]
    u_sym = ode["p"]
    param_sym = ode_opt["param"]
    fun = ode["ode"]
    h = (t_span[1] - t_span[0]) / n_step  # Length of steps

    def dxdt(h, states, controls, params):
        u = controls
        x = MX(states.shape[0], n_step + 1)
        p = params
        x[:, 0] = states

        for i in range(1, n_step + 1):
            k1 = fun(x[:, i - 1], u, p)[:, idx]
            k2 = fun(x[:, i - 1] + h / 2 * k1, u, p)[:, idx]
            k3 = fun(x[:, i - 1] + h / 2 * k2, u, p)[:, idx]
            k4 = fun(x[:, i - 1] + h * k3, u, p)[:, idx]
            x[:, i] = x[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        Quat_idx = []
        for j in range(Model.nbDof() + 1):
            Name = Model.nameDof()[j].to_string()
            if Name[-5:-1] == "Quat":
                Quat_idx += [j]

        for j in range(Model.nbQuat()):
            quaternion = vertcat(x[Quat_idx[j + 3], i], x[Quat_idx[j], i], x[Quat_idx[j + 1], i], x[Quat_idx[j + 2], i])
            quaternion /= norm_fro(quaternion)
            x[Quat_idx[j] : Quat_idx[j + 3], i] = vertcat(quaternion[1], quaternion[2], quaternion[3])
            x[Quat_idx[j + 3], i] = quaternion[0]
        return x[:, -1], x

    return Function(
        "integrator",
        [x_sym, u_sym, param_sym],
        dxdt(h, x_sym, u_sym, param_sym),
        ["x0", "p", "params"],
        ["xf", "xall"],
    )
