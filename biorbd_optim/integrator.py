from casadi import MX, Function


def RK4(ode, ode_opt):
    t_span = ode_opt['t0'], ode_opt['tf']
    n_step = ode_opt['number_of_finite_elements']
    x_sym = ode['x']
    u_sym = ode['p']
    fun = ode['ode']
    h = (t_span[1] - t_span[0]) / n_step  # Length of steps

    def dxdt(h, states, controls):
        u = controls
        x = MX(states.shape[0], n_step + 1)
        x[:, 0] = states

        for i in range(1, n_step + 1):
            k1 = fun(x[:, i - 1], u)
            k2 = fun(x[:, i - 1] + h / 2 * k1, u)
            k3 = fun(x[:, i - 1] + h / 2 * k2, u)
            k4 = fun(x[:, i - 1] + h * k3, u)
            x[:, i] = x[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x[:, -1]

    return Function("integrator",
             [x_sym, u_sym],
             [dxdt(h, x_sym, u_sym)],
             ["x0", "p"],
             ["xf"])
