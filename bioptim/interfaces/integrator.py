from casadi import Function, vertcat, norm_fro
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
