import biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeOption,
    DynamicsType,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    ShowResult,
    PlotType,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    x_init = InitialGuessOption([0] * (n_q + n_qdot))

    # Define control path constraint
    u_bounds = BoundsOption([[torque_min] * n_tau, [torque_max] * n_tau])
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuessOption([torque_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
    )


def plot_callback(x, q_to_plot):
    return x[q_to_plot, :]


if __name__ == "__main__":
    # Prepare the Optimal Control Program
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=2, number_shooting_points=50)

    # Add my lovely new plot
    ocp.add_plot("My New Extra Plot", lambda x, u, p: plot_callback(x, [0, 1, 3]), PlotType.PLOT)
    ocp.add_plot(
        "My New Extra Plot", lambda x, u, p: plot_callback(x, [1, 3]), plot_type=PlotType.STEP, axes_idx=[1, 2]
    )
    ocp.add_plot(
        "My Second New Extra Plot",
        lambda x, u, p: plot_callback(x, [1, 3]),
        plot_type=PlotType.INTEGRATED,
        axes_idx=[1, 2],
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
