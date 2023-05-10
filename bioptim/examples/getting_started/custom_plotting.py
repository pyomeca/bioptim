"""
This example is a trivial example using the pendulum without any objective. It is designed to show how to create new
plots and how to expand pre-existing one with new information
"""

from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Bounds,
    InitialGuess,
    PlotType,
    Solver,
)


def custom_plot_callback(x: MX, q_to_plot: list) -> MX:
    """
    Create a used defined plot function with extra_parameters

    Parameters
    ----------
    x: MX
        The current states of the optimization
    q_to_plot: list
        The slice indices to plot

    Returns
    -------
    The value to plot
    """

    return x[q_to_plot, :]


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    torque_min, torque_max, torque_init = -100, 100, 0
    n_tau = bio_model.nb_tau
    u_bounds = Bounds([torque_min] * n_tau, [torque_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([torque_init] * n_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Create multiple new plot and add new stuff to them using a custom defined function
    """

    # Prepare the Optimal Control Program
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=2, n_shooting=50)

    # Add my lovely new plot
    ocp.add_plot("My New Extra Plot", lambda t, x, u, p: custom_plot_callback(x, [0, 1, 3]), plot_type=PlotType.PLOT)
    ocp.add_plot(
        "My New Extra Plot",
        lambda t, x, u, p: custom_plot_callback(x, [1, 3]),
        plot_type=PlotType.STEP,
        axes_idx=[1, 2],
    )
    ocp.add_plot(
        "My Second New Extra Plot",
        lambda t, x, u, p: custom_plot_callback(x, [1, 3]),
        plot_type=PlotType.INTEGRATED,
        axes_idx=[1, 2],
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    sol.graphs()


if __name__ == "__main__":
    main()
