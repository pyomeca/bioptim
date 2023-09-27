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
    BoundsList,
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
    expand_dynamics: bool = True,
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
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand=expand_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    torque_min, torque_max = -100, 100
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [torque_min] * n_tau, [torque_max] * n_tau

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    # Add my lovely new plot
    ocp.add_plot("My New Extra Plot", lambda t, x, u, p, s: custom_plot_callback(x, [0, 1, 3]), plot_type=PlotType.PLOT)
    ocp.add_plot(  # This one combines to the previous one as they have the same name
        "My New Extra Plot",
        lambda t, x, u, p, s: custom_plot_callback(x, [1, 3]),
        plot_type=PlotType.STEP,
        axes_idx=[1, 2],
    )
    ocp.add_plot(
        "My Second New Extra Plot",
        lambda t, x, u, p, s: custom_plot_callback(x, [2, 1]),
        plot_type=PlotType.INTEGRATED,
        axes_idx=[0, 2],
    )

    return ocp


def main():
    """
    Create multiple new plot and add new stuff to them using a custom defined function
    """

    # Prepare the Optimal Control Program
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=2, n_shooting=50)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    sol.graphs()


if __name__ == "__main__":
    main()
