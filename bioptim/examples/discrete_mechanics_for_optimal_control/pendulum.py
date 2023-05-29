"""
A pendulum simulation copying the example from bioptim/examples/getting_started/pendulum.py but integrated by the
variational integrator. Moreover, the model has been freed on the z-axis, it is constrained by the holonomic constraint.
"""
import pickle
import numpy as np

from bioptim import (
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    Solver,
    BiorbdModel,
    DynamicsList,
    ParameterList,
    InterpolationType,
)

from variational_integrator import *
from save_results import save_results


def prepare_ocp(
        bio_model_path: str,
        final_time: float,
        n_shooting: int,
        use_sx: bool = True,
        assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    bio_model_path: str
        The path to the biorbd model.
    final_time: float
        The time in second required to perform the task.
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program.
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM).
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node.

    Returns
    -------
    The OptimalControlProgram ready to be solved.
    """

    bio_model = BiorbdModel(bio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14
    # Initial guess
    n_q = bio_model.nb_q
    x_0_guess = np.zeros(n_shooting + 1)
    x_linear_guess = np.linspace(0, np.pi, n_shooting + 1)
    x_init = InitialGuess([x_0_guess, x_linear_guess], interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate
    # Initial guess
    u_init = InitialGuess([0] * n_q)

    # Declare parameters for the initial and final velocities
    parameters = ParameterList()
    # Give the parameter some min and max bounds
    qdot0_bounds = Bounds([0.0, 0.0], [0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    qdotN_bounds = Bounds([0.0, 0.0], [0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    # And an initial guess
    qdot0_init = InitialGuess([0] * n_q)
    qdotN_init = InitialGuess([0] * n_q)

    parameters.add(
        "qdot0",  # The name of the parameter
        function=qdot_function,  # The function that modifies the biorbd model
        initial_guess=qdot0_init,  # The initial guess
        bounds=qdot0_bounds,  # The bounds
        size=n_q,  # The number of elements this particular parameter vector has
    )
    parameters.add(
        "qdotN",  # The name of the parameter
        function=qdot_function,  # The function that modifies the biorbd model
        initial_guess=qdotN_init,  # The initial guess
        bounds=qdotN_bounds,  # The bounds
        size=n_q,  # The number of elements this particular parameter vector has
    )

    # Dynamics
    dynamics = DynamicsList()
    expand = True
    dynamics.add(custom_configure_unconstrained, bio_model=bio_model, expand=expand)

    multinode_constraints = variational_continuity(n_shooting)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
        assume_phase_dynamics=assume_phase_dynamics,
        skip_continuity=True,
        parameters=parameters,
        multinode_constraints=multinode_constraints,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """
    n_shooting = 100

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(bio_model_path="models/pendulum.bioMod", final_time=1, n_shooting=n_shooting)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show the results in a bioviz animation --- #
    sol.detailed_cost_values()  # /!\ Since the last controls are nan the costs are not accurate /!\
    sol.print_cost()
    sol.animate()

    save_results(sol, f"results/varint_{n_shooting}_nodes")

    with open(f"results/RK4_100_nodes", "rb") as f:
        data = pickle.load(f)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title("q_Seg1_TransY-0")
    axs[0, 0].plot(sol.time, sol.states['q'][0], "purple")
    axs[0, 0].plot(data["time"], data["states"]["q"][0], "--m")
    axs[0, 1].set_title("q_Seg1_RotX-0")
    axs[0, 1].plot(sol.time, sol.states['q'][1], "purple")
    axs[0, 1].plot(data["time"], data["states"]["q"][1], "--m")
    axs[1, 0].set_title("tau_Seg1_TransY-0")
    axs[1, 0].step(sol.time, sol.controls['tau'][0], "orange")
    axs[1, 0].step(data["time"], data["controls"]["tau"][0], "--y")
    axs[1, 1].set_title("tau_Seg1_RotX-0")
    axs[1, 1].step(sol.time, sol.controls['tau'][1], "orange")
    axs[1, 1].step(data["time"], data["controls"]["tau"][1], "--y")

    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlabel("Time (s)")
            axs[i, j].legend(["Variational integrator", "RK4"])

    plt.show()


if __name__ == "__main__":
    main()
