"""
A pendulum simulation copying the example from bioptim/examples/getting_started/pendulum.py but integrated by the
variational integrator.
"""
from bioptim import (
    Bounds,
    InitialGuess,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    Solver,
)
import matplotlib.pyplot as plt
import numpy as np

from biorbd_model_holonomic import BiorbdModelCustomHolonomic
from variational_optimal_control_program import VariationalOptimalControlProgram


def prepare_ocp(
    bio_model_path: str,
    final_time: float,
    n_shooting: int,
    use_sx: bool = True,
) -> VariationalOptimalControlProgram:
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

    Returns
    -------
    The OptimalControlProgram ready to be solved.
    """

    bio_model = BiorbdModelCustomHolonomic(bio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Path constraint
    x_bounds = Bounds([-1, -1, -2 * np.pi, -20], [5, 5, 2 * np.pi, 20])
    x_bounds[:3, [0, -1]] = 0
    x_bounds[2, -1] = 3.14
    # Initial guess
    n_q = bio_model.nb_q
    x_0_guess = np.zeros(n_shooting + 1)
    x_linear_guess = np.linspace(0, np.pi, n_shooting + 1)
    x_init = InitialGuess([x_0_guess, x_0_guess, x_linear_guess, x_0_guess], interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively impose forces on z axis
    u_bounds[2, :] = 0  # Prevent the model from actively rotate
    # Initial guess
    u_init = InitialGuess([tau_init] * n_q)

    # Give the initial and final velocities some min and max bounds
    qdot0_bounds = Bounds([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    qdotN_bounds = Bounds([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    # And an initial guess
    qdot0_init = InitialGuess([0] * n_q)
    qdotN_init = InitialGuess([0] * n_q)

    # Holonomic constraints: The pendulum must not move on the z axis
    constraints, jac = bio_model.generate_constraint_and_jacobian_functions(marker_1="marker_1", index=slice(2, 3))

    return VariationalOptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        q_init=x_init,
        u_init=u_init,
        q_bounds=x_bounds,
        u_bounds=u_bounds,
        qdot0_init=qdot0_init,
        qdot0_bounds=qdot0_bounds,
        qdotN_init=qdotN_init,
        qdotN_bounds=qdotN_bounds,
        holonomic_constraints=constraints,
        holonomic_constraints_jacobian=jac,
        objective_functions=objective_functions,
        use_sx=use_sx,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """
    n_shooting = 100

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(bio_model_path="models/pendulum_holonomic.bioMod", final_time=1, n_shooting=n_shooting)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show the results in a bioviz animation --- #
    sol.detailed_cost_values()  # /!\ Since the last controls are nan the costs are not accurate /!\
    sol.print_cost()
    sol.animate()

    # save_results(sol, f"results/varint_{n_shooting}_nodes_holonomic")

    # with open(f"results/varint_{n_shooting}_nodes", "rb") as f:
    #     data = pickle.load(f)

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].set_title("q_Seg1_TransY-0")
    axs[0, 0].plot(sol.time, sol.states["q"][0], "purple")
    # axs[0, 0].plot(data["time"], data["states"]["q"][0], "--m")
    axs[0, 1].set_title("q_Seg1_TransZ-0")
    axs[0, 1].plot(sol.time, sol.states["q"][1], "purple")
    axs[0, 2].set_title("q_Seg1_RotX-0")
    axs[0, 2].plot(sol.time, sol.states["q"][2], "purple")
    # axs[0, 2].plot(data["time"], data["states"]["q"][1], "--m")
    axs[1, 0].set_title("tau_Seg1_TransY-0")
    axs[1, 0].step(sol.time, sol.controls["tau"][0], "orange")
    # axs[1, 0].step(data["time"], data["controls"]["tau"][0], "--y")
    axs[1, 1].set_title("tau_Seg1_TransZ-0")
    axs[1, 1].step(sol.time, sol.controls["tau"][1], "orange")
    axs[1, 2].set_title("tau_Seg1_RotX-0")
    axs[1, 2].step(sol.time, sol.controls["tau"][2], "orange")
    # axs[1, 2].step(data["time"], data["controls"]["tau"][1], "--y")

    for i in range(2):
        for j in [0, 2]:
            axs[i, j].set_xlabel("Time (s)")
            axs[i, j].legend(["With holonomic constraint", "Without holonomic constraint"])
        axs[i, 1].set_xlabel("Time (s)")
        axs[i, 1].legend(["With holonomic constraint"])

    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Constraint force (N)")
    plt.title("Constraint force on q_Seg1_TransZ-0")
    dt = sol.time[1] - sol.time[0]
    plt.plot(sol.time, sol.states["lambdas"][0] / dt, "purple")

    plt.show()


if __name__ == "__main__":
    main()
