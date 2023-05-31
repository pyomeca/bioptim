"""
A pendulum simulation copying the example from bioptim/examples/getting_started/pendulum.py but integrated by the
variational integrator.
"""
from bioptim import (
    Bounds,
    InitialGuess,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
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
    n_q = bio_model.nb_q

    bio_model.stabilization = True
    gamma = 50
    bio_model.alpha = gamma ^ 2
    bio_model.beta = 2 * gamma

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=1, max_bound=2)

    # Path constraint
    q_bounds = bio_model.bounds_from_ranges(["q"])
    x_bounds = Bounds(
        [q_bounds.min[0, 0], q_bounds.min[1, 0], q_bounds.min[2, 0], q_bounds.min[3, 0], -20, -20],
        [q_bounds.max[0, 0], q_bounds.max[1, 0], q_bounds.max[2, 0], q_bounds.max[3, 0], 20, 20],
    )
    # Initial guess
    q_t0 = np.array([1.54, 1.54])  # change this line for automatic initial guess that satisfies the constraints
    # Translations between Seg0 and Seg1 at t0, calculated with cos and sin as Seg1 has no parent
    t_t0 = np.array([np.sin(q_t0[0]), -np.cos(q_t0[0])])
    all_x_t0 = np.array([q_t0[0], t_t0[0], t_t0[1], q_t0[1], 0, 0])
    x_init = InitialGuess(all_x_t0)
    x_bounds[:4, 0] = all_x_t0[:4]

    # Initial and final velocities
    qdot0_bounds = Bounds([0.0] * n_q, [0.0] * n_q, interpolation=InterpolationType.CONSTANT)
    qdotN_bounds = Bounds([-10 * np.pi] * n_q, [10 * np.pi] * n_q, interpolation=InterpolationType.CONSTANT)
    qdot0_init = InitialGuess([0.0] * n_q)
    qdotN_init = InitialGuess([0.0] * n_q)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -1, 1, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    # Initial guess
    u_init = InitialGuess([tau_init] * n_q)

    # Holonomic constraints
    constraints, jac = bio_model.generate_constraint_and_jacobian_functions(
        marker_1="marker_1", marker_2="marker_3", index=slice(1, 3)
    )

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
    ocp = prepare_ocp(bio_model_path="models/two_pendulums.bioMod", final_time=1, n_shooting=n_shooting)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show the results in a bioviz animation --- #
    sol.detailed_cost_values()  # /!\ Since the last controls are nan the costs are not accurate /!\
    sol.print_cost()
    sol.animate()

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].set_title("q_Seg1_TransY-0")
    axs[0, 0].plot(sol.time, sol.states["q"][0], "purple")
    axs[0, 1].set_title("q_Seg1_TransZ-0")
    axs[0, 1].plot(sol.time, sol.states["q"][1], "purple")
    axs[0, 2].set_title("q_Seg1_RotX-0")
    axs[0, 2].plot(sol.time, sol.states["q"][2], "purple")
    axs[1, 0].set_title("tau_Seg1_TransY-0")
    axs[1, 0].step(sol.time, sol.controls["tau"][0], "orange")
    axs[1, 1].set_title("tau_Seg1_TransZ-0")
    axs[1, 1].step(sol.time, sol.controls["tau"][1], "orange")
    axs[1, 2].set_title("tau_Seg1_RotX-0")
    axs[1, 2].step(sol.time, sol.controls["tau"][2], "orange")

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
