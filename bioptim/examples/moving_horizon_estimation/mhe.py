"""
TODO: Cleaning
In this example, mhe (Moving Horizon Estimation) is applied on a simple pendulum simulation. Data are generated (states,
controls, and marker trajectories) to simulate the movement of a pendulum, using scipy.integrate.solve_ivp. These data
are used to perform mhe.

In this example, 500 shooting nodes are defined. As the size of the mhe window is 10, 490 iterations are performed to
solve the complete problem.

For each iteration, the new marker trajectory is taken into account so that a real time data acquisition is simulated.
For each iteration, the list of objectives is updated, the problem is solved with the new frame added to the window,
the oldest frame is discarded with the warm_start_mhe function, and it is saved. The results are plotted so that
estimated data can be compared to real data.
"""

from copy import copy

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import casadi as cas
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    BioModel,
    BiorbdModel,
    MovingHorizonEstimator,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    Solver,
    Node,
)


def states_to_markers(bio_model, states):
    nq = bio_model.nb_q
    n_mark = bio_model.nb_markers
    q = cas.MX.sym("q", nq)
    markers_func = biorbd.to_casadi_func("makers", bio_model.markers, q)
    return np.array(markers_func(states[:nq, :])).reshape((3, n_mark, -1), order="F")


def generate_data(bio_model, tf, x0, t_max, n_shoot, noise_std, show_plots=False):
    def pendulum_ode(t, x, u):
        return np.concatenate((x[nq:, np.newaxis], qddot_func(x[:nq], x[nq:], u)))[:, 0]

    nq = bio_model.nb_q
    q = cas.MX.sym("q", nq)
    qdot = cas.MX.sym("qdot", nq)
    tau = cas.MX.sym("tau", nq)
    qddot_func = biorbd.to_casadi_func("forw_dyn", bio_model.forward_dynamics, q, qdot, tau)

    # Simulated data
    dt = tf / n_shoot
    controls = np.zeros((bio_model.nb_tau, n_shoot))  # Control trajectory
    controls[0, :] = (-np.ones(n_shoot) + np.sin(np.linspace(0, tf, num=n_shoot))) * t_max
    states = np.zeros((bio_model.nb_q + bio_model.nb_qdot, n_shoot))  # State trajectory

    for n in range(n_shoot):
        sol = solve_ivp(pendulum_ode, [0, dt], x0, args=(controls[:, n],))
        states[:, n] = x0
        x0 = sol["y"][:, -1]
    states[:, -1] = x0
    markers = states_to_markers(bio_model, states[: bio_model.nb_q, :])

    # Simulated noise
    np.random.seed(42)
    noise = (np.random.randn(3, bio_model.nb_markers, n_shoot) - 0.5) * noise_std
    markers_noised = markers + noise

    if show_plots:
        q_plot = plt.plot(states[:nq, :].T)
        dq_plot = plt.plot(states[nq:, :].T, "--")
        name_dof = bio_model.name_dof
        plt.legend(q_plot + dq_plot, name_dof + ["d" + name for name in name_dof])
        plt.title("Real position and velocity trajectories")

        plt.figure()
        marker_plot = plt.plot(markers[1, :, :].T, markers[2, :, :].T)
        plt.legend(marker_plot, bio_model.marker_names())
        plt.gca().set_prop_cycle(None)
        plt.plot(markers_noised[1, :, :].T, markers_noised[2, :, :].T, "x")
        plt.title("2D plot of markers trajectories + noise")
        plt.show()

    return states, markers, markers_noised, controls


def prepare_mhe(
    bio_model: BioModel,
    window_len: int,
    window_duration: float,
    max_torque: float,
    x_init: np.ndarray,
    u_init: np.ndarray,
    assume_phase_dynamics: bool = True,
    n_threads: int = 4,
):
    """

    Parameters
    ----------
    bio_model
        The model to perform the optimization on
    window_len:
        The length of the sliding window. It is translated into n_shooting in each individual optimization program
    window_duration
        The time in second of the sliding window
    max_torque
        The maximal torque the model is able to apply
    x_init
        The states initial guess
    u_init
        The controls initial guess
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
    n_threads: int
        Number of threads to use

    Returns
    -------

    """
    new_objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, node=Node.ALL, weight=1000, list_index=0)

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    u_bounds = BoundsList()
    u_bounds["tau"] = [-max_torque, 0.0], [max_torque, 0.0]

    x_init_list = InitialGuessList()
    x_init_list.add("q", x_init[: bio_model.nb_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init_list.add("qdot", x_init[bio_model.nb_q :, :], interpolation=InterpolationType.EACH_FRAME)

    u_init_list = InitialGuessList()
    u_init_list.add("tau", u_init, interpolation=InterpolationType.EACH_FRAME)

    return MovingHorizonEstimator(
        bio_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        window_len,
        window_duration,
        objective_functions=new_objectives,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init_list,
        u_init=u_init_list,
        n_threads=n_threads,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def get_solver_options(solver):
    mhe_dict = {"solver_first_iter": None, "solver": solver}
    if isinstance(solver, Solver.ACADOS):
        mhe_dict["solver"].set_maximum_iterations(1000)
        mhe_dict["solver"].set_print_level(0)
        mhe_dict["solver"].set_integrator_type("ERK")

    elif isinstance(solver, Solver.IPOPT):
        mhe_dict["solver"].set_hessian_approximation("limited-memory")
        mhe_dict["solver"].set_limited_memory_max_history(50)
        mhe_dict["solver"].set_maximum_iterations(5)
        mhe_dict["solver"].set_print_level(0)
        mhe_dict["solver"].set_tol(1e-1)
        mhe_dict["solver"].set_initialization_options(1e-10)

        mhe_dict["solver_first_iter"] = copy(mhe_dict["solver"])
        mhe_dict["solver_first_iter"].set_maximum_iterations(50)
        mhe_dict["solver_first_iter"].set_tol(1e-6)
    else:
        raise NotImplementedError("Solver not recognized")

    return mhe_dict


def main():
    biorbd_model_path = "models/cart_pendulum.bioMod"
    bio_model = BiorbdModel(biorbd_model_path)

    solver = Solver.IPOPT()  # or Solver.ACADOS()  # If ACADOS is used, it must be manually installed
    final_time = 5
    n_shoot_per_second = 100
    window_len = 10
    window_duration = 1 / n_shoot_per_second * window_len
    n_frames_total = final_time * n_shoot_per_second - window_len - 1

    x0 = np.array([0, np.pi / 2, 0, 0])
    noise_std = 0.05  # STD of noise added to measurements
    torque_max = 2  # Max torque applied to the model
    states, markers, markers_noised, controls = generate_data(
        bio_model, final_time, x0, torque_max, n_shoot_per_second * final_time, noise_std, show_plots=False
    )

    x_init = np.zeros((bio_model.nb_q * 2, window_len + 1))
    u_init = np.zeros((bio_model.nb_q, window_len))
    torque_max = 5  # Give a bit of slack on the max torque

    bio_model = BiorbdModel(biorbd_model_path)
    mhe = prepare_mhe(
        bio_model,
        window_len=window_len,
        window_duration=window_duration,
        max_torque=torque_max,
        x_init=x_init,
        u_init=u_init,
    )

    def update_functions(mhe, t, _):
        def target(i: int):
            return markers_noised[:, :, i : i + window_len + 1]

        mhe.update_objectives_target(target=target(t), list_index=0)
        return t < n_frames_total  # True if there are still some frames to reconstruct

    # Solve the program
    sol = mhe.solve(update_functions, **get_solver_options(solver))

    print("ACADOS with Bioptim")
    print(f"Window size of MHE : {window_duration} s.")
    print(f"New measurement every : {1 / n_shoot_per_second} s.")
    print(f"Average time per iteration of MHE : {sol.solver_time_to_optimize / (n_frames_total - 1)} s.")
    print(f"Average real time per iteration of MHE : {sol.real_time_to_optimize / (n_frames_total - 1)} s.")
    print(f"Norm of the error on q = {np.linalg.norm(states[:bio_model.nb_q, :n_frames_total] - sol.states['q'])}")

    markers_estimated = states_to_markers(bio_model, sol.states["q"])

    plt.plot(
        markers_noised[1, :, :n_frames_total].T,
        markers_noised[2, :, :n_frames_total].T,
        "x",
        label="Noised markers trajectory",
    )
    plt.gca().set_prop_cycle(None)
    plt.plot(markers[1, :, :n_frames_total].T, markers[2, :, :n_frames_total].T, label="True markers trajectory")
    plt.gca().set_prop_cycle(None)
    plt.plot(markers_estimated[1, :, :].T, markers_estimated[2, :, :].T, "o", label="Estimated marker trajectory")
    plt.legend()

    plt.figure()
    plt.plot(sol.states["q"].T, "--", label="States estimate (q)")
    plt.plot(sol.states["qdot"].T, "--", label="States estimate (qdot)")
    plt.plot(states[:, :n_frames_total].T, label="State truth")
    plt.legend()
    plt.show()

    sol.animate()


if __name__ == "__main__":
    main()
