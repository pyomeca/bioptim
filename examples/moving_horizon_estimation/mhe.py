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

import time
from copy import copy

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import casadi as cas
import numpy as np
import biorbd
from bioptim import (
    MovingHorizonEstimator,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    InterpolationType,
    Solver,
)


def states_to_markers(biorbd_model, states):
    nq = biorbd_model.nbQ()
    n_mark = biorbd_model.nbMarkers()
    q = cas.MX.sym("q", nq)
    markers_func = biorbd.to_casadi_func("makers_kyn", biorbd_model.markers, q)
    return np.array(markers_func(states[:nq, :])).reshape((3, n_mark, -1), order="F")


def generate_data(biorbd_model, tf, x0, t_max, n_shoot, noise_std, show_plots=False):
    def pendulum_ode(t, x, u):
        return np.concatenate((x[nq:, np.newaxis], qddot_func(x[:nq], x[nq:], u)))[:, 0]

    nq = biorbd_model.nbQ()
    q = cas.MX.sym("q", nq)
    qdot = cas.MX.sym("qdot", nq)
    tau = cas.MX.sym("tau", nq)
    qddot_func = biorbd.to_casadi_func("forw_dyn", biorbd_model.ForwardDynamics, q, qdot, tau)

    # Simulated data
    dt = tf / n_shoot
    controls = np.zeros((biorbd_model.nbGeneralizedTorque(), n_shoot))  # Control trajectory
    controls[0, :] = (-np.ones(n_shoot) + np.sin(np.linspace(0, tf, num=n_shoot))) * t_max
    states = np.zeros((biorbd_model.nbQ() + biorbd_model.nbQdot(), n_shoot))  # State trajectory

    for n in range(n_shoot):
        sol = solve_ivp(pendulum_ode, [0, dt], x0, args=(controls[:, n],))
        states[:, n] = x0
        x0 = sol["y"][:, -1]
    states[:, -1] = x0
    markers = states_to_markers(biorbd_model, states[: biorbd_model.nbQ(), :])

    # Simulated noise
    np.random.seed(42)
    noise = (np.random.randn(3, biorbd_model.nbMarkers(), n_shoot) - 0.5) * noise_std
    markers_noised = markers + noise

    if show_plots:
        q_plot = plt.plot(states[:nq, :].T)
        dq_plot = plt.plot(states[nq:, :].T, "--")
        name_dof = [name.to_string() for name in biorbd_model.nameDof()]
        plt.legend(q_plot + dq_plot, name_dof + ["d" + name for name in name_dof])
        plt.title("Real position and velocity trajectories")

        plt.figure()
        marker_plot = plt.plot(markers[1, :, :].T, markers[2, :, :].T)
        plt.legend(marker_plot, [i.to_string() for i in biorbd_model.markerNames()])
        plt.gca().set_prop_cycle(None)
        plt.plot(markers_noised[1, :, :].T, markers_noised[2, :, :].T, "x")
        plt.title("2D plot of markers trajectories + noise")
        plt.show()

    return states, markers, markers_noised, controls


def prepare_mhe(
    biorbd_model_path,
    window_len,
    window_duration,
    max_torque,
    x_init,
    u_init,
    interpolation=InterpolationType.EACH_FRAME,
):
    biorbd_model = biorbd.Model(biorbd_model_path)
    return MovingHorizonEstimator(
        biorbd_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        window_len,
        window_duration,
        x_init=InitialGuess(x_init, interpolation=interpolation),
        u_init=InitialGuess(u_init, interpolation=interpolation),
        x_bounds=QAndQDotBounds(biorbd_model),
        u_bounds=Bounds([-max_torque, 0.0], [max_torque, 0.0]),
        n_threads=4,
    )


def get_solver_options(solver):
    mhe_dict = {"solver": None, "solver_options": None, "solver_options_first_iter": None}
    if solver == Solver.ACADOS:
        mhe_dict["solver"] = Solver.ACADOS
        mhe_dict["solver_options"] = {
            "nlp_solver_max_iter": 1000,
            "integrator_type": "ERK",
        }
    elif solver == Solver.IPOPT:
        mhe_dict["solver"] = Solver.IPOPT
        mhe_dict["solver_options"] = {
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 50,
            "max_iter": 5,
            "print_level": 0,
            "tol": 1e-1,
            "linear_solver": "ma57",
            "bound_frac": 1e-10,
            "bound_push": 1e-10,
        }
        mhe_dict["solver_options_first_iter"] = copy(mhe_dict["solver_options"])
        mhe_dict["solver_options_first_iter"]["max_iter"] = 50
        mhe_dict["solver_options_first_iter"]["tol"] = 1e-6
    else:
        raise NotImplementedError("Solver not recognized")

    return mhe_dict


def main():
    biorbd_model_path = "./cart_pendulum.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)

    solver = Solver.ACADOS
    final_time = 5
    n_shoot_per_second = 100
    window_len = 10
    window_duration = 1 / n_shoot_per_second * window_len
    n_frames_total = final_time * n_shoot_per_second - window_len - 1

    x0 = np.array([0, np.pi / 2, 0, 0])
    noise_std = 0.05  # STD of noise added to measurements
    torque_max = 2  # Max torque applied to the model
    states, markers, markers_noised, controls = generate_data(
        biorbd_model, final_time, x0, torque_max, n_shoot_per_second * final_time, noise_std, show_plots=False
    )

    x0 = np.zeros((biorbd_model.nbQ() * 2, window_len + 1))
    u0 = np.zeros((biorbd_model.nbQ(), window_len))
    torque_max = 5  # Give a bit of slack on the max torque

    mhe = prepare_mhe(
        biorbd_model_path,
        window_len=window_len,
        window_duration=window_duration,
        max_torque=torque_max,
        x_init=x0,
        u_init=u0,
    )

    new_objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, weight=1000, list_index=0)
    mhe.update_objectives(new_objectives)

    def update_functions(mhe, t, _):
        def target(i: int):
            return markers_noised[:, :, i : i + window_len + 1]

        if t == 1:
            # Start the timer after having compiled
            timer.append(time.time())

        new_objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, weight=1000, target=target(t), list_index=0)
        mhe.update_objectives(new_objectives)
        # mhe.update_objectives_target(list_index=0, target=target(t))

        return t < n_frames_total  # True if there are still some frames to reconstruct

    # Solve the program
    timer = []  # Do not start timer yet (because of ACADOS compilation
    sol = mhe.solve(update_functions, **get_solver_options(solver))

    timer.append(time.time())
    print("ACADOS with BiorbdOptim")
    print(f"Window size of MHE : {window_duration} s.")
    print(f"New measurement every : {1/n_shoot_per_second} s.")
    print(f"Average time per iteration of MHE : {(timer[1]-timer[0])/(n_frames_total - 1)} s.")
    print(f"Norm of the error on state = {np.linalg.norm(states[:,:n_frames_total] - sol.states['all'])}")

    markers_estimated = states_to_markers(biorbd_model, sol.states["all"])

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
    plt.plot(sol.states["all"].T, "--", label="States estimate")
    plt.plot(states[:, :n_frames_total].T, label="State truth")
    plt.legend()
    plt.show()

    sol.animate()


if __name__ == "__main__":
    main()
