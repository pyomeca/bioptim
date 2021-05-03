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
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InterpolationType,
    Solver,
)


def to_markers(states):
    nq = biorbd_model.nbQ()
    n_mark = biorbd_model.nbMarkers()
    q = cas.MX.sym("q", nq)
    markers_func = biorbd.to_casadi_func("makers_kyn", biorbd_model.markers, q)
    return np.array(markers_func(states[:nq, :])).reshape((3, n_mark, -1), order='F')


def generate_data(biorbd_model, tf, x0, t_max, n_shoot, noise_std, show_plots=False):
    nq = biorbd_model.nbQ()
    q = cas.MX.sym("q", nq)
    qdot = cas.MX.sym("qdot", nq)
    tau = cas.MX.sym("tau", nq)

    qddot_func = biorbd.to_casadi_func("forw_dyn", biorbd_model.ForwardDynamics, q, qdot, tau)
    pendulum_ode = lambda t, x, u: np.concatenate((x[nq:, np.newaxis], qddot_func(x[:nq], x[nq:], u)))[:, 0]

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
    markers = to_markers(states[: biorbd_model.nbQ(), :])

    # Simulated noise
    np.random.seed(42)
    noise = (np.random.randn(3, biorbd_model.nbMarkers(), n_shoot) - 0.5) * noise_std
    markers_noised = markers + noise

    if show_plots:
        q_plot = plt.plot(states[: nq, :].T)
        dq_plot = plt.plot(states[nq :, :].T, "--")
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


def prepare_ocp(
    biorbd_model_path,
    n_shooting,
    final_time,
    max_torque,
    x0,
    u0,
    target=None,
    interpolation=InterpolationType.EACH_FRAME,
):
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -max_torque, max_torque, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, weight=1000, target=target)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=100, target=x0)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:, 0] = -10000  # inf
    x_bounds[0].max[:, 0] = 10000  # inf

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min, 0.0], [tau_max, 0.0])

    # Initial guesses
    x_init = InitialGuessList()
    x_init.add(x0, interpolation=interpolation)

    u_init = InitialGuessList()
    u_init.add(u0, interpolation=interpolation)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        n_threads=4,
        use_sx=True,
    )


if __name__ == "__main__":
    biorbd_model_path = "./cart_pendulum.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)

    tf = 5  # duration of the simulation
    x0 = np.array([0, np.pi / 2, 0, 0])
    n_shoot = tf * 100  # number of shooting nodes per sec
    noise_std = 0.05  # STD of noise added to measurements
    t_max = 2  # Max torque applied to the model
    window_size = 10  # size of MHE window
    tf_mhe = tf / n_shoot * window_size  # duration of MHE window

    states, markers, markers_noised, controls = generate_data(biorbd_model, tf, x0, t_max, n_shoot, noise_std, show_plots=False)

    x0 = np.zeros((biorbd_model.nbQ() * 2, window_size + 1))
    x0[:, 0] = np.array([0, np.pi / 2, 0, 0])
    u0 = np.zeros((biorbd_model.nbQ(), window_size))
    x_est = np.zeros((biorbd_model.nbQ() * 2, n_shoot - window_size))
    t_max = 5  # Give a bit of slack on the max torque

    def get_markers(i: int):
        return markers_noised[:, :, i: i + window_size + 1]

    ocp = prepare_ocp(
        biorbd_model_path,
        n_shooting=window_size,
        final_time=tf_mhe,
        max_torque=t_max,
        x0=x0,
        u0=u0,
        target=get_markers(0),
    )
    options_ipopt = {
        "hessian_approximation": "limited-memory",
        "limited_memory_max_history": 50,
        "max_iter": 5,
        "print_level": 0,
        "tol": 1e-1,
        "linear_solver": "ma57",
        "bound_frac": 1e-10,
        "bound_push": 1e-10,
    }
    first_ipopt = copy(options_ipopt)
    first_ipopt["max_iter"] = 50
    first_ipopt["tol"] = 1e-6

    options_acados = {
        "nlp_solver_max_iter": 1000,
        "integrator_type": "ERK",
    }

    def update_objective_functions(ocp, t, _):
        if t == 1:
            # Start the timer after the compiling
            timer.append(time.time())
        new_objectives = ObjectiveList()
        new_objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, weight=1000, target=get_markers(t), list_index=0)
        new_objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=100, target=x0, phase=0, list_index=1)
        ocp.update_objectives(new_objectives)
        return t < n_shoot - window_size - 1

    timer = []
    # sol = ocp.solve(update_objective_functions, solver=Solver.IPOPT, solver_options_first_iter=first_ipopt, solver_options=options_ipopt)
    sol = ocp.solve_mhe(update_objective_functions, solver=Solver.ACADOS, solver_options_first_iter=options_acados)

    timer.append(time.time())
    n_frames = n_shoot - window_size - 1
    print("ACADOS with BiorbdOptim")
    print(f"Window size of MHE : {tf_mhe} s.")
    print(f"New measurement every : {tf/n_shoot} s.")
    print(f"Average time per iteration of MHE : {(timer[1]-timer[0])/(n_shoot-window_size-2)} s.")
    print(f"Norm of the error on state = {np.linalg.norm(states[:,:n_frames] - sol.states['all'])}")

    markers_estimated = to_markers(sol.states['all'])

    plt.plot(markers_noised[1, :, : n_frames].T, markers_noised[2, :, : n_frames].T, "x", label="markers traj noise")
    plt.gca().set_prop_cycle(None)
    plt.plot(markers[1, :, : n_frames].T, markers[2, :, : n_frames].T, label="markers traj truth")
    plt.gca().set_prop_cycle(None)
    plt.plot(markers_estimated[1, :, :].T, markers_estimated[2, :, :].T, "o", label="markers traj est")
    plt.legend()

    plt.figure()
    plt.plot(sol.states["all"].T, "--", label="x estimate")
    plt.plot(states[:, :n_frames].T, label="x truth")
    plt.legend()
    plt.show()
