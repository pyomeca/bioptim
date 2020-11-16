import time

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import casadi as cas
import numpy as np
import biorbd

from bioptim import (
    Instant,
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    ObjectiveList,
    Objective,
    ConstraintList,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InterpolationType,
    PlotType,
    Data,
    Solver,
)


def generate_data(biorbd_model, Tf, X0, T_max, N, noise_std, SHOW_PLOTS=False):
    # Casadi functions
    x = cas.MX.sym("x", biorbd_model.nbQ() + biorbd_model.nbQdot())
    u = cas.MX.sym("u", 1)
    q = cas.MX.sym("q", biorbd_model.nbQ())

    forw_dyn = cas.Function(
        "forw_dyn",
        [x, u],
        [
            cas.vertcat(
                x[biorbd_model.nbQ() :],
                biorbd_model.ForwardDynamics(
                    x[: biorbd_model.nbQ()], x[biorbd_model.nbQ() :], cas.vertcat(u, 0)
                ).to_mx(),
            )
        ],
    ).expand()
    pendulum_ode = lambda t, x, u: forw_dyn(x, u).toarray().squeeze()

    markers_kyn = cas.Function("makers_kyn", [q], [biorbd_model.markers(q)]).expand()

    # Simulated data
    h = Tf / N
    # U_ = (np.random.rand(N)-0.5)*2*T_max # Control trajectory
    U_ = (-np.ones(N) + np.sin(np.linspace(0, Tf, num=N))) * T_max  # Control trajectory
    X_ = np.zeros((biorbd_model.nbQ() + biorbd_model.nbQdot(), N))  # State trajectory
    Y_ = np.zeros((3, biorbd_model.nbMarkers(), N))  # Measurements trajectory

    for n in range(N):
        sol = solve_ivp(pendulum_ode, [0, h], X0, args=(U_[n],))
        X_[:, n] = X0
        Y_[:, :, n] = markers_kyn(X0[: biorbd_model.nbQ()])
        X0 = sol["y"][:, -1]
    X_[:, -1] = X0
    Y_[:, :, -1] = markers_kyn(X0[: biorbd_model.nbQ()])

    # Simulated noise
    np.random.seed(42)
    N_ = (np.random.randn(3, biorbd_model.nbMarkers(), N) - 0.5) * noise_std
    Y_N_ = Y_ + N_

    if SHOW_PLOTS:
        q_plot = plt.plot(X_[: biorbd_model.nbQ(), :].T)
        dq_plot = plt.plot(X_[biorbd_model.nbQ() :, :].T, "--")
        plt.legend(
            q_plot + dq_plot,
            [i.to_string() for i in biorbd_model.nameDof()] + ["d" + i.to_string() for i in biorbd_model.nameDof()],
        )
        plt.title("Real position and velocity trajectories")
        plt.figure()
        marker_plot = plt.plot(Y_[1, :, :].T, Y_[2, :, :].T)
        plt.legend(marker_plot, [i.to_string() for i in biorbd_model.markerNames()])
        plt.gca().set_prop_cycle(None)
        marker_plot = plt.plot(Y_N_[1, :, :].T, Y_N_[2, :, :].T, "x")
        plt.title("2D plot of markers trajectories + noise")
        plt.show()

    return X_, Y_, Y_N_, np.vstack([U_, np.zeros((N,))])


def check_results(biorbd_model, N, Xs):
    # Casadi functions
    x = cas.MX.sym("x", biorbd_model.nbQ() + biorbd_model.nbQdot())
    u = cas.MX.sym("u", 1)
    q = cas.MX.sym("q", biorbd_model.nbQ())

    markers_kyn = cas.Function("makers_kyn", [q], [biorbd_model.markers(q)]).expand()
    Y_est = np.zeros((3, biorbd_model.nbMarkers(), N))  # Measurements trajectory

    for n in range(N):
        Y_est[:, :, n] = markers_kyn(Xs[: biorbd_model.nbQ(), n])

    return Y_est


def plot_true_X(q_to_plot):
    return X_[q_to_plot, :]


def plot_true_U(q_to_plot):
    return U_[q_to_plot, :]


def warm_start_mhe(data_sol_prev):
    # TODO: This should be moved in a MHE module
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    X0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    U0 = u[:, 1:]  # discard oldest estimate of the window
    X_out = x[:, 0]
    return X0, U0, X_out


def prepare_ocp(
    biorbd_model_path,
    number_shooting_points,
    final_time,
    max_torque,
    X0,
    U0,
    target=None,
    interpolation=InterpolationType.EACH_FRAME,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -max_torque, max_torque, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000, target=target)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=100, target=X0)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:, 0] = -np.inf
    x_bounds[0].max[:, 0] = np.inf
    # x_bounds[0].min[:biorbd_model.nbQ(), 0] = X0[:biorbd_model.nbQ(),0]
    # x_bounds[0].max[:biorbd_model.nbQ(), 0] = X0[:biorbd_model.nbQ(),0]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min, 0.0], [tau_max, 0.0]])

    # Initial guesses
    x = X0
    u = U0
    x_init = InitialGuessList()
    x_init.add(x, interpolation=interpolation)

    u_init = InitialGuessList()
    u_init.add(u, interpolation=interpolation)
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
        objective_functions,
        nb_threads=4,
        use_SX=True,
    )


if __name__ == "__main__":

    biorbd_model_path = "./cart_pendulum.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)

    Tf = 5  # duration of the simulation
    X0 = np.array([0, np.pi / 2, 0, 0])
    N = Tf * 100  # number of shooting nodes per sec
    noise_std = 0.05  # STD of noise added to measurements
    T_max = 2  # Max torque applied to the model
    N_mhe = 10  # size of MHE window
    Tf_mhe = Tf / N * N_mhe  # duration of MHE window

    X_, Y_, Y_N_, U_ = generate_data(biorbd_model, Tf, X0, T_max, N, noise_std, SHOW_PLOTS=False)

    X0 = np.zeros((biorbd_model.nbQ() * 2, N_mhe + 1))
    X0[:, 0] = np.array([0, np.pi / 2, 0, 0])
    U0 = np.zeros((biorbd_model.nbQ(), N_mhe))
    X_est = np.zeros((biorbd_model.nbQ() * 2, N - N_mhe))
    T_max = 5  # Give a bit of slack on the max torque

    Y_i = Y_N_[:, :, : N_mhe + 1]

    ocp = prepare_ocp(
        biorbd_model_path,
        number_shooting_points=N_mhe,
        final_time=Tf_mhe,
        max_torque=T_max,
        X0=X0,
        U0=U0,
        target=Y_i,
    )
    options_ipopt = {
        "hessian_approximation": "limited-memory",
        "limited_memory_max_history": 50,
        "max_iter": 50,
        "print_level": 0,
        "tol": 1e-6,
        "linear_solver": "ma57",
        "bound_frac": 1e-10,
        "bound_push": 1e-10,
    }
    options_acados = {
        "nlp_solver_max_iter": 1000,
        "integrator_type": "ERK",
    }
    # sol = ocp.solve(solver_options=options_ipopt)
    sol = ocp.solve(solver=Solver.ACADOS, solver_options=options_acados)
    data_sol = Data.get_data(ocp, sol)
    X0, U0, X_out = warm_start_mhe(data_sol)
    X_est[:, 0] = X_out
    t0 = time.time()

    # Reduce ipopt tol for moving estimation
    options_ipopt["max_iter"] = 5
    options_ipopt["tol"] = 1e-1

    # TODO: The following loop should be move in a MHE module that yields after iteration so the user can change obj
    for i in range(1, N - N_mhe):
        Y_i = Y_N_[:, :, i : i + N_mhe + 1]
        new_objectives = ObjectiveList()
        new_objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000, target=Y_i, idx=0)
        new_objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=100, target=X0, phase=0, idx=1)

        ocp.update_objectives(new_objectives)

        # sol = ocp.solve(solver_options=options_ipopt)
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=options_acados)
        data_sol = Data.get_data(ocp, sol, concatenate=False)
        X0, U0, X_out = warm_start_mhe(data_sol)
        X_est[:, i] = X_out
    t1 = time.time()
    print("ACADOS with BiorbdOptim")
    print(f"Window size of MHE : {Tf_mhe} s.")
    print(f"New measurement every : {Tf/N} s.")
    print(f"Average time per iteration of MHE : {(t1-t0)/(N-N_mhe-2)} s.")
    print(f"Norm of the error on state = {np.linalg.norm(X_[:,:-N_mhe] - X_est)}")

    Y_est = check_results(biorbd_model, N - N_mhe, X_est)
    # Print estimation vs truth

    plt.plot(Y_N_[1, :, : N - N_mhe].T, Y_N_[2, :, : N - N_mhe].T, "x", label="markers traj noise")
    plt.gca().set_prop_cycle(None)
    plt.plot(Y_[1, :, : N - N_mhe].T, Y_[2, :, : N - N_mhe].T, label="markers traj truth")
    plt.gca().set_prop_cycle(None)
    plt.plot(Y_est[1, :, :].T, Y_est[2, :, :].T, "o", label="markers traj est")
    plt.legend()

    plt.figure()
    plt.plot(X_est.T, "--", label="x estimate")
    plt.plot(X_[:, :-N_mhe].T, label="x truth")
    plt.legend()
    plt.show()
