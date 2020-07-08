import numpy as np
import casadi as cas
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def run_simulation(biorbd_model, Tf, X0, T_max, N, noise_std, SHOW_PLOTS=False):

    ### Casadi functions
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

    ### Simulated data
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

    ### Simulated noise
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

    ### Casadi functions
    x = cas.MX.sym("x", biorbd_model.nbQ() + biorbd_model.nbQdot())
    u = cas.MX.sym("u", 1)
    q = cas.MX.sym("q", biorbd_model.nbQ())

    markers_kyn = cas.Function("makers_kyn", [q], [biorbd_model.markers(q)]).expand()
    Y_est = np.zeros((3, biorbd_model.nbMarkers(), N))  # Measurements trajectory

    for n in range(N):
        Y_est[:, :, n] = markers_kyn(Xs[: biorbd_model.nbQ(), n])

    return Y_est
