from mhe_simulation import run_simulation, check_results
import biorbd
import numpy as np
import time

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    ObjectiveList,
    Objective,
    ConstraintList,
    Constraint,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    InterpolationType,
    Data,
)


def warm_start_mhe(data_sol_prev):
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
    data_to_track=[],
    interpolation_type=InterpolationType.EACH_FRAME,
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
    objective_functions.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000, data_to_track=data_to_track)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=0, data_to_track=0, states_idx=0)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:, 0] = -np.inf
    x_bounds[0].max[:, 0] = np.inf
    # X_bounds[0].min[:biorbd_model.nbQ(), 0] = X0[:biorbd_model.nbQ(),0]
    # X_bounds[0].max[:biorbd_model.nbQ(), 0] = X0[:biorbd_model.nbQ(),0]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min, 0.0], [tau_max, 0.0]])

    # Initial guesses
    x = X0
    u = U0
    x_init = InitialConditionsList()
    x_init.add(x, interpolation=interpolation_type)

    u_init = InitialConditionsList()
    u_init.add(u, interpolation=interpolation_type)
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
        nb_threads=1,
        use_SX=True,
    )


def plot_true_X(q_to_plot):
    return X_[q_to_plot, :]


def plot_true_U(q_to_plot):
    return U_[q_to_plot, :]


if __name__ == "__main__":

    biorbd_model_path = "./cart_pendulum.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)

    Tf = 10  # duration of the simulation
    X0 = np.array([0, np.pi / 2, 0, 0])
    N = Tf * 50  # number of shooting nodes per sec
    noise_std = 0.05  # STD of noise added to measurements
    T_max = 2  # Max torque applied to the model
    N_mhe = 25  # size of MHE window
    Tf_mhe = Tf / N * N_mhe  # duration of MHE window

    X_, Y_, Y_N_, U_ = run_simulation(biorbd_model, Tf, X0, T_max, N, noise_std, SHOW_PLOTS=False)

    X0 = np.zeros((biorbd_model.nbQ() * 2, N_mhe))
    U0 = np.zeros((biorbd_model.nbQ(), N_mhe - 1))
    X_est = np.zeros((biorbd_model.nbQ() * 2, N - N_mhe))
    T_max = 5  # Give a bit of slack on the max torque

    Y_i = Y_N_[:, :, :N_mhe]
    ocp = prepare_ocp(
        biorbd_model_path,
        number_shooting_points=N_mhe - 1,
        final_time=Tf_mhe,
        max_torque=T_max,
        X0=X0,
        U0=U0,
        data_to_track=Y_i,
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
    }
    # sol = ocp.solve(solver_options=options_ipopt)
    sol = ocp.solve(solver="acados", solver_options=options_acados)
    data_sol = Data.get_data(ocp, sol)
    X0, U0, X_out = warm_start_mhe(data_sol)
    X_est[:, 0] = X_out
    t0 = time.time()

    # Reduce ipopt tol for moving estimation
    options_ipopt["max_iter"] = 4
    options_ipopt["tol"] = 1e-1

    for i in range(1, N - N_mhe):
        Y_i = Y_N_[:, :, i : i + N_mhe]
        new_objectives = ObjectiveList()
        new_objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000, data_to_track=Y_i)
        new_objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=0, data_to_track=X0.T)
        ocp.modify_objective_function(new_objectives[0][0], 0)
        ocp.modify_objective_function(new_objectives[0][1], 1)  # This doesn't work, but that doesn't seem to related to the merge

        # sol = ocp.solve(solver_options=options_ipopt)
        sol = ocp.solve(solver="acados", solver_options=options_acados)
        data_sol = Data.get_data(ocp, sol)
        X0, U0, X_out = warm_start_mhe(data_sol)
        X_est[:, i] = X_out
    t1 = time.time()
    print(f"Window size of MHE : {Tf_mhe} s.")
    print(f"New measurement every : {Tf/N} s.")
    print(f"Average time per iteration of MHE : {(t1-t0)/(N-N_mhe-2)} s.")
    print(f"Norm of the error on state = {np.linalg.norm(X_[:,:-N_mhe] - X_est)}")

    Y_est = check_results(biorbd_model, N - N_mhe, X_est)
    # Print estimation vs truth
    import matplotlib.pyplot as plt

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
