from mhe_simulation import run_simulation
import biorbd
import numpy as np

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    InterpolationType,
    PlotType,
    Data
)

def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, max_torque, X0, data_to_track = []):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    torque_min, torque_max, torque_init = -max_torque, max_torque, 0

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_MARKERS, "weight": 1000, "data_to_track" : data_to_track}

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, 0] = X0
    X_bounds.max[:, 0] = X0


    # Define control path constraint
    U_bounds = Bounds([torque_min, 0.], [torque_max, 0.],)

    # Initial guesses
    x = X0
    u = [torque_init] * ntau
    X_init = InitialConditions(x, interpolation_type=InterpolationType.CONSTANT)
    U_init = InitialConditions(u, interpolation_type=InterpolationType.CONSTANT)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
    )

def plot_true_X(q_to_plot):
    return X_[q_to_plot, :]

def plot_true_U(q_to_plot):
    return U_[q_to_plot, :]

if __name__ == "__main__":

    biorbd_model_path = "./cart_pendulum.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)

    Tf = 4
    X0 = [0, np.pi / 2, 0, 0]
    N = Tf * 50
    noise_std = 0.1
    T_max = 2

    X_, Y_, Y_N_, U_ = run_simulation(biorbd_model, Tf, X0, T_max, N, noise_std, SHOW_PLOTS=True)
    ocp = prepare_ocp(biorbd_model_path, number_shooting_points=N-1, final_time=Tf, max_torque=T_max,
                      X0=X0 , data_to_track=Y_N_)
    sol = ocp.solve()
    print("\n")

    data_sol = Data.get_data(ocp, sol)
    q_sol, dq_sol = data_sol[0]['q'], data_sol[0]['q_dot']
    tau_sol = data_sol[1]['tau']

    print(f'Error on q = {np.linalg.norm(q_sol-X_[:biorbd_model.nbQ()])}')
    print(f'Error on dq = {np.linalg.norm(dq_sol-X_[biorbd_model.nbQ():])}')
    print(f'Error on tau = {np.linalg.norm(tau_sol-U_)}')
    # Print estimation and truth
    estimation_plot = ShowResult(ocp, sol)
    ocp.add_plot("True states", lambda x, u: plot_true_X([0, 1, 2, 3]), PlotType.PLOT)
    ocp.add_plot("True control", lambda x, u: plot_true_U([0, 1]), PlotType.PLOT)
    estimation_plot.graphs()


