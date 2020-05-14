import numpy as np
import biorbd
from IPython import embed
 
from biorbd_optim import (
    Instant, # a revoir
    OptimalControlProgram,
    Constraint,
    Objective,
    ProblemType,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
)


def prepare_ocp(show_online_optim=False,):

    # Model path
    biorbd_model = (biorbd.Model("/home/user/Documents/Programmation/Eve/Modeles/ModeleBatons_2D_CAS.bioMod"),)
    
    # Problem parameters
    number_shooting_intervals = 300
    phase_time = 1.22 
    
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = nb_q
    nb_root = 6
    nb_tau = nb_q - nb_root
    
    # tolerances on final position of the root
    tolTilt = 15   # Tilt angle 
    tolSom = 10   # Somersault angle
    
    
    # Path constraints
    QinitMin = [-0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -2.8-0.0001, 2.8-0.0001]
    QinitMax = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -2.8+0.0001, 2.8+0.0001]
    VinitMin = [-10, -10, -10, -100, -0.0001, -0.0001, -0.0001, -0.0001]
    VinitMax = [10, 10, 10, 100, 0.0001, 0.0001, 0.0001, 0.0001]
    
    Qmin = [-1000000, -1000000, -1000000, -1000000, -np.pi/4, -1000000, -np.pi, 0 ] 
    Qmax = [ 1000000,  1000000, 1000000,   1000000,  np.pi/4,  1000000,  0,   np.pi]
    Vmin = [  -1000000, -1000000,  -1000000,  -1000000,  -1000000, -1000000, -100, -100]
    Vmax = [  1000000,  1000000,   1000000,   1000000,   1000000,  1000000,  100,  100]
    
    Qfinalmin = [-1, -1, -0.1, 2*np.pi-tolSom*np.pi/180, -tolTilt/180*np.pi, -1000000, -2.9, 2.7]
    Qfinalmax = [ 1,  1, 0.1,   2*np.pi+tolSom*np.pi/180,  tolTilt/180*np.pi,  1000000, -2.7, 2.9]
    Vfinalmin = [ -1000000, -1000000,  -1000000,  -1000000,  -1000000, -1000000, -100, -100]
    Vfinalmax = [  1000000,  1000000,   1000000,   1000000,   1000000,  1000000,  100,  100]
    
    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model[0])]
    X_bounds[0].first_node_min = QinitMin + VinitMin
    X_bounds[0].first_node_max = QinitMax + VinitMax
    
    X_bounds[0].min = Qmin + Vmin
    X_bounds[0].max = Qmax + Vmax    
    
    X_bounds[0].last_node_min = Qfinalmin + Vfinalmin
    X_bounds[0].last_node_max = Qfinalmax + Vfinalmax
    
    
    # Define control path constraint
    torque_min, torque_max, torque_init = -50, 50, 0
    
    U_bounds = [Bounds(min_bound=[0] * nb_root + [torque_min] * nb_tau, max_bound=[0] * nb_root + [torque_max] * nb_tau)]


    # States and controles initialization
    X_init = [InitialConditions([0] * nb_q + [0] * nb_qdot)]
    
    U_init = [InitialConditions([0] * nb_root + [torque_init] * nb_tau)]


    # Add objective functions
    objective_functions = (
        (
            {"type": Objective.Mayer.MINIMIZE_STATE, "instant": number_shooting_intervals-1, "states_idx": 5, "data_to_track": np.zeros((number_shooting_intervals+1,nb_q)), "weight": -1}, # "instant": Instant.END ??
            # {"type": Objective.Lagrange.MINIMIZE_ALL_CONTROLS, "weight": 1 / 100},
        ),
    )


    # Dynamics
    problem_type = (ProblemType.torque_driven,) 
    


    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_intervals, # points ?
        phase_time,
        objective_functions,
        X_init, #GuessInitial, # 
        U_init,
        X_bounds,
        U_bounds,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    from matplotlib import pyplot as plt
    from casadi import vertcat, Function 


    for i, nlp in enumerate(ocp.nlp):
        q, q_dot, u = ProblemType.get_data(ocp, sol["x"], i)
        # x = vertcat(q, q_dot)
        nb_q = np.shape(q)[0]
        print(nb_q)
    fig, axs = plt.subplots(4,2, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for iplt in range(nb_q):
        axs[iplt].plot(q[iplt,:],'-r', label='Position')
        axs[iplt].plot(q_dot[iplt,:],'-b', label='Velocity')
        if iplt in range(nb_root,nb_q):
            axs[iplt].plot(u[iplt,:],'-b', label='Velocity')
    plt.show()
    
    try:
        from BiorbdViz import BiorbdViz

        x, _, _ = ProblemType.get_data(ocp, sol["x"])
        q = np.ndarray((ocp.nlp[0]["model"].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
        # for i in range(len(ocp.nlp)):
        #     if i == 0:
        #         q[:, : ocp.nlp[i]["ns"]] = ocp.nlp[i]["q_mapping"].expand.map(x[i])[:, :-1]
        #     else:
        #        # q[:, ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + ocp.nlp[i]["ns"]] = ocp.nlp[i][
        #        #     "q_mapping"
        #        # ].expand.map(x[i])[:, :-1]
        # # q[:, -1] = ocp.nlp[-1]["q_mapping"].expand.map(x[-1])[:, -1]

        b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
        b.load_movement(q.T)
        b.exec()
    except ModuleNotFoundError:
        print("Install BiorbdViz if you want to have a live view of the optimization")
        plt.show()





















