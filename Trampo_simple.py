import numpy as np
import biorbd
from IPython import embed
 
from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    Constraint,
    Objective,
    ProblemType,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
)


def prepare_ocp(show_online_optim=False,):

    # --- Options --- #
    # Model path
    biorbd_model = (biorbd.Model("/home/laboratoire/mnt/Serveur2/clientbd8/Documents/muscod/Saut_Eve/S2M_Modeles/Jumper/ModeleBatons_2D_CAS.bioMod"),)
    
    nb_q = biorbd_model[0].nbQ()
    # nb_q = q_mapping.reduce.len
    nb_qdot = nb_q
    nb_root = 6
    nb_tau = nb_q - nb_root
    
    tolTilt = 15
    tolSalto = 10
    nb_salto = 1
    vz0 = 6 #vitesse verticale
    
    # nb_phases = 1
    torque_min, torque_max, torque_init = -50, 50, 0

    # Problem parameters
    number_shooting_intervals = 300
    phase_time = 1.22 # temps fixe !

    # Add objective functions
    objective_functions = (
        (
            {"type": Objective.Mayer.MINIMIZE_STATE, "instant": number_shooting_intervals-1, "states_idx": 5, "data_to_track": np.zeros((number_shooting_intervals+1,nb_q)), "weight": -1}, # "instant": Instant.END
            {"type": Objective.Lagrange.MINIMIZE_ALL_CONTROLS, "weight": 1 / 100},
        ),
    )

    # Dynamics
    problem_type = (ProblemType.torque_driven,) 

    # constraints_first_phase = []

    # # contact_axes = (1, 2, 4, 5)
    # # for i in contact_axes:
    # #     constraints_first_phase.append(
    # #         {"type": Constraint.CONTACT_FORCE_GREATER_THAN, "instant": Instant.ALL, "idx": i, "boundary": 0,}
    # #     )
    # # contact_axes = (1, 3)
    # # for i in contact_axes:
    # #     constraints_second_phase.append(
    # #         {"type": Constraint.CONTACT_FORCE_GREATER_THAN, "instant": Instant.ALL, "idx": i, "boundary": 0,}
    # #     )
    
    # constraints_first_phase.append(
    #     {
    #         "type": Constraint.MINIMIZE_STATE,
    #         "instant": number_shooting_points,
    #         # "normal_component_idx": (1, 2, 4, 5),
    #         # "tangential_component_idx": (0, 3),
    #         # "static_friction_coefficient": 0.5,
    #     }
    # )


    # # first_dof = (3, 4, 7, 8, 9)
    # # second_dof = (5, 6, 10, 11, 12)
    # # coeff = (-1, 1, 1, 1, 1)
    # # for i in range(len(first_dof)):
    # #     constraints_first_phase.append(
    # #         {
    # #             "type": Constraint.PROPORTIONAL_STATE,
    # #             "instant": Instant.ALL,
    # #             "first_dof": first_dof[i],
    # #             "second_dof": second_dof[i],
    # #             "coef": coeff[i],
    # #         }
    # #     )

    # constraints = constraints_first_phase

    # Path constraint
    QinitMin = [-0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -2.8-0.0001, 2.8-0.0001]
    QinitMax = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, -2.8+0.0001, 2.8+0.0001]
    VinitMin = [-10, -10, -10, -100, -0.0001, -0.0001, -0.0001, -0.0001] # Kin.V1(3,1)
    VinitMax = [10, 10, 10, 100, 0.0001, 0.0001, 0.0001, 0.0001] # Kin.V1(3,1)
    
    Qmin = [-1000000, -1000000, -1000000, -1000000, -np.pi/4, -1000000, -np.pi, 0 ] 
    Qmax = [ 1000000,  1000000, 1000000,   1000000,  np.pi/4,  1000000,  0,   np.pi]
    Vmin = [  -1000000, -1000000,  -1000000,  -1000000,  -1000000, -1000000, -100, -100]
    Vmax = [  1000000,  1000000,   1000000,   1000000,   1000000,  1000000,  100,  100]
    
    Qfinalmin = [-1, -1, -0.1, 2*np.pi*nb_salto-tolSalto*np.pi/180, -tolTilt/180*np.pi, -1000000, -2.9, 2.7]  # Tx et Ty -0.1
    Qfinalmax = [ 1,  1, 0.1,   2*np.pi*nb_salto+tolSalto*np.pi/180,  tolTilt/180*np.pi,  1000000, -2.7, 2.9]
    Vfinalmin = [ -1000000, -1000000,  -1000000,  -1000000,  -1000000, -1000000, -100, -100]
    Vfinalmax = [  1000000,  1000000,   1000000,   1000000,   1000000,  1000000,  100,  100]
    
    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model[0])] #, q_mapping=q_mapping, q_dot_mapping=q_mapping
    X_bounds[0].first_node_min = QinitMin + VinitMin
    X_bounds[0].first_node_max = QinitMax + VinitMax
    
    X_bounds[0].min = Qmin + Vmin
    X_bounds[0].max = Qmax + Vmax    
    
    X_bounds[0].last_node_min = Qfinalmin + Vfinalmin
    X_bounds[0].last_node_max = Qfinalmax + Vfinalmax
    
    
    # Initial guess
    # X_init = [InitialConditions([0] * nb_q + [0] * nb_qdot)]
    Interp_conditionInitiale = np.zeros((nb_q + nb_qdot, number_shooting_intervals+1))
    Interp_conditionInitiale[3,:] = np.linspace(0,nb_salto*2*np.pi, number_shooting_intervals+1)
    Interp_conditionInitiale[6,0] = -2.8
    Interp_conditionInitiale[6,1:-2] = np.pi*np.random.rand(1,number_shooting_intervals-2) - np.pi
    Interp_conditionInitiale[6,-1] = -2.8
    Interp_conditionInitiale[7,0] = 2.8
    Interp_conditionInitiale[7,1:-2] = np.pi*np.random.rand(1,number_shooting_intervals-2)
    Interp_conditionInitiale[7,-1] = 2.8
    
    Interp_conditionInitiale[nb_q + 2,:] = -9.81*np.linspace(0,phase_time,number_shooting_intervals+1) + vz0
    Interp_conditionInitiale[nb_q + 3,:] = np.linspace(0,(2*np.pi*nb_salto)/phase_time,number_shooting_intervals+1)

    Interp_Uinitial = 5*np.random.rand(4,number_shooting_intervals+1) # 50/10
    
    # %petits indices pour muscod guess initiaux explicites
    # kin.Q1(3,:) = vz0*linspace(0,data.Duration,length(kin.Q1(3,:)))-4.9*linspace(0,data.Duration,length(kin.Q1(3,:))).^2;
    # kin.Q1(6,:) = linspace(0, choix.vrille*2*pi, length(kin.Q1(6,:)));
    # kin.V1(6,2:end) = choix.vrille/data.Duration;
        
    
    # InitialCond = np.ndarray.tolist(Interp_conditionInitiale)
    
    # X_init = [InitialConditions(InitialCond)]
    X_init = [
    InitialConditions(Interp_conditionInitiale[:,0]),
    ]
        
    GuessInitial = []
    for iGuess in range(number_shooting_intervals+1):
        GuessInitial += [Interp_Uinitial[:,iGuess]]
        GuessInitial += [Interp_conditionInitiale[:,iGuess]]
    
    # Define control path constraint
    U_bounds = [Bounds(min_bound=[0] * nb_root + [torque_min] * nb_tau, max_bound=[0] * nb_root + [torque_max] * nb_tau)]

    U_init = [InitialConditions([0] * nb_root + [torque_init] * nb_tau)]


    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_intervals, # points ?
        phase_time,
        objective_functions,
        GuessInitial, # X_init,
        U_init,
        X_bounds,
        Interp_Uinitial, # U_bounds,
        # constraints,
        # q_mapping=q_mapping,
        # q_dot_mapping=q_mapping,
        # tau_mapping=tau_mapping,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=False) # True

    # --- Solve the program --- #
    sol = ocp.solve()

    from matplotlib import pyplot as plt
    from casadi import vertcat, Function 
    
    #########

    contact_forces = np.zeros((6, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    cs_map = (range(6), (0, 1, 3, 4))

    for i, nlp in enumerate(ocp.nlp):
        CS_func = Function(
            "Contact_force_inequality",
            [ocp.symbolic_states, ocp.symbolic_controls],
            [nlp["model"].getConstraints().getForce().to_mx()],
            ["x", "u"],
            ["CS"],
        ).expand()

        q, q_dot, u = ProblemType.get_data(ocp, sol["x"], i)
        x = vertcat(q, q_dot)
        if i == 0:
            contact_forces[cs_map[i], : nlp["ns"] + 1] = CS_func(x, u)
        else:
            contact_forces[cs_map[i], ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = CS_func(x, u)
    plt.plot(contact_forces.T)
    plt.show()
    
    try:
        from BiorbdViz import BiorbdViz

        x, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
        q = np.ndarray((ocp.nlp[0]["model"].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
        for i in range(len(ocp.nlp)):
            if i == 0:
                q[:, : ocp.nlp[i]["ns"]] = ocp.nlp[i]["q_mapping"].expand.map(x[i])[:, :-1]
            else:
                q[:, ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + ocp.nlp[i]["ns"]] = ocp.nlp[i][
                    "q_mapping"
                ].expand.map(x[i])[:, :-1]
        q[:, -1] = ocp.nlp[-1]["q_mapping"].expand.map(x[-1])[:, -1]

        b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
        b.load_movement(q.T)
        b.exec()
    except ModuleNotFoundError:
        print("Install BiorbdViz if you want to have a live view of the optimization")
        plt.show()





















