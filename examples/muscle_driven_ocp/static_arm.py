import biorbd
import numpy as np
from matplotlib import pyplot as plt

import biorbd_optim
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.dynamics import Dynamics


def prepare_nlp(biorbd_model_path="arm26.bioMod"):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Results path
    optimization_name = "eocar"
    results_path = "Results/"
    control_results_file_name = results_path + "Controls" + optimization_name + ".txt"
    state_results_file_name = results_path + "States" + optimization_name + ".txt"

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    ode_solver = biorbd_optim.OdeSolver.RK
    velocity_max = 15
    is_cyclic_constraint = False
    is_cyclic_objective = False

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

    # Dynamics
    variable_type = biorbd_optim.Variable.muscle_and_torque_driven
    dynamics_func = Dynamics.forward_dynamics_torque_muscle_driven

    # Constraints
    constraints = (
        #(Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),
    )

    # Path constraint
    X_bounds = biorbd_optim.Bounds()
    X_init = biorbd_optim.InitialConditions()

    # Gets bounds from biorbd model
    ranges = []
    for i in range(biorbd_model.nbSegment()):
        ranges.extend(
            [
                biorbd_model.segment(i).ranges()[j]
                for j in range(len(biorbd_model.segment(i).ranges()))
            ]
        )
    X_bounds.min = [ranges[i].min() for i in range(biorbd_model.nbQ())]
    X_bounds.max = [ranges[i].max() for i in range(biorbd_model.nbQ())]

    X_bounds.first_node_min = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.first_node_min[0] = X_bounds.min[0]
    X_bounds.first_node_max = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.first_node_max[0] = X_bounds.max[0]

    X_bounds.last_node_min = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.last_node_min[0] = X_bounds.min[0]
    X_bounds.last_node_min[2] = 1.57
    X_bounds.last_node_max = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.last_node_max[0] = X_bounds.max[0]
    X_bounds.last_node_max[2] = 1.57

    # Path constraint velocity
    velocity_max = 15
    X_bounds.min.extend([-velocity_max] * (biorbd_model.nbQdot()))
    X_bounds.max.extend([velocity_max] * (biorbd_model.nbQdot()))

    # Initial guess
    X_init.init = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())

    # Define control path constraint
    U_bounds = biorbd_optim.Bounds()
    U_init = biorbd_optim.InitialConditions()

    muscle_min = 0
    muscle_max = 1
    muscle_init = 0.1

    U_bounds.min = [muscle_min for _ in range(biorbd_model.nbMuscleTotal())]
    U_bounds.max = [muscle_max for _ in range(biorbd_model.nbMuscleTotal())]
    U_init.init = [muscle_init for _ in range(biorbd_model.nbMuscleTotal())]

    torque_min = -100
    torque_max = 100
    torque_init = 0

    U_bounds.min.extend([torque_min for _ in range(biorbd_model.nbGeneralizedTorque())])
    U_bounds.max.extend([torque_max for _ in range(biorbd_model.nbGeneralizedTorque())])
    U_init.init.extend([torque_init for _ in range(biorbd_model.nbGeneralizedTorque())])
    # ------------- #

    return biorbd_optim.OptimalControlProgram(
        biorbd_model,
        variable_type,
        dynamics_func,
        ode_solver,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        is_cyclic_constraint=is_cyclic_constraint,
        is_cyclic_objective=is_cyclic_objective,
    )


def multi_plots():
    nlp = prepare_nlp()
    sol = nlp.solve()

    step = nlp.nu + nlp.nx
    npy_fichier = sol["x"]
    # for k in range (nlp.ns):
    #     npy_fichier.append(sol["x"][k*step])
    #     npy_fichier.append(sol["x"][k*step+1])
    #     npy_fichier.append(sol["x"][k*step+2])
    #     npy_fichier.append(sol["x"][k*step+3])
    #     npy_fichier.append(sol["x"][k*step+4])
    #     npy_fichier.append(sol["x"][k*step+5])
    np.save("static_arm", np.array(npy_fichier))

    muscles = []

    plt.figure("States - Torques - Muscles")
    for i in range(nlp.model.nbQ()):
        plt.subplot(nlp.model.nbQ(), 5, 1 + (5 * i))
        plt.plot(sol["x"][i:: step])
        plt.title("Q - " + str(i))

    for i in range(nlp.model.nbQdot()):
        plt.subplot(nlp.model.nbQdot(), 5, 2 + (5 * i))
        plt.plot(sol["x"][i + nlp.model.nbQ():: step])
        plt.title("Qdot - " + str(i))

    for i in range(nlp.model.nbGeneralizedTorque()):
        plt.subplot(nlp.model.nbGeneralizedTorque(), 5, 3 + (5 * i))
        plt.plot(sol["x"][i + nlp.nx:: step])
        plt.title("Torque - " + str(i))

    cmp = 0
    for i in range(nlp.model.nbMuscleGroups()):
        for j in range(nlp.model.muscleGroup(i).nbMuscles()):

            plt.subplot(nlp.model.muscleGroup(i).nbMuscles(), 5, 4 + i + (5 * j))
            plt.plot(sol["x"][nlp.nx + nlp.model.nbGeneralizedTorque() + cmp:: step])
            plt.title(nlp.model.muscleNames()[cmp].to_string())

            cmp += 1
    plt.show()

# if m.nbMuscleTotal() > 0:
#     plt.figure("Activations")
#     cmp = 0
#     if muscle_plot_mapping is None:
#         for i in range(m.nbMuscleGroups()):
#             for j in range(m.muscleGroup(i).nbMuscles()):
#                 plt.subplot(3, 6, cmp + 1)
#                 utils.plot_piecewise_constant(t_final, all_u[cmp, :])
#                 plt.title(m.muscleGroup(i).muscle(j).name().to_string())
#                 plt.ylim((0, 1))
#                 cmp += 1
#     else:
#         nb_row = np.max(muscle_plot_mapping, axis=0)[3] + 1
#         nb_col = np.max(muscle_plot_mapping, axis=0)[4] + 1
#         created_axes = [None] * nb_col * nb_row
#         for muscle_map in muscle_plot_mapping:
#             i_axis = muscle_map[3] * nb_col + muscle_map[4]
#             if created_axes[i_axis]:
#                 plt.sca(created_axes[i_axis])
#             else:
#                 created_axes[i_axis] = plt.subplot(nb_row, nb_col, i_axis + 1)
#             utils.plot_piecewise_constant(t_final, all_u[muscle_map[0], :])
#             # plt.title(m.muscleGroup(map[1]).muscle(map[2]).name().getString())
#             plt.title(muscle_plot_names[muscle_map[5]])
#             plt.ylim((0, 1))
#         plt.tight_layout(w_pad=-1.0, h_pad=-1.0)
#
# plt.show()
#
#     plt.plot(q, label='Position')
#     plt.plot(q_dot, label='Vitesse')
#     plt.plot(u_muscle, label='Controls Muscles')
#     plt.plot(u_torque, label='Controls Torques')
#     plt.legend()
#     plt.show()



if __name__ == "__main__":
    #nlp = prepare_nlp()

    # --- Solve the program --- #
    #sol = nlp.solve()
    multi_plots()
