import biorbd
import numpy as np
from matplotlib import pyplot as plt

import biorbd_optim
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.problem_type import ProblemType


def prepare_nlp(biorbd_model_path="arm26.bioMod", show_online_optim=False):
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
    velocity_max = 15
    is_cyclic_constraint = False
    is_cyclic_objective = False

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

    # Dynamics
    problem_type = ProblemType.muscles_and_torque_driven

    # Constraints
    constraints = (
        # (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),
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
        problem_type,
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
        show_online_optim=show_online_optim,
    )


def multi_plots(nlp):

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
        plt.plot(sol["x"][i::step])
        plt.title("Q - " + str(i))

    for i in range(nlp.model.nbQdot()):
        plt.subplot(nlp.model.nbQdot(), 5, 2 + (5 * i))
        plt.plot(sol["x"][i + nlp.model.nbQ() :: step])
        plt.title("Qdot - " + str(i))

    for i in range(nlp.model.nbGeneralizedTorque()):
        plt.subplot(nlp.model.nbGeneralizedTorque(), 5, 3 + (5 * i))
        plt.plot(sol["x"][i + nlp.nx :: step])
        plt.title("Torque - " + str(i))

    cmp = 0
    for i in range(nlp.model.nbMuscleGroups()):
        for j in range(nlp.model.muscleGroup(i).nbMuscles()):

            plt.subplot(nlp.model.muscleGroup(i).nbMuscles(), 5, 4 + i + (5 * j))
            plt.plot(sol["x"][nlp.nx + nlp.model.nbGeneralizedTorque() + cmp :: step])
            plt.title(nlp.model.muscleNames()[cmp].to_string())

            cmp += 1
    plt.show()


if __name__ == "__main__":
    nlp = prepare_nlp(show_online_optim=True)

    # --- Solve the program --- #
    sol = nlp.solve()
    # multi_plots(nlp)
