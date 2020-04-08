import numpy as np
import biorbd
from matplotlib import pyplot as plt

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions


def prepare_ocp(biorbd_model_path="eocar.bioMod"):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

    # Dynamics
    variable_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (0, 1)),
        (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    for i in range(1, 6):
        X_bounds.first_node_min[i] = 0
        X_bounds.last_node_min[i] = 0
        X_bounds.first_node_max[i] = 0
        X_bounds.last_node_max[i] = 0
    X_bounds.last_node_min[2] = 1.57
    X_bounds.last_node_max[2] = 1.57

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque(),
        [torque_max] * biorbd_model.nbGeneralizedTorque(),
    )
    U_init = InitialConditions([torque_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    nlp = ocp.nlp[0]
    for idx in range(nlp["model"].nbQ()):
        plt.figure()
        q = sol["x"][0 * nlp["model"].nbQ() + idx :: 3 * nlp["model"].nbQ()]
        q_dot = sol["x"][1 * nlp["model"].nbQ() + idx :: 3 * nlp["model"].nbQ()]
        u = np.array(sol["x"][2 * nlp["model"].nbQ() + idx :: 3 * nlp["model"].nbQ()])
        u = np.append(u, u[-1])
        t = np.linspace(0, nlp["tf"], nlp["ns"] + 1)
        plt.plot(t, q, label=nlp["x"][idx * 2])
        plt.plot(t, q_dot, label=nlp["x"][1 + idx * 2])
        plt.step(t, u, label=nlp["x"][idx], where="post")
        plt.title("DoF : " + nlp["model"].nameDof()[idx].to_string())

    plt.legend()
    plt.show()
