import biorbd
import numpy as np
from matplotlib import pyplot as plt

import biorbd_optim
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.plot import PlotOcp


def prepare_nlp(biorbd_model_path="arm26.bioMod", show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    velocity_max = 5
    is_cyclic_constraint = False
    is_cyclic_objective = False

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

    # Dynamics
    problem_type = ProblemType.muscles_and_torque_driven

    # Constraints
    constraints = (
        # (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 5)),
    )

    # Path constraint
    X_bounds = biorbd_optim.Bounds()
    X_init = biorbd_optim.InitialConditions()

    # Gets bounds from biorbd model
    ranges = []
    for i in range(biorbd_model.nbSegment()):
        ranges.extend(
            [
                biorbd_model.segment(i).QRanges()[j]
                for j in range(len(biorbd_model.segment(i).QRanges()))
            ]
        )
    X_bounds.min = [ranges[i].min() for i in range(biorbd_model.nbQ())]
    X_bounds.max = [ranges[i].max() for i in range(biorbd_model.nbQ())]

    X_bounds.first_node_min = [-1] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.first_node_min[0] = 0.07
    X_bounds.first_node_min[1] = 1.4
    X_bounds.first_node_max = [3] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.first_node_max[0] = 0.07
    X_bounds.first_node_max[1] = 1.4

    X_bounds.last_node_min = [-1] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.last_node_min[0] = 1.64
    X_bounds.last_node_min[1] = 2.04
    X_bounds.last_node_max = [3] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
    X_bounds.last_node_max[0] = 1.64
    X_bounds.last_node_max[1] = 2.04

    # Path constraint velocity
    velocity_max = 15
    X_bounds.min.extend([-velocity_max] * (biorbd_model.nbQdot()))
    X_bounds.max.extend([velocity_max] * (biorbd_model.nbQdot()))

    # Initial guess
    X_init.init = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())

    # Define control path constraint
    U_bounds = biorbd_optim.Bounds()
    U_init = biorbd_optim.InitialConditions()

    torque_min = -100
    torque_max = 100
    torque_init = 0

    U_bounds.min = [torque_min for _ in range(biorbd_model.nbGeneralizedTorque())]
    U_bounds.max = [torque_max for _ in range(biorbd_model.nbGeneralizedTorque())]
    U_init.init = [torque_init for _ in range(biorbd_model.nbGeneralizedTorque())]

    muscle_min = 0
    muscle_max = 1
    muscle_init = 0.5

    U_bounds.min.extend([muscle_min for _ in range(biorbd_model.nbMuscleTotal())])
    U_bounds.max.extend([muscle_max for _ in range(biorbd_model.nbMuscleTotal())])
    U_init.init.extend([muscle_init for _ in range(biorbd_model.nbMuscleTotal())])
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




if __name__ == "__main__":
    ocp = prepare_nlp(show_online_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    x, _, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
    x = ocp.nlp[0]["dof_mapping"].expand(x)

    np.save("static_arm", x.T)

    # plt_ocp = PlotOcp(ocp)
    # plt_ocp.update_data(sol["x"])
    # plt_ocp.show()

    # try:
    #     from BiorbdViz import BiorbdViz
    #
    #     b = BiorbdViz(loaded_model=ocp.nlp[0]["model"], show_meshes=False)
    #     b.load_movement(x.T)
    #     b.exec()
    # except ModuleNotFoundError:
    #     print("Install BiorbdViz if you want to have a live view of the optimization")

