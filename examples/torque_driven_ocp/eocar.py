import biorbd
from matplotlib import pyplot as plt

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, InitialConditions


def prepare_nlp(biorbd_model_path="eocar.bioMod"):
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
    variable_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (0, 1)),
        (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),
    )

    # Path constraint
    X_bounds = Bounds()
    X_init = InitialConditions()

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
    X_bounds.min.extend([-velocity_max] * (biorbd_model.nbQdot()))
    X_bounds.max.extend([velocity_max] * (biorbd_model.nbQdot()))

    # Initial guess
    X_init.init = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())

    # Define control path constraint
    torque_min = -100
    torque_max = 100
    torque_init = 0
    U_bounds = Bounds()
    U_init = InitialConditions()

    U_bounds.min = [torque_min for _ in range(biorbd_model.nbGeneralizedTorque())]
    U_bounds.max = [torque_max for _ in range(biorbd_model.nbGeneralizedTorque())]
    U_init.init = [torque_init for _ in range(biorbd_model.nbGeneralizedTorque())]
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
        is_cyclic_constraint=is_cyclic_constraint,
        is_cyclic_objective=is_cyclic_objective,
    )


if __name__ == "__main__":
    nlp = prepare_nlp()

    # --- Solve the program --- #
    sol = nlp.solve()

    for idx in range(nlp.model.nbQ()):
        plt.figure()
        q = sol["x"][0 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]
        q_dot = sol["x"][1 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]
        u = sol["x"][2 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]
        plt.plot(q, label=nlp.x[idx*2])
        plt.plot(q_dot, label=nlp.x[1+idx*2])
        plt.plot(u, label=nlp.u[idx])
        plt.title("DoF : " + nlp.model.nameDof()[idx].to_string())

    plt.legend()
    plt.show()
