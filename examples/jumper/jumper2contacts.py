import biorbd
from matplotlib import pyplot as plt
import biorbd_optim
from biorbd_optim import Mapping
from biorbd_optim.dynamics import Dynamics
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.variable import Variable


def prepare_nlp(biorbd_model_path="jumper2contacts.bioMod"):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Results path
    optimization_name = "jumper"
    results_path = "Results/"
    control_results_file_name = results_path + "Controls" + optimization_name + ".txt"
    state_results_file_name = results_path + "States" + optimization_name + ".txt"

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    ode_solver = biorbd_optim.OdeSolver.RK
    is_cyclic_constraint = False
    is_cyclic_objective = False
    dof_mapping = Mapping([0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7],
                          [0, 1, 2, 3, 4, 7, 8, 9],
                          [5])

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

    # Dynamics
    variable_type = Variable.torque_driven
    dynamics_func = Dynamics.forward_dynamics_torque_driven

    # Constraints
    constraints = ()
    # constraints = (
    #     (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (0, 1)),
    #     (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),
    # )

    # Path constraint
    X_bounds = biorbd_optim.Bounds()
    X_init = biorbd_optim.InitialConditions()

    pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0.8, -0.9, 0.47]
    # Initialize X_bounds (filled later)
    X_bounds.min = [0] * dof_mapping.nb_reduced
    X_bounds.max = [0] * dof_mapping.nb_reduced
    X_bounds.first_node_min = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.first_node_max = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.last_node_min = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.last_node_max = [0] * 2 * dof_mapping.nb_reduced

    for i in range(dof_mapping.nb_reduced):
        X_bounds.min[i] = -10
        X_bounds.max[i] = 10
        X_bounds.first_node_min[i] = pose_at_first_node[i]
        X_bounds.first_node_max[i] = pose_at_first_node[i]
        X_bounds.last_node_min[i] = pose_at_first_node[i]
        X_bounds.last_node_max[i] = pose_at_first_node[i]

    # Path constraint velocity
    velocity_max = 15
    X_bounds.min.extend([-velocity_max] * dof_mapping.nb_reduced)
    X_bounds.max.extend([velocity_max] * dof_mapping.nb_reduced)

    # Initial guess
    X_init.init = pose_at_first_node + [0] * dof_mapping.nb_reduced

    # Define control path constraint
    torque_min = -2000
    torque_max = 2000
    torque_init = 0
    U_bounds = biorbd_optim.Bounds()
    U_init = biorbd_optim.InitialConditions()

    U_bounds.min = [torque_min for _ in range(dof_mapping.nb_reduced)]
    U_bounds.max = [torque_max for _ in range(dof_mapping.nb_reduced)]
    U_init.init = [torque_init for _ in range(dof_mapping.nb_reduced)]
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
        dof_mapping,
        is_cyclic_constraint=is_cyclic_constraint,
        is_cyclic_objective=is_cyclic_objective,
    )


if __name__ == "__main__":
    nlp = prepare_nlp()

    # --- Solve the program --- #
    sol = nlp.solve()

    for idx in range(nlp.model.nbQ()):
        plt.figure()
        q = sol["x"][0 * nlp.model.nbQ() + idx:: 3 * nlp.model.nbQ()]
        q_dot = sol["x"][1 * nlp.model.nbQ() + idx:: 3 * nlp.model.nbQ()]
        u = sol["x"][2 * nlp.model.nbQ() + idx:: 3 * nlp.model.nbQ()]
        plt.plot(q)
        plt.plot(q_dot)
        plt.plot(u)
    #plt.show()
