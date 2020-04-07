import numpy as np
import biorbd
from matplotlib import pyplot as plt
import biorbd_optim
from biorbd_optim import Mapping
from biorbd_optim.dynamics import Dynamics
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.variable import Variable
from biorbd_optim.constraints import Constraint


def prepare_nlp(biorbd_model_path="jumper2contacts.bioMod",
                show_online_optim=False,
                use_symmetry=True):
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
    final_time = 0.5
    ode_solver = biorbd_optim.OdeSolver.RK
    is_cyclic_constraint = False
    is_cyclic_objective = False
    if use_symmetry:
        dof_mapping = Mapping([0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7],
                              [0, 1, 2, 3, 4, 7, 8, 9],
                              [5])
    else:
        dof_mapping = Mapping(range(13), range(13))

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 1),
                           )

    # Dynamics
    variable_type = Variable.torque_driven
    dynamics_func = Dynamics.forward_dynamics_torque_driven

    # Constraints
    if use_symmetry:
        constraints = ()
    else:
        constraints = ((Constraint.Type.PROPORTIONAL_CONTROL, Constraint.Instant.All, (3, 5, -1)),
                       )

    # Path constraint
    X_bounds = biorbd_optim.Bounds()
    X_init = biorbd_optim.InitialConditions()

    if use_symmetry:
        pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0.8, -0.9, 0.47]
    else:
        pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0, -1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47]
    # Initialize X_bounds (partially filled later)
    X_bounds.min = [-10, -10, -30, -0.1, -1.7, -0.4, -2.3, -1, -20, -20, -15, -17, -17, -10, -13, -17]
    X_bounds.max = [10, 10, 30, 0.1, 3.1, 2.6, -0.05, 0.7, 20, 20, 15, 13, 17, 8, 20, 17]
    X_bounds.first_node_min = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.first_node_max = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.last_node_min = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.last_node_max = [0] * 2 * dof_mapping.nb_reduced

    for i in range(dof_mapping.nb_reduced):
        X_bounds.first_node_min[i] = pose_at_first_node[i]
        X_bounds.first_node_max[i] = pose_at_first_node[i]
        X_bounds.last_node_min[i] = pose_at_first_node[i]
        X_bounds.last_node_max[i] = pose_at_first_node[i]

    # Initial guess
    X_init.init = pose_at_first_node + [0] * dof_mapping.nb_reduced

    # Define control path constraint
    torque_min = -1000
    torque_max = 1000
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
        show_online_optim=show_online_optim
    )


if __name__ == "__main__":
    nlp = prepare_nlp(show_online_optim=True)

    # --- Solve the program --- #
    sol = nlp.solve()

    all_q = np.ndarray((nlp.ns+1, nlp.model.nbQ()))
    cmp = 0
    for idx in nlp.dof_mapping.expand_idx:
        q = sol["x"][0 * nlp.nbQ + idx:: 3 * nlp.nbQ]
        all_q[:, cmp:cmp+1] = np.array(q)
        cmp += 1

    import BiorbdViz
    # np.save("Results", all_q)
    b = BiorbdViz.BiorbdViz(loaded_model=nlp.model)
    b.load_movement(all_q)
    b.exec()
