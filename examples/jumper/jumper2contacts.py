import numpy as np
import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.mapping import Mapping
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions


def prepare_nlp(
    biorbd_model_path="jumper2contacts.bioMod",
    show_online_optim=False,
    use_symmetry=True,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Results path
    optimization_name = "jumper"
    results_path = "Results/"
    control_results_file_name = results_path + "Controls" + optimization_name + ".txt"
    state_results_file_name = results_path + "States" + optimization_name + ".txt"

    # Problem parameters
    number_shooting_points = [20, 20]
    phase_time = [0.5, 0.5]
    is_cyclic_constraint = False
    is_cyclic_objective = False
    if use_symmetry:
        dof_mapping = Mapping(
            [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7], [0, 1, 2, 3, 4, 7, 8, 9], [5]
        )
    else:
        dof_mapping = Mapping(range(biorbd_model.nbQ()), range(biorbd_model.nbQ()))
    dof_mapping = dof_mapping, dof_mapping

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 1),
                           (ObjectiveFunction.minimize_states, 1),), \
                          ((ObjectiveFunction.minimize_states, 1),)

    # Dynamics
    problem_type = ProblemType.torque_driven, ProblemType.torque_driven

    # Constraints
    if use_symmetry:
        constraints = (), ()
    else:
        constraints = ((Constraint.Type.PROPORTIONAL_CONTROL, Constraint.Instant.All, (3, 5, -1)),),

    # Path constraint
    if use_symmetry:
        nb_reduced = 8
        pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0.8, -0.9, 0.47]
    else:
        nb_reduced = biorbd_model[0].nbQdot()
        pose_at_first_node = [
            0,
            0,
            -0.5336,
            0,
            1.4,
            0,
            -1.4,
            0.8,
            -0.9,
            0.47,
            0.8,
            -0.9,
            0.47,
        ]
    pose_at_first_node += [0] * nb_reduced  # Add Qdot

    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(m) for m in biorbd_model]
    X_bounds[0].first_node_min = pose_at_first_node
    X_bounds[0].first_node_max = pose_at_first_node
    X_bounds[0].last_node_min = pose_at_first_node
    X_bounds[0].last_node_max = pose_at_first_node
    X_bounds[1].first_node_min = pose_at_first_node
    X_bounds[1].first_node_max = pose_at_first_node
    X_bounds[1].last_node_min = pose_at_first_node
    X_bounds[1].last_node_max = pose_at_first_node

    # Initial guess
    X_init = [InitialConditions(pose_at_first_node), InitialConditions(pose_at_first_node)]

    # Define control path constraint
    U_bounds = [Bounds(min_bound=[torque_min] * nb_reduced, max_bound=[torque_max] * nb_reduced),
                Bounds(min_bound=[torque_min] * nb_reduced, max_bound=[torque_max] * nb_reduced)]

    U_init = [InitialConditions([torque_init] * nb_reduced), InitialConditions([torque_init] * nb_reduced)]
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        phase_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        dof_mapping=dof_mapping,
        is_cyclic_constraint=is_cyclic_constraint,
        is_cyclic_objective=is_cyclic_objective,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    nlp = prepare_nlp(show_online_optim=True)

    # --- Solve the program --- #
    sol = nlp.solve()

    all_q = np.ndarray((nlp.ns + 1, nlp.model.nbQ()))
    cmp = 0
    for idx in nlp.dof_mapping.expand_idx:
        q = sol["x"][0 * nlp.nbQ + idx :: 3 * nlp.nbQ]
        all_q[:, cmp : cmp + 1] = np.array(q)
        cmp += 1

    import BiorbdViz
    b = BiorbdViz.BiorbdViz(loaded_model=nlp.model)
    b.load_movement(all_q)
    b.exec()
