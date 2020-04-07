import numpy as np
import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.mapping import Mapping
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, InitialConditions


def prepare_nlp(
    biorbd_model_path="jumper2contacts.bioMod",
    show_online_optim=False,
    use_symmetry=True,
):
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
    is_cyclic_constraint = False
    is_cyclic_objective = False
    if use_symmetry:
        dof_mapping = Mapping(
            [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7], [0, 1, 2, 3, 4, 7, 8, 9], [5]
        )
    else:
        dof_mapping = Mapping(range(13), range(13))

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 1),)

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    if use_symmetry:
        constraints = ()
    else:
        constraints = (
            (Constraint.Type.PROPORTIONAL_CONTROL, Constraint.Instant.All, (3, 5, -1)),
        )

    # Path constraint
    X_bounds = Bounds()
    X_init = InitialConditions()

    if use_symmetry:
        pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0.8, -0.9, 0.47]
    else:
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

    # Gets bounds from biorbd model
    QRanges = []
    QDotRanges = []
    for i in range(biorbd_model.nbSegment()):
        QRanges.extend(
            [
                biorbd_model.segment(i).QRanges()[j]
                for j in range(len(biorbd_model.segment(i).QRanges()))
            ]
        )
        QDotRanges.extend(
            [
                biorbd_model.segment(i).QDotRanges()[j]
                for j in range(len(biorbd_model.segment(i).QDotRanges()))
            ]
        )

    # Initialize X_bounds
    X_bounds.min = [QRanges[i].min() for i in dof_mapping.reduce_idx] \
                   + [QDotRanges[i].min() for i in dof_mapping.reduce_idx]
    X_bounds.max = [QRanges[i].max() for i in dof_mapping.reduce_idx] \
                   + [QDotRanges[i].max() for i in dof_mapping.reduce_idx]

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
    U_bounds = Bounds()
    U_init = InitialConditions()

    U_bounds.min = [torque_min for _ in range(dof_mapping.nb_reduced)]
    U_bounds.max = [torque_max for _ in range(dof_mapping.nb_reduced)]
    U_init.init = [torque_init for _ in range(dof_mapping.nb_reduced)]
    # ------------- #

    return OptimalControlProgram(
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
