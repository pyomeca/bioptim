import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.mapping import Mapping
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, InitialConditions


def prepare_ocp(biorbd_model_path="eocar.bioMod", show_online_optim=False):
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
    dof_mapping = Mapping([0, 1, 2, 1, 2], [0, 1, 2], [3, 4])

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
    X_bounds.min = [ranges[i].min() for i in dof_mapping.reduce_idx]
    X_bounds.min.extend([-velocity_max] * dof_mapping.nb_reduced)
    X_bounds.max = [ranges[i].max() for i in dof_mapping.reduce_idx]
    X_bounds.max.extend([velocity_max] * dof_mapping.nb_reduced)

    X_bounds.first_node_min = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.first_node_min[0] = ranges[0].min()
    X_bounds.first_node_max = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.first_node_max[0] = ranges[0].max()

    X_bounds.last_node_min = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.last_node_min[0] = ranges[0].min()
    X_bounds.last_node_min[2] = 2
    X_bounds.last_node_max = [0] * 2 * dof_mapping.nb_reduced
    X_bounds.last_node_max[0] = ranges[0].max()
    X_bounds.last_node_max[2] = 2

    # Initial guess
    X_init.init = [0] * 2 * dof_mapping.nb_reduced

    # Define control path constraint
    torque_min = -100
    torque_max = 100
    torque_init = 0
    U_bounds = Bounds()
    U_init = InitialConditions()

    U_bounds.min = [torque_min for _ in range(dof_mapping.nb_reduced)]
    U_bounds.max = [torque_max for _ in range(dof_mapping.nb_reduced)]
    U_init.init = [torque_init for _ in range(dof_mapping.nb_reduced)]
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
        dof_mapping=dof_mapping,
        is_cyclic_constraint=is_cyclic_constraint,
        is_cyclic_objective=is_cyclic_objective,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve()
