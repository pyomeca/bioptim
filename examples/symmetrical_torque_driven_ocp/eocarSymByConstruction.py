import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.mapping import Mapping
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions
from biorbd_optim.plot import PlotOcp


def prepare_ocp(biorbd_model_path="eocarSym.bioMod", show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0
    dof_mapping = Mapping([0, 1, 2, 2], [0, 1, 2], [3])

    # Add objective functions
    objective_functions = {"type": ObjectiveFunction.minimize_torque, "weight": 100}

    # Dynamics
    variable_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        {
            "type": Constraint.Type.MARKERS_TO_PAIR,
            "instant": Constraint.Instant.START,
            "first_marker": 0,
            "second_marker": 1,
        },
        {
            "type": Constraint.Type.MARKERS_TO_PAIR,
            "instant": Constraint.Instant.END,
            "first_marker": 0,
            "second_marker": 2,
        },
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model, dof_mapping)
    for i in range(3, 6):
        X_bounds.first_node_min[i] = 0
        X_bounds.last_node_min[i] = 0
        X_bounds.first_node_max[i] = 0
        X_bounds.last_node_max[i] = 0

    # Initial guess
    X_init = InitialConditions([0] * dof_mapping.nb_reduced * 2)

    # Define control path constraint
    U_bounds = Bounds([torque_min] * dof_mapping.nb_reduced, [torque_max] * dof_mapping.nb_reduced,)
    U_init = InitialConditions([torque_init] * dof_mapping.nb_reduced)

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
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    x, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
    x = ocp.nlp[0]["dof_mapping"].expand(x)

    plt_ocp = PlotOcp(ocp)
    plt_ocp.update_data(sol["x"])
    plt_ocp.show()

    try:
        from BiorbdViz import BiorbdViz

        b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
        b.load_movement(x.T)
        b.exec()
    except ModuleNotFoundError:
        print("Install BiorbdViz if you want to have a live view of the optimization")
