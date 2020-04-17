import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.mapping import Mapping
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions


def prepare_ocp(
    show_online_optim=False, use_symmetry=False,
):
    # --- Options --- #
    # Model path
    biorbd_model = (
        biorbd.Model("jumper2contacts.bioMod"),
        biorbd.Model("jumper1contacts.bioMod"),
    )
    nb_phases = len(biorbd_model)
    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Problem parameters
    number_shooting_points = [20, 20]
    phase_time = [0.5, 0.5]

    if use_symmetry:
        q_mapping = Mapping([0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7], [0, 1, 2, 3, 4, 7, 8, 9], [5])
        q_mapping = q_mapping, q_mapping
        tau_mapping = Mapping([-1, -1, -1, 0, 1, 0, 1, 2, 3, 4, 2, 3, 4], [3, 4, 7, 8, 9], [5])
        tau_mapping = tau_mapping, tau_mapping

    else:
        q_mapping = [Mapping(range(model.nbQ()), range(model.nbQ())) for model in biorbd_model]
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = (
        ((ObjectiveFunction.minimize_torque, {"weight": 1}), (ObjectiveFunction.minimize_states, {"weight": 1}),),
        ((ObjectiveFunction.minimize_states, 1),),
    )

    # Dynamics
    problem_type = (
        ProblemType.torque_driven_with_contact,
        ProblemType.torque_driven_with_contact,
    )

    # Constraints
    constraints_first_phase = []
    constraints_second_phase = []
    if use_symmetry:
        constraints_second_phase = []
    else:
        symmetrical_constraint = (
            Constraint.Type.PROPORTIONAL_Q,
            Constraint.Instant.ALL,
            ((3, 5, -1), (4, 6, 1), (7, 10, 1), (8, 11, 1), (9, 12, 1)),
        )
        constraints_first_phase.append(symmetrical_constraint)
        constraints_second_phase.append(symmetrical_constraint)

    non_pulling_on_floor_2_contacts = (
        Constraint.Type.CONTACT_FORCE_GREATER_THAN,
        Constraint.Instant.ALL,
        ((1, 0), (2, 0), (4, 0), (5, 0),),
    )
    non_pulling_on_floor_1_contacts = (
        Constraint.Type.CONTACT_FORCE_GREATER_THAN,
        Constraint.Instant.ALL,
        ((1, 0), (3, 0)),
    )

    constraints_first_phase.append(non_pulling_on_floor_2_contacts)
    constraints_second_phase.append(non_pulling_on_floor_1_contacts)
    constraints = (constraints_first_phase, constraints_second_phase)

    # Path constraint
    if use_symmetry:
        q_reduced = q_mapping[0].nb_reduced
        pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0.8, -0.9, 0.47]
    else:
        q_reduced = q_mapping[0].nb_reduced
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
    pose_at_first_node += [0] * q_reduced  # Adds Qdot

    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]) for i in range(nb_phases)]
    X_bounds[0].first_node_min = pose_at_first_node
    X_bounds[0].first_node_max = pose_at_first_node
    # X_bounds[0].last_node_min = pose_at_first_node
    # X_bounds[0].last_node_max = pose_at_first_node
    # X_bounds[1].first_node_min = pose_at_first_node
    # X_bounds[1].first_node_max = pose_at_first_node
    X_bounds[1].last_node_min = pose_at_first_node
    X_bounds[1].last_node_max = pose_at_first_node

    # Initial guess
    X_init = [
        InitialConditions(pose_at_first_node),
        InitialConditions(pose_at_first_node),
    ]

    # Define control path constraint
    U_bounds = [
        Bounds(min_bound=[torque_min] * tau_m.nb_reduced, max_bound=[torque_max] * tau_m.nb_reduced)
        for tau_m in tau_mapping
    ]

    U_init = [InitialConditions([torque_init] * tau_m.nb_reduced) for tau_m in tau_mapping]
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
        q_mapping=q_mapping,
        q_dot_mapping=q_mapping,
        tau_mapping=tau_mapping,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=True, use_symmetry=True)

    # --- Solve the program --- #
    sol = ocp.solve()

    # x, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
    # x = ocp.nlp[0]["q_mapping"].expand(x)
    #
    # plt_ocp = PlotOcp(ocp)
    # plt_ocp.update_data(sol["x"])
    # plt_ocp.show()
    #
    # try:
    #     from BiorbdViz import BiorbdViz
    #
    #     b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
    #     b.load_movement(x.T)
    #     b.exec()
    # except ModuleNotFoundError:
    #     print("Install BiorbdViz if you want to have a live view of the optimization")
