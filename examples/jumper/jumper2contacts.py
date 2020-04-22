import numpy as np
import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.mapping import BidirectionalMapping, Mapping
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions


def prepare_ocp(
    show_online_optim=False, use_symmetry=True,
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
    number_shooting_points = [8, 8]  # 8, 8 for dev test, echec avec 20,20 et 0.5,0.3
    phase_time = [0.4, 0.2]  # 0.4, 0.2 for dev test

    if use_symmetry:
        q_mapping = BidirectionalMapping(
            Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]),
            Mapping([0, 1, 2, 3, 7, 8, 9]))
        q_mapping = q_mapping, q_mapping
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]),
            Mapping([4, 7, 8, 9]))
        tau_mapping = tau_mapping, tau_mapping

    else:
        q_mapping = BidirectionalMapping(
            Mapping([Mapping(range(model.nbQ()), range(model.nbQ())) for model in biorbd_model]),
            Mapping([Mapping(range(model.nbQ()), range(model.nbQ())) for model in biorbd_model]))
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = (
        (),
        (
            (ObjectiveFunction.maximize_predicted_height_jump, {"weight": 1}),
            (ObjectiveFunction.minimize_all_controls, {"weight": 1 / 100}),
        ),
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
        nb_q = q_mapping[0].reduce.len
        nb_qdot = nb_q
        # pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0.8, -0.9, 0.47]
        pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]
    else:
        nb_q = q_mapping[0].reduce.len
        nb_qdot = nb_q
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

    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]) for i in range(nb_phases)]
    X_bounds[0].first_node_min = pose_at_first_node + [0] * nb_qdot
    X_bounds[0].first_node_max = pose_at_first_node + [0] * nb_qdot
    # X_bounds[0].last_node_min = pose_at_first_node
    # X_bounds[0].last_node_max = pose_at_first_node
    # X_bounds[1].first_node_min = pose_at_first_node
    # X_bounds[1].first_node_max = pose_at_first_node
    # X_bounds[1].last_node_min = pose_at_first_node + X_bounds[1].min[nb_q:]
    # X_bounds[1].last_node_max = pose_at_first_node + X_bounds[1].max[nb_q:]

    # Initial guess
    X_init = [
        InitialConditions(pose_at_first_node + [0] * nb_qdot),
        InitialConditions(pose_at_first_node + [0] * nb_qdot),
    ]

    # Define control path constraint
    U_bounds = [
        Bounds(min_bound=[torque_min] * tau_m.reduce.len, max_bound=[torque_max] * tau_m.reduce.len)
        for tau_m in tau_mapping
    ]

    U_init = [InitialConditions([torque_init] * tau_m.reduce.len) for tau_m in tau_mapping]
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

    from matplotlib import pyplot as plt
    from casadi import vertcat, Function

    contact_forces = np.zeros((6, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    cs_map = (range(6), (0, 1, 3, 4))

    for i, nlp in enumerate(ocp.nlp):
        CS_func = Function(
            "Contact_force_inequality",
            [ocp.symbolic_states, ocp.symbolic_controls],
            [nlp["model"].getConstraints().getForce().to_mx()],
            ["x", "u"],
            ["CS"],
        ).expand()

        q, q_dot, u = ProblemType.get_data_from_V(ocp, sol["x"], i)
        x = vertcat(q, q_dot)
        if i == 0:
            contact_forces[cs_map[i], : nlp["ns"] + 1] = CS_func(x, u)
        else:
            contact_forces[cs_map[i], ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = CS_func(x, u)
    plt.plot(contact_forces.T)
    plt.show()

    try:
        from BiorbdViz import BiorbdViz

        x, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
        q = np.ndarray((ocp.nlp[0]["model"].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
        for i in range(len(ocp.nlp)):
            if i == 0:
                q[:, : ocp.nlp[i]["ns"]] = ocp.nlp[i]["q_mapping"].expand(x[i])[:, :-1]
            else:
                q[:, ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + ocp.nlp[i]["ns"]] = ocp.nlp[i]["q_mapping"].expand(
                    x[i]
                )[:, :-1]
        q[:, -1] = ocp.nlp[-1]["q_mapping"].expand(x[-1])[:, -1]

        np.save("results2", q.T)

        b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
        b.load_movement(q.T)
        b.exec()
    except ModuleNotFoundError:
        print("Install BiorbdViz if you want to have a live view of the optimization")
        plt.show()

    # np.save("results2CF", q.T)
    # S = np.load("/home/iornaith/Documents/GitKraken/biorbdOptim/BiorbdOptim/examples/jumper/results2.npy")
    # print("Results loaded")
