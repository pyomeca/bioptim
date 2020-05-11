import numpy as np
import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    Constraint,
    Objective,
    ProblemType,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    Dynamics,
    Data,
    ShowResult,
)


def custom_func_phase_transition(ocp, nlp, t, x, u, first_marker_idx, second_marker_idx):
    nq = nlp["q_mapping"].reduce.len
    for v in x:
        q = nlp["q_mapping"].expand.map(v[:nq])
        first_marker = nlp["model"].marker(q, first_marker_idx).to_mx()
        second_marker = nlp["model"].marker(q, second_marker_idx).to_mx()

    return nlp - ocp.nlp[nlp["phase_idx"] + 1]


def prepare_ocp(
    model_path, phase_time, number_shooting_points, show_online_optim=False, use_symmetry=True,
):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)
    torque_min, torque_max, torque_init = -1000, 1000, 0

    if use_symmetry:
        q_mapping = BidirectionalMapping(
            Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
        )
        q_mapping = q_mapping, q_mapping
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
        )
        tau_mapping = tau_mapping, tau_mapping

    else:
        q_mapping = BidirectionalMapping(
            Mapping([i for i in range(biorbd_model[0].nbQ())]), Mapping([i for i in range(biorbd_model[0].nbQ())]),
        )
        q_mapping = q_mapping, q_mapping
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = (
        (),
        (
            {"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, "weight": -1},
            {"type": Objective.Lagrange.MINIMIZE_ALL_CONTROLS, "weight": 1 / 100},
        ),
    )

    # Dynamics
    problem_type = (
        ProblemType.torque_driven_with_contact,
        ProblemType.torque_driven_with_contact,
    )

    constraints_first_phase = []
    constraints_second_phase = []

    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints_first_phase.append(
            {
                "type": Constraint.CONTACT_FORCE_INEQUALITY,
                "direction": "GREATER_THAN",
                "instant": Instant.ALL,
                "contact_force_idx": i,
                "boundary": 0,
            }
        )
    contact_axes = (1, 3)
    for i in contact_axes:
        constraints_second_phase.append(
            {
                "type": Constraint.CONTACT_FORCE_INEQUALITY,
                "direction": "GREATER_THAN",
                "instant": Instant.ALL,
                "contact_force_idx": i,
                "boundary": 0,
            }
        )
    constraints_first_phase.append(
        {
            "type": Constraint.NON_SLIPPING,
            "instant": Instant.ALL,
            "normal_component_idx": (1, 2),
            "tangential_component_idx": 0,
            "static_friction_coefficient": 0.5,
        }
    )
    constraints_second_phase.append(
        {
            "type": Constraint.NON_SLIPPING,
            "instant": Instant.ALL,
            "normal_component_idx": 1,
            "tangential_component_idx": 0,
            "static_friction_coefficient": 0.5,
        }
    )
    if not use_symmetry:
        first_dof = (3, 4, 7, 8, 9)
        second_dof = (5, 6, 10, 11, 12)
        coeff = (-1, 1, 1, 1, 1)
        for i in range(len(first_dof)):
            constraints_first_phase.append(
                {
                    "type": Constraint.PROPORTIONAL_STATE,
                    "instant": Instant.ALL,
                    "first_dof": first_dof[i],
                    "second_dof": second_dof[i],
                    "coef": coeff[i],
                }
            )

        for i in range(len(first_dof)):
            constraints_second_phase.append(
                {
                    "type": Constraint.PROPORTIONAL_STATE,
                    "instant": Instant.ALL,
                    "first_dof": first_dof[i],
                    "second_dof": second_dof[i],
                    "coef": coeff[i],
                }
            )
    constraints = (constraints_first_phase, constraints_second_phase)

    # Path constraint
    if use_symmetry:
        nb_q = q_mapping[0].reduce.len
        nb_qdot = nb_q
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
            1.4,
            0.8,
            -0.9,
            0.47,
            0.8,
            -0.9,
            0.47,
        ]

    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]) for i in range(nb_phases)]
    X_bounds[0].min[:, 0] = pose_at_first_node + [0] * nb_qdot
    X_bounds[0].max[:, 0] = pose_at_first_node + [0] * nb_qdot

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


def run_and_save_ocp(model_path):
    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=[0.4, 0.2],
        number_shooting_points=[6, 6],
        show_online_optim=False,
        use_symmetry=True,
    )
    sol = ocp.solve()
    OptimalControlProgram.save(ocp, sol, "jumper2contacts_sol")


if __name__ == "__main__":
    model_path = ("jumper2contacts.bioMod", "jumper1contacts.bioMod")
    run_and_save_ocp(model_path)
    ocp, sol = OptimalControlProgram.load(biorbd_model_path=model_path, name="jumper2contacts_sol.bo")

    from matplotlib import pyplot as plt
    from casadi import vertcat, Function, MX

    contact_forces = np.zeros((6, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    cs_map = (range(6), (0, 1, 3, 4))

    for i, nlp in enumerate(ocp.nlp):
        states, controls = Data.get_data(ocp, sol["x"], phase_idx=i)
        q = states["q"], q_dot = states["q_dot"], u = controls["tau"]
        x = np.concatenate(q, q_dot)
        if i == 0:
            contact_forces[cs_map[i], : nlp["ns"] + 1] = nlp["contact_forces_func"](x, u)
        else:
            contact_forces[cs_map[i], ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = nlp[
                "contact_forces_func"
            ](x, u)

    names_contact_forces = ocp.nlp[0]["model"].contactNames()
    for i, elt in enumerate(contact_forces):
        plt.plot(elt.T, label=f"{names_contact_forces[i].to_string()}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # try:
    #     from BiorbdViz import BiorbdViz
    #
    #     states, _ = Data.get_data_from_V(ocp, sol["x"])
    #     q = states["q"].to_matrix()
    #     q_dot = states["q_dot"].to_matrix()
    #     x = vertcat(q, q_dot)
    #     q_total = np.ndarray((ocp.nlp[0]["model"].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    #     for i in range(len(ocp.nlp)):
    #         if i == 0:
    #             q_total[:, : ocp.nlp[i]["ns"]] = ocp.nlp[i]["q_mapping"].expand.map(x[i])[:, :-1]
    #         else:
    #             q_total[:, ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + ocp.nlp[i]["ns"]] = ocp.nlp[i][
    #                 "q_mapping"
    #             ].expand.map(x[i])[:, :-1]
    #     q_total[:, -1] = ocp.nlp[-1]["q_mapping"].expand.map(x[-1])[:, -1]
    #
    #     # np.save("results2", q.T)
    #
    #     b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
    #     b.load_movement(q_total.T)
    #     b.exec()
    # except ModuleNotFoundError:
    #     print("Install BiorbdViz if you want to have a live view of the optimization")
    #     plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate(nb_frames=40)
