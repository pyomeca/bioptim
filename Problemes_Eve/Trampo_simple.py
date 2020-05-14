import numpy as np
import biorbd

from biorbd_optim import (
    # Instant,
    OptimalControlProgram,
    Constraint,
    Objective,
    ProblemType,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
)


def prepare_ocp(show_online_optim=False):
    
    # --- Options --- #
    # Model path
    biorbd_model = (biorbd.Model("/home/laboratoire/mnt/Serveur2/clientbd8/Documents/muscod/Saut_Eve/S2M_Modeles/Jumper/ModeleBatons_2D_CAS.bioMod"))
    nb_phases = 1
    torque_min, torque_max, torque_init = -50, 50, 0

    # Problem parameters
    number_shooting_points = 300
    phase_time = 1.22 # temps fixe !

    q_mapping = BidirectionalMapping(Mapping([i for i in range(biorbd_model[0].nbQ())]))
    tau_mapping = q_mapping

    # Add objective functions
    objective_functions = (
        (),
        (
            {"type": Objective.Mayer.MINIMIZE_STATE, "instant": number_shooting_points, "weight": -1},
            {"type": Objective.Lagrange.MINIMIZE_ALL_CONTROLS, "weight": 1 / 100},
        ),
    )

    # Dynamics
    problem_type = (ProblemType.torque_driven())

    constraints_first_phase = []

    # contact_axes = (1, 2, 4, 5)
    # for i in contact_axes:
    #     constraints_first_phase.append(
    #         {"type": Constraint.CONTACT_FORCE_GREATER_THAN, "instant": Instant.ALL, "idx": i, "boundary": 0,}
    #     )
    # contact_axes = (1, 3)
    # for i in contact_axes:
    #     constraints_second_phase.append(
    #         {"type": Constraint.CONTACT_FORCE_GREATER_THAN, "instant": Instant.ALL, "idx": i, "boundary": 0,}
    #     )
    
    constraints_first_phase.append(
        {
            "type": Constraint.MINIMIZE_STATE,
            "instant": number_shooting_points,
            # "normal_component_idx": (1, 2, 4, 5),
            # "tangential_component_idx": (0, 3),
            # "static_friction_coefficient": 0.5,
        }
    )


    # first_dof = (3, 4, 7, 8, 9)
    # second_dof = (5, 6, 10, 11, 12)
    # coeff = (-1, 1, 1, 1, 1)
    # for i in range(len(first_dof)):
    #     constraints_first_phase.append(
    #         {
    #             "type": Constraint.PROPORTIONAL_STATE,
    #             "instant": Instant.ALL,
    #             "first_dof": first_dof[i],
    #             "second_dof": second_dof[i],
    #             "coef": coeff[i],
    #         }
    #     )

    constraints = constraints_first_phase

    # Path constraint
    nb_q = q_mapping[0].reduce.len
    nb_qdot = nb_q
    pose_at_first_node = [
        0,
        0,
        0,
        0,
        0,
        0,
        -2.8,
        2.8,
    ]

    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]) for i in range(nb_phases)]
    X_bounds[0].first_node_min = pose_at_first_node + [0] * nb_qdot
    X_bounds[0].first_node_max = pose_at_first_node + [0] * nb_qdot

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
    ocp = prepare_ocp(show_online_optim=True, use_symmetry=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    from matplotlib import pyplot as plt

    try:
        from BiorbdViz import BiorbdViz

        x, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
        q = np.ndarray((ocp.nlp[0]["model"].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
        for i in range(len(ocp.nlp)):
            if i == 0:
                q[:, : ocp.nlp[i]["ns"]] = ocp.nlp[i]["q_mapping"].expand.map(x[i])[:, :-1]
            else:
                q[:, ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + ocp.nlp[i]["ns"]] = ocp.nlp[i][
                    "q_mapping"
                ].expand.map(x[i])[:, :-1]
        q[:, -1] = ocp.nlp[-1]["q_mapping"].expand.map(x[-1])[:, -1]

        b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
        b.load_movement(q.T)
        b.exec()
    except ModuleNotFoundError:
        print("Install BiorbdViz if you want to have a live view of the optimization")
        plt.show()





















