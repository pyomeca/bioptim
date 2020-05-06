from math import inf

import numpy as np
import biorbd

from biorbd_optim import (
    Instant,

    OptimalControlProgram,
    Constraint,
    Objective,
    ProblemType,
    Dynamics,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
)


def prepare_ocp(model_path, phase_time, number_shooting_points, show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)
    torque_min, torque_max, torque_init = -500, 500, 0

    q_mapping = BidirectionalMapping(
        Mapping(range(biorbd_model.nbQ())), Mapping(range(biorbd_model.nbQ())))
    tau_mapping = BidirectionalMapping(
        Mapping([-1, -1, -1, 0]), Mapping([3]))

    # Add objective functions
    objective_functions = (
        {"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, "weight": -1},
    )

    # Dynamics
    problem_type = ProblemType.torque_driven_with_contact

    # Constraints
    constraints = (
        {
            "type": Constraint.CONTACT_FORCE_INEQUALITY,
            "direction": "GREATER_THAN",
            "instant": Instant.ALL,
            "contact_force_idx": 1,
            "boundary": 0,
        },
        {
            "type": Constraint.CONTACT_FORCE_INEQUALITY,
            "direction": "GREATER_THAN",
            "instant": Instant.ALL,
            "contact_force_idx": 2,
            "boundary": 0,
        },
        {
            "type": Constraint.NON_SLIPPING,
            "instant": Instant.ALL,
            "normal_component_idx": (1, 2),
            "tangential_component_idx": 0,
            "static_friction_coefficient": 0.25,
        }
    )

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, -0.5, 0.5]

    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model)]
    X_bounds[0].first_node_min = pose_at_first_node + [0] * nb_qdot
    X_bounds[0].first_node_max = pose_at_first_node + [0] * nb_qdot

    # Initial guess
    X_init = [InitialConditions(pose_at_first_node + [0] * nb_qdot)]

    # Define control path constraint
    U_bounds = [
        Bounds(min_bound=[torque_min] * tau_mapping.reduce.len, max_bound=[torque_max] * tau_mapping.reduce.len)
    ]

    U_init = [InitialConditions([torque_init] * tau_mapping.reduce.len)]
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
    model_path = "2segments_4dof_2contacts.bioMod"
    ocp = prepare_ocp(model_path=model_path, phase_time=0.6, number_shooting_points=10, show_online_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    from matplotlib import pyplot as plt
    from casadi import vertcat, Function

    nlp = ocp.nlp[0]
    contact_forces = np.ndarray((nlp["model"].nbContacts(), nlp["ns"] + 1))

    nlp["model"] = biorbd.Model(model_path)
    contact_forces_func = Function(
        "contact_forces_func",
        [ocp.symbolic_states, ocp.symbolic_controls],
        [Dynamics.forces_from_forward_dynamics_with_contact(ocp.symbolic_states, ocp.symbolic_controls, nlp)],
        ["x", "u"],
        ["contact_forces"],
    ).expand()

    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    q_dot = states["q_dot"].to_matrix()
    u = controls["tau"].to_matrix()
    x = vertcat(q, q_dot)
    contact_forces[:, : nlp["ns"] + 1] = contact_forces_func(x, u)

    names_contact_forces = ocp.nlp[0]["model"].contactNames()
    for i, elt in enumerate(contact_forces):
        plt.plot(elt.T, label=f"{names_contact_forces[i].to_string()}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()

