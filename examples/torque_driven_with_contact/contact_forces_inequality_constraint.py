from matplotlib import pyplot as plt
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
    ShowResult,
    Data,
)


def prepare_ocp(model_path, phase_time, number_shooting_points, direction, boundary, show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)
    torque_min, torque_max, torque_init = -500, 500, 0
    tau_mapping = BidirectionalMapping(Mapping([-1, -1, -1, 0]), Mapping([3]))

    # Add objective functions
    objective_functions = ({"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, "weight": -1},)

    # Dynamics
    problem_type = ProblemType.torque_driven_with_contact

    # Constraints
    constraints = (
        {
            "type": Constraint.CONTACT_FORCE_INEQUALITY,
            "direction": direction,
            "instant": Instant.ALL,
            "contact_force_idx": 1,
            "boundary": boundary,
        },
        {
            "type": Constraint.CONTACT_FORCE_INEQUALITY,
            "direction": direction,
            "instant": Instant.ALL,
            "contact_force_idx": 2,
            "boundary": boundary,
        },
    )

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, -0.75, 0.75]

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
        tau_mapping=tau_mapping,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    model_path = "2segments_4dof_2contacts.bioMod"
    t = 0.3
    ns = 10
    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=t,
        number_shooting_points=ns,
        direction="GREATER_THAN",
        boundary=50,
        show_online_optim=False,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    nlp = ocp.nlp[0]
    nlp["model"] = biorbd.Model(model_path)

    states, controls = Data.get_data(ocp, sol["x"])
    q, q_dot, u = states["q"], states["q_dot"], controls["tau"]
    x = np.concatenate((q, q_dot))
    contact_forces = np.array(nlp["contact_forces_func"](x[:, :-1], u[:, :-1]))

    names_contact_forces = ocp.nlp[0]["model"].contactNames()
    for i, elt in enumerate(contact_forces):
        plt.plot(np.linspace(0, t, ns + 1)[:-1], elt, ".-", label=f"{names_contact_forces[i].to_string()}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
