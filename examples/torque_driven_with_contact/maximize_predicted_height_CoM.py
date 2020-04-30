from math import inf

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
)


def prepare_ocp(show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model("example_maximize_predicted_height_CoM.bioMod")
    torque_min, torque_max, torque_init = -500, 500, 0

    # Problem parameters
    number_shooting_points = 30
    phase_time = 0.5

    q_mapping = BidirectionalMapping(
        Mapping(range(biorbd_model.nbQ())), Mapping(range(biorbd_model.nbQ())))
    tau_mapping = BidirectionalMapping(
        Mapping([-1, -1, -1, 0, 1, 2]), Mapping([3, 4, 5]))

    # Add objective functions
    objective_functions = (
        {"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, "weight": -1},
    )

    # Dynamics
    problem_type = ProblemType.torque_driven_with_contact

    # Constraints
    constraints = ()

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, 0.6, -1, -1.3, 0.6]

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
    ocp = prepare_ocp(show_online_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve()

    from matplotlib import pyplot as plt
    from casadi import vertcat, Function

    nlp = ocp.nlp[0]  # why [0] is necessary if there is no multiphase?
    contact_forces = np.zeros((2, nlp["ns"] + 1))
    cs_map = range(2)

    CS_func = Function(
        "Contact_force_inequality",
        [ocp.symbolic_states, ocp.symbolic_controls],
        [nlp["model"].getConstraints().getForce().to_mx()],
        ["x", "u"],
        ["CS"],
    ).expand()
    #
    # q, q_dot, u = ProblemType.get_data_from_V(ocp, sol["x"])
    # x = vertcat(q, q_dot)
    # contact_forces[cs_map, : nlp["ns"] + 1] = CS_func(x, u)
    #
    # plt.plot(contact_forces.T)
    # plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()