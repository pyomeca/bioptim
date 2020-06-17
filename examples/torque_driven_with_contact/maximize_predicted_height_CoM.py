import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    Objective,
    ProblemType,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)


def prepare_ocp(model_path, phase_time, number_shooting_points, use_actuators=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)

    if use_actuators:
        torque_min, torque_max, torque_init = -1, 1, 0
    else:
        torque_min, torque_max, torque_init = -500, 500, 0

    tau_mapping = BidirectionalMapping(Mapping([-1, -1, -1, 0]), Mapping([3]))

    # Add objective functions
    objective_functions = (
        {"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, "weight": -1},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1 / 100},
    )

    # Dynamics
    if use_actuators:
        problem_type = {"type": ProblemType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT}
    else:
        problem_type = {"type": ProblemType.TORQUE_DRIVEN_WITH_CONTACT}

    # Constraints
    constraints = ()

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, -0.5, 0.5]

    # Initialize X_bounds
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, 0] = pose_at_first_node + [0] * nb_qdot
    X_bounds.max[:, 0] = pose_at_first_node + [0] * nb_qdot

    # Initial guess
    X_init = InitialConditions(pose_at_first_node + [0] * nb_qdot)

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * tau_mapping.reduce.len, max_bound=[torque_max] * tau_mapping.reduce.len)

    U_init = InitialConditions([torque_init] * tau_mapping.reduce.len)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        phase_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        tau_mapping=tau_mapping,
    )


if __name__ == "__main__":
    model_path = "2segments_4dof_2contacts.bioMod"
    t = 0.5
    ns = 20
    ocp = prepare_ocp(model_path=model_path, phase_time=t, number_shooting_points=ns, use_actuators=False)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(nb_frames=40)
