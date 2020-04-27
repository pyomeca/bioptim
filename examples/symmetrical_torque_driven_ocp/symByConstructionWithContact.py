import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path="symByConstructionWithContact.bioMod", show_online_optim=False, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1}

    # Dynamics
    variable_type = ProblemType.torque_driven_with_contact

    # Constraints
    constraints = (
        {"type": Constraint.CONTACT_FORCE_GREATER_THAN, "instant": Instant.ALL, "idx": 1, "boundary": 0,},
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker": 0, "second_marker": 2, },
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker": 0, "second_marker": 3, },
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    for i in range(3, 6):
        X_bounds.first_node_min[i] = 0
        X_bounds.last_node_min[i] = 0
        X_bounds.first_node_max[i] = 0
        X_bounds.last_node_max[i] = 0

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = Bounds([torque_min] * biorbd_model.nbQ(), [torque_max] * biorbd_model.nbQ(),)
    U_init = InitialConditions([torque_init] * biorbd_model.nbQ())

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
        ode_solver=ode_solver,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()

    import numpy as np
    import matplotlib.pyplot as plt
    from casadi import Function, vertcat

    contact_forces = np.zeros((2, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    cs_map = (range(2), range(2))

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
            contact_forces[cs_map[i], ocp.nlp[i - 1]["ns"]: ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = CS_func(x, u)
    plt.plot(contact_forces.T)
    plt.show()