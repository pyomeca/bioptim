from matplotlib import pyplot as plt
import numpy as np
import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ConstraintList,
    Constraint,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BidirectionalMapping,
    Mapping,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    ShowResult,
    Data,
)


def prepare_ocp(model_path, phase_time, number_shooting_points, direction, boundary):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)
    tau_min, tau_max, tau_init = -500, 500, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    tau_mapping = BidirectionalMapping(Mapping([-1, -1, -1, 0]), Mapping([3]))

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction=direction,
        instant=Instant.ALL,
        contact_force_idx=1,
        boundary=boundary,
    )
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction=direction,
        instant=Instant.ALL,
        contact_force_idx=2,
        boundary=boundary,
    )

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize X_bounds
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:, 0] = pose_at_first_node + [0] * nb_qdot
    x_bounds[0].max[:, 0] = pose_at_first_node + [0] * nb_qdot

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add(pose_at_first_node + [0] * nb_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * tau_mapping.reduce.len + [activation_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * tau_mapping.reduce.len + [activation_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    u_init = InitialConditionsList()
    u_init.add([tau_init] * tau_mapping.reduce.len + [activation_init] * biorbd_model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        tau_mapping=tau_mapping,
    )


if __name__ == "__main__":
    model_path = "2segments_4dof_2contacts_1muscle.bioMod"
    t = 0.3
    ns = 10
    ocp = prepare_ocp(
        model_path=model_path, phase_time=t, number_shooting_points=ns, direction="GREATER_THAN", boundary=50
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    nlp = ocp.nlp[0]
    nlp["model"] = biorbd.Model(model_path)

    states, controls = Data.get_data(ocp, sol["x"])
    q, q_dot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]
    x = np.concatenate((q, q_dot))
    u = np.concatenate((tau, mus))
    contact_forces = np.array(nlp["contact_forces_func"](x[:, :-1], u[:, :-1], []))

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
