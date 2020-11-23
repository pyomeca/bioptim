from matplotlib import pyplot as plt
import numpy as np
import biorbd

from bioptim import (
    Node,
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
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    Data,
)


def prepare_ocp(model_path, phase_time, number_shooting_points, min_bound):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)
    torque_min, torque_max, torque_init = -500, 500, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    tau_mapping = BidirectionalMapping(Mapping([-1, -1, -1, 0]), Mapping([3]))

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        Constraint.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=1,
    )
    constraints.add(
        Constraint.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=2,
    )

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    nb_mus = biorbd_model.nbMuscleTotal()
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].concatenate(Bounds([activation_min] * nb_mus, [activation_max] * nb_mus))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * nb_qdot + [0.5] * nb_mus

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * nb_qdot + [0.5] * nb_mus)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [torque_min] * tau_mapping.reduce.len + [activation_min] * biorbd_model.nbMuscleTotal(),
            [torque_max] * tau_mapping.reduce.len + [activation_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    u_init = InitialGuessList()
    u_init.add([torque_init] * tau_mapping.reduce.len + [activation_init] * biorbd_model.nbMuscleTotal())
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
        objective_functions=objective_functions,
        constraints=constraints,
        tau_mapping=tau_mapping,
    )


if __name__ == "__main__":
    model_path = "2segments_4dof_2contacts_1muscle.bioMod"
    t = 0.3
    ns = 10
    ocp = prepare_ocp(model_path=model_path, phase_time=t, number_shooting_points=ns, min_bound=50)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    nlp = ocp.nlp[0]
    nlp.model = biorbd.Model(model_path)

    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    x = np.concatenate((q, q_dot, activations))
    u = np.concatenate((tau, excitations))
    contact_forces = np.array(nlp.contact_forces_func(x[:, :-1], u[:, :-1]))

    names_contact_forces = ocp.nlp[0].model.contactNames()
    for i, elt in enumerate(contact_forces):
        plt.plot(np.linspace(0, t, ns + 1)[:-1], elt, ".-", label=f"{names_contact_forces[i].to_string()}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
