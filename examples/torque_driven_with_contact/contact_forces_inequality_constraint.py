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
)


def prepare_ocp(model_path, phase_time, number_shooting_points, direction, boundary):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)
    tau_min, tau_max, tau_ini = -500, 500, 0
    tau_mapping = BidirectionalMapping(Mapping([-1, -1, -1, 0]), Mapping([3]))

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

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
    u_bounds.add([[tau_min] * tau_mapping.reduce.len, [tau_max] * tau_mapping.reduce.len])

    u_init = InitialConditionsList()
    u_init.add([tau_ini] * tau_mapping.reduce.len)
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
    model_path = "2segments_4dof_2contacts.bioMod"
    t = 0.3
    ns = 10
    ocp = prepare_ocp(
        model_path=model_path, phase_time=t, number_shooting_points=ns, direction="GREATER_THAN", boundary=50
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
