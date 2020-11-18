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
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
)


def prepare_ocp(model_path, phase_time, number_shooting_points, mu):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)
    tau_min, tau_max, tau_init = -500, 500, 0
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
        Constraint.CONTACT_FORCE,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=1,
    )
    constraints.add(
        Constraint.CONTACT_FORCE,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=2,
    )
    constraints.add(
        Constraint.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=mu,
    )

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, -0.5, 0.5]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * nb_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * nb_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * tau_mapping.reduce.len, [tau_max] * tau_mapping.reduce.len])

    u_init = InitialGuessList()
    u_init.add([tau_init] * tau_mapping.reduce.len)
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
    t = 0.6
    ns = 10
    mu = 0.2
    ocp = prepare_ocp(model_path=model_path, phase_time=t, number_shooting_points=ns, mu=mu)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
