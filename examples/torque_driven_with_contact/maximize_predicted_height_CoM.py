import biorbd
import numpy as np

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BidirectionalMapping,
    Mapping,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    OdeSolver,
    Axe,
    ConstraintList,
    ConstraintFcn,
    Node,
)


def prepare_ocp(
    model_path,
    phase_time,
    number_shooting_points,
    use_actuators=False,
    ode_solver=OdeSolver.RK4,
    objective_name="MINIMIZE_PREDICTED_COM_HEIGHT",
    com_constraints=False,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)

    if use_actuators:
        tau_min, tau_max, tau_init = -1, 1, 0
    else:
        tau_min, tau_max, tau_init = -500, 500, 0

    tau_mapping = BidirectionalMapping(Mapping([-1, -1, -1, 0]), Mapping([3]))

    # Add objective functions
    objective_functions = ObjectiveList()
    if objective_name == "MINIMIZE_PREDICTED_COM_HEIGHT":
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)
    elif objective_name == "MINIMIZE_COM_POSITION":
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, axis=Axe.Z, weight=-1)
    elif objective_name == "MINIMIZE_COM_VELOCITY":
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_VELOCITY, axis=Axe.Z, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1 / 100)

    # Dynamics
    dynamics = DynamicsList()
    if use_actuators:
        dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
    else:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    if com_constraints:
        constraints.add(
            ConstraintFcn.COM_VELOCITY,
            node=Node.ALL,
            min_bound=np.array([-100, -100, -100]),
            max_bound=np.array([100, 100, 100]),
        )
        constraints.add(
            ConstraintFcn.COM_POSITION, node=Node.ALL, min_bound=np.array([-1, -1, -1]), max_bound=np.array([1, 1, 1])
        )

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, -0.5, 0.5]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * nb_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * nb_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * tau_mapping.reduce.len, [tau_max] * tau_mapping.reduce.len)

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
        constraints=constraints,
        tau_mapping=tau_mapping,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    model_path = "2segments_4dof_2contacts.bioMod"
    t = 0.5
    ns = 20
    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=t,
        number_shooting_points=ns,
        use_actuators=False,
        objective_name="MINIMIZE_COM_VELOCITY",
        com_constraints=True,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(nb_frames=40)
