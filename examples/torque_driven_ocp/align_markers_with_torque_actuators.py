import biorbd

from bioptim import (
    Node,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, actuator_type=None, ode_solver=OdeSolver.RK4):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    if actuator_type and actuator_type == 1:
        tau_min, tau_max, tau_init = -1, 1, 0
    else:
        tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    if actuator_type:
        if actuator_type == 1:
            dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN)
        elif actuator_type == 2:
            dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
        else:
            raise ValueError("actuator_type is 1 (torque activations) or 2 (torque max constraints)")
    else:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.ALIGN_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(ConstraintFcn.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)
    if actuator_type == 2:
        constraints.add(ConstraintFcn.TORQUE_MAX_FROM_ACTUATORS, node=Node.ALL, min_torque=7.5)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][3:6, [0, -1]] = 0
    x_bounds[0][2, [0, -1]] = [0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    ocp = prepare_ocp("cube.bioMod", number_shooting_points=30, final_time=2, actuator_type=2)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
