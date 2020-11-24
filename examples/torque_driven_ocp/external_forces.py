import numpy as np

import biorbd

from bioptim import (
    Node,
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    ObjectiveList,
    Objective,
    ConstraintList,
    Constraint,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path="cube_with_forces.bioMod", ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)

    # External forces
    external_forces = [
        np.repeat(
            np.array([[0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 5]]).T[:, :, np.newaxis], number_shooting_points, axis=2
        )
    ]

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0][3:6, [0, -1]] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque()])

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
        objective_functions=objective_functions,
        constraints=constraints,
        external_forces=external_forces,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
