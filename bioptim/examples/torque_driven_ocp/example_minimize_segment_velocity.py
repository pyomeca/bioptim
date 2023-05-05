"""
This example shown hot to use the objective MINIMIZE_JCS.
The third segments must stay aligned with the vertical.
Note that there are other ways to do this, here it is used to examplify how to use the function MINIMIZE_JCS.
"""

import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Node,
)


def prepare_ocp(
    biorbd_model_path: str = "models/triple_pendulum.bioMod",
    n_shooting: int = 40,
) -> OptimalControlProgram:
    # Adding the models to the same phase
    bio_model = BiorbdModel((biorbd_model_path))

    # Problem parameters
    final_time = 1.5
    tau_min, tau_max, tau_init = -200, 200, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_ROTATION, segment=2, target=np.zeros((3,)), weight=100
    )  # here the target is the Euler angles of the third JCS in the global (sequence 'xyz')
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment=2, weight=100)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = 0
    x_bounds[0][0, -1] = 0
    x_bounds[0][1, -1] = np.pi
    x_bounds[0][2, -1] = -np.pi

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model.nb_q + bio_model.nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        assume_phase_dynamics=True,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    sol.animate()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
