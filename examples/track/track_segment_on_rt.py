"""
This example is a trivial example where a stick must keep its coordinate system of axes aligned with the one
from a box during the whole duration of the movement. The initial and final position of the box are dictated,
the rest is fully optimized. It is designed to show how one can use the tracking RT function to track
any RT (for instance Inertial Measurement Unit [IMU]) with a body segment
"""

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
    OdeSolver,
)


def prepare_ocp(
    biorbd_model_path: str, final_time: float, n_shooting: int, ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the model
    final_time: float
        The time of the final node
    n_shooting: int
        The number of shooting points
    ode_solver:
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_SEGMENT_WITH_CUSTOM_RT, node=Node.ALL, segment="seg_rt", rt=0)

    # Path constraint
    nq = biorbd_model.nbQ()
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][2, [0, -1]] = [-1.57, 1.57]
    x_bounds[0][nq:, [0, -1]] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


def main():
    """
    Prepares, solves and animates the program
    """

    ocp = prepare_ocp(
        biorbd_model_path="cube_and_line.bioMod",
        n_shooting=30,
        final_time=1,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
