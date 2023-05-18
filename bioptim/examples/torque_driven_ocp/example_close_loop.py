import bioviz
import numpy as np
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
)

"""
This example is a example who reproduces the behavior of the contact forces constrained with 3 bars.
"""

def prepare_ocp(biorbd_model_path: str, phase_time, n_shooting, ode_solver: OdeSolver = OdeSolver.IRK()) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolver
        The type of ode solver used

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
    )

    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    # dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="m1",
        second_marker="m2",
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.END,
        marker_index="m1",
        reference_jcs=1,
        phase=0,
    )

    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    pose_at_first_node = [-1.1454, -1.3999, 0.0]
    pose_together = [-8.642583760080003e-17, 1.5707963267948966, -3.5698704918073804e-17]

    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot

    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))
    x_bounds[1][:, 0] = pose_together + [0] * n_qdot

    x_bounds.add(bounds=bio_model[2].bounds_from_ranges(["q", "qdot"]))
    x_bounds[2][:, 0] = pose_at_first_node + [0] * n_qdot


    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)
    x_init.add(pose_together + [0] * n_qdot)
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[1].nb_tau, [tau_max] * bio_model[1].nb_tau)
    u_bounds.add([tau_min] * bio_model[2].nb_tau, [tau_max] * bio_model[2].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[1].nb_tau)
    u_init.add([tau_init] * bio_model[2].nb_tau)

    # ------------- #

    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        n_threads=3,
    )


def main():
    """
    Solve and animate the solution
    """

    ocp = prepare_ocp(
        biorbd_model_path=(
                      "models/three_bar_closed_close_loop.bioMod",
                      "models/three_bar_closed_close_loop.bioMod",
                      "models/three_bar_closed_close_loop.bioMod",
                      ),
        phase_time=(0.5, 1, 0.5),
        n_shooting=(50, 100, 50),
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))


    # --- Show results --- #
    q = []
    for i in range(len(sol.states)):
        q.append(sol.states[i]["q"])
    Q = np.concatenate((q[0], q[1], q[2]), axis=1)
    biorbd_viz1 = bioviz.Viz(model_path="models/three_bar_closed_close_loop.bioMod", show_contacts=False)
    biorbd_viz1.load_movement(Q)
    biorbd_viz1.exec()


if __name__ == "__main__":
    main()



