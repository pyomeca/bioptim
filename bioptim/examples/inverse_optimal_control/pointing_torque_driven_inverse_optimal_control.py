"""
This is a basic example on how to use inverse optimal control to recover the weightings from an optimal reaching task.

Please note that this example is dependant on the external library Platypus which can be installed through
conda install -c conda-forge platypus-opt
"""


import importlib.util
from pathlib import Path

import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Solver,
    ConstraintList,
    ConstraintFcn,
    Node,
    CostType,
)

# # Load track_segment_on_rt
# spec = importlib.util.spec_from_file_location(
#     "data_to_track", str(Path(__file__).parent) + "/contact_forces_inequality_constraint_muscle.py"
# )
# data_to_track = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(data_to_track)


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, ode_solver=OdeSolver.RK4()):
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -50, 50, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    # Objective functions from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002183
    # hand jerk (minimize_marker_position_acceleration derivative=True)
    # angle jerk (minimize_states_acceleration, derivative=True)
    # angle acceleration (minimize_states_velocity, derivative=True)
    # Torque change (minimize_torques, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1)
    # Effort/Snap (minimize_jerks, derivative=True)
    # Geodesic/hand trajectory (minimize_marker, derivative=True, mayer)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, node=Node.ALL_SHOOTING, derivative=True, weight=1)
    # Energy (norm(qdot*tau))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, marker_index="marker_4", target=np.array([-0.0005*3, 0.0688*3, -0.9542*3]))
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.END, marker_index="marker_4", target=np.array([0, 0, 0]))
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, min_bound=0.5, max_bound=3)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    n_q = biorbd_model.nbQ()
    n_qdot = n_q

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * n_q + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * biorbd_model.nbGeneralizedTorque(),
        [tau_max] * biorbd_model.nbGeneralizedTorque(),
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        n_threads=4,
    )


def main():
    # Define the problem
    model_path = "models/multiple_pendulum.bioMod"
    phase_time = 1.5
    n_shooting = 30

    # Generate data using OCP
    ocp_to_track = prepare_ocp(biorbd_model_path=model_path, phase_time=phase_time, n_shooting=n_shooting)
    ocp_to_track.add_plot_penalty(CostType.ALL)

    sol = ocp_to_track.solve(Solver.IPOPT(show_online_optim=True))
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    x = np.concatenate((q, qdot))

    sol.animate()

    # trying to retrieve the weightings from the previous OCP with IOCP
    ocp = prepare_iocp(
        biorbd_model_path=model_path,
        phase_time=final_time,
        n_shooting=ns,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
