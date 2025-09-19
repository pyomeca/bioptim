"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is two single pendulum that are forced to be coherent with a holonomic constraint. It is then a double
pendulum simulation.
"""

import platform
import numpy as np
from casadi import DM

from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintList,
    DynamicsOptions,
    DynamicsOptionsList,
    HolonomicTorqueBiorbdModel,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    Solver,
    SolutionMerge,
    OdeSolver,
)
from two_pendulums_2constraint_4DOF import compute_all_states


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1,
    expand_dynamics: bool = False,
) -> (HolonomicTorqueBiorbdModel, OptimalControlProgram):
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """
    # Create a holonomic constraint to create a double pendulum from two single pendulums
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        marker_1="marker_1",
        marker_2="marker_3",
        index=slice(0, 3),
        local_frame_index=0,
    )
    holonomic_constraints.add(
        "holonomic_constraints2",
        HolonomicConstraintsFcn.superimpose_markers,
        marker_1="DownOffset",
        marker_2="MidTige",
        index=slice(0, 1),
        local_frame_index=0,
    )
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint
    bio_model = HolonomicTorqueBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0],
        dependent_joint_index=[1, 2, 3, 4],
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), expand_dynamics=expand_dynamics))

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    variable_bimapping = BiMappingList()
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint, so we need to map the states and controls to only compute the dynamics of the independent joints
    # The dynamics of the dependent joints will be computed from the holonomic constraint
    variable_bimapping.add("q", to_second=[0, None, None, None, None], to_first=[0])
    variable_bimapping.add("qdot", to_second=[0, None, None, None, None], to_first=[0])

    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=variable_bimapping)
    x_bounds["q_u"][0, 0] = -0.5
    x_bounds["q_u"][0, -1] = 0.5
    x_bounds["qdot_u"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q_u", [-0.5])

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    # Only the rotations are controlled
    variable_bimapping.add("tau", to_second=[0, None, None, None, None], to_first=[0])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min], max_bound=[tau_max])
    u_init = InitialGuessList()
    # u_init.add("tau", [tau_init])

    # ------------- #

    return (
        OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            variable_mappings=variable_bimapping,
            constraints=constraints,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "models/two_pendulums_2.bioMod"
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))
    print(sol.real_time_to_optimize)

    # --- Show results --- #
    q = compute_all_states(sol, bio_model)

    viewer = "pyorerun"
    if viewer == "bioviz":
        import bioviz

        viz = bioviz.Viz(model_path)
        viz.load_movement(q)
        viz.exec()

    if viewer == "pyorerun":
        import pyorerun

        viz = pyorerun.PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
        viz.add_animated_model(pyorerun.BiorbdModel(model_path), q=q)

        viz.rerun("double_pendulum")

    sol.graphs()


if __name__ == "__main__":
    main()
