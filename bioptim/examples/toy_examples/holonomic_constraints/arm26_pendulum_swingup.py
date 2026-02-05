"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is two single pendulum that are forced to be coherent with a holonomic constraint. It is then a double
pendulum simulation.
"""

import numpy as np

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

from bioptim.examples.utils import ExampleUtils
from common import compute_all_q


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
        marker_1="COM_hand",
        marker_2="marker_3",
        index=slice(0, 2),
        local_frame_index=1,
    )

    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint
    bio_model = HolonomicTorqueBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0, 1, 4],
        dependent_joint_index=[2, 3],
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, multi_thread=False)

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
    variable_bimapping.add("q", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    variable_bimapping.add("qdot", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=variable_bimapping)

    x_bounds["q_u"][2, 0] = 0
    x_bounds["q_u"][2, -1] = -np.pi
    x_bounds["qdot_u"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess
    x_init = InitialGuessList()
    x_init["q_u"] = [0.2] * 3

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    # Only the rotations are controlled
    variable_bimapping.add("tau", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 3, max_bound=[tau_max] * 3)
    u_bounds["tau"][2, :] = 0
    u_init = InitialGuessList()
    # u_init.add("tau", [tau_init] * 2)

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
            n_threads=8,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """

    import pyorerun

    model_path = ExampleUtils.folder + "/models/arm26_w_pendulum.bioMod"
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    print(sol.real_time_to_optimize)

    print(sol.decision_states(to_merge=SolutionMerge.NODES)["q_u"])

    # --- Show results --- #
    q = compute_all_q(sol, bio_model)

    viz = pyorerun.PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
    viz.add_animated_model(pyorerun.BiorbdModel(model_path), q=q)

    viz.rerun("double_pendulum")

    sol.graphs()


if __name__ == "__main__":

    main()
