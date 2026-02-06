"""
This example presents how to implement a holonomic constraint in bioptim with algebraic states.
The simulation is an arm with a pendulum attached, where the pendulum attachment is enforced through holonomic
constraints. Unlike the non-algebraic version, this implementation uses algebraic states (q_v) for dependent 
coordinates, requiring explicit constraint enforcement at each node.

Methods used from HolonomicBiorbdModel:
---------------------------------------
- compute_q_from_u_iterative(q_u_array, q_v_init=None): 
    Reconstructs the full generalized coordinates q from independent coordinates q_u.
    In the algebraic version, q_v_init is provided from the algebraic states to warm-start 
    the iterative solver, improving convergence and ensuring consistency with the OCP solution.
    
Note: The algebraic formulation explicitly includes q_v as algebraic states in the OCP, which are
constrained through path constraints (constraint_holonomic). This differs from the non-algebraic
version where q_v is implicitly computed within the dynamics.
"""

import numpy as np

from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintList,
    ControlType,
    DynamicsOptions,
    DynamicsOptionsList,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    InitialGuessList,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    CostType,
)
from bioptim.examples.utils import ExampleUtils
import numpy as np

try:
    from .custom_dynamics import AlgebraicHolonomicMusclesBiorbdModel, constraint_holonomic, constraint_holonomic_end
except ImportError:
    from custom_dynamics import AlgebraicHolonomicMusclesBiorbdModel, constraint_holonomic, constraint_holonomic_end


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1,
    expand_dynamics: bool = False,
    control_type: ControlType = ControlType.CONSTANT,
) -> (AlgebraicHolonomicMusclesBiorbdModel, OptimalControlProgram):
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
    bio_model = AlgebraicHolonomicMusclesBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0, 1, 4],
        dependent_joint_index=[2, 3],
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=200, multi_thread=False)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(), expand_dynamics=expand_dynamics))

    # Boundaries
    u_variable_bimapping = BiMappingList()
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint, so we need to map the states and controls to only compute the dynamics of the independent joints
    # The dynamics of the dependent joints will be computed from the holonomic constraint
    u_variable_bimapping.add("q", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    u_variable_bimapping.add("qdot", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])

    v_variable_bimapping = BiMappingList()
    v_variable_bimapping.add("q", to_second=[None, None, 0, 1, None], to_first=[2, 3])

    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=u_variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=u_variable_bimapping)

    x_bounds["q_u"][0, 0] = 0
    x_bounds["q_u"][1, 0] = np.pi / 2
    x_bounds["q_u"][2, 0] = 0
    x_bounds["q_u"][2, -1] = -np.pi
    x_bounds["qdot_u"][:, 0] = 0  # Start without any velocity

    a_bounds = BoundsList()
    a_bounds.add("q_v", bio_model.bounds_from_ranges("q", mapping=v_variable_bimapping))

    # Initial guess
    x_init = InitialGuessList()
    x_init["q_u"] = [0.2] * 3

    # Define control path constraint

    u_bounds = BoundsList()
    u_bounds.add("muscles", min_bound=[0] * 6, max_bound=[1] * 6)

    tau_min, tau_max, tau_init = -3, 3, 0  # Residual torques
    u_bounds.add("tau", min_bound=[tau_min] * 3, max_bound=[tau_max] * 3)
    u_init = InitialGuessList()

    # Path Constraints
    constraints = ConstraintList()
    constraints.add(
        constraint_holonomic,
        node=Node.ALL_SHOOTING,
    )
    constraints.add(
        constraint_holonomic_end,
        node=Node.END,
    )

    # ------------- #

    # Only the rotations are controlled, not the translations, which are constrained by the holonomic constraint
    tau_variable_bimapping = BiMappingList()
    tau_variable_bimapping.add("tau", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    
    return (
        OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            a_bounds=a_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            variable_mappings=tau_variable_bimapping,
            constraints=constraints,
            control_type=control_type,
            n_threads=24,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """
    import pyorerun

    model_path = ExampleUtils.folder + "/models/arm26_w_pendulum.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.5,
        n_shooting=10,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)
    q = bio_model.compute_q_from_u_iterative(states["q_u"], q_v_init=algebraic_states["q_v"])

    viz = pyorerun.PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
    viz.add_animated_model(pyorerun.BiorbdModel(model_path), q=q)
    viz.rerun()

    sol.graphs()


if __name__ == "__main__":
    main()
    main()
