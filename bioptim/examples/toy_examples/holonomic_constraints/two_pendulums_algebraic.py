"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is two single pendulum that are forced to be coherent with a holonomic constraint. It is then a double
pendulum simulation. But this time, the dynamics are computed with the algebraic states, namely q_v the dependent joints
"""

import platform
import numpy as np
from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintList,
    CostType,
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
)
from bioptim.examples.utils import ExampleUtils

from .custom_dynamics import ModifiedHolonomicTorqueBiorbdModel, constraint_holonomic, constraint_holonomic_end


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1,
    expand_dynamics: bool = False,
    ode_solver: OdeSolver = OdeSolver.COLLOCATION(polynomial_degree=2),
) -> (ModifiedHolonomicTorqueBiorbdModel, OptimalControlProgram):
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
        index=slice(1, 3),
        local_frame_index=0,
    )
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint
    bio_model = ModifiedHolonomicTorqueBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0, 3],
        dependent_joint_index=[1, 2],
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(
        DynamicsOptions(
            ode_solver=ode_solver,
            expand_dynamics=expand_dynamics,
        )
    )

    # Boundaries
    u_variable_bimapping = BiMappingList()
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint, so we need to map the states and controls to only compute the dynamics of the independent joints
    # The dynamics of the dependent joints will be computed from the holonomic constraint
    u_variable_bimapping.add("q", to_second=[0, None, None, 1], to_first=[0, 3])
    u_variable_bimapping.add("qdot", to_second=[0, None, None, 1], to_first=[0, 3])

    v_variable_bimapping = BiMappingList()
    v_variable_bimapping.add("q", to_second=[None, 0, 1, None], to_first=[1, 2])

    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=u_variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=u_variable_bimapping)

    # q_v is the state of the dependent joints
    a_bounds = BoundsList()
    a_bounds.add("q_v", bio_model.bounds_from_ranges("q", mapping=v_variable_bimapping))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q_u", [1.54, 1.54])
    x_init.add("qdot_u", [0, 0])
    x_bounds["q_u"][:, 0] = [1.54, 1.54]
    x_bounds["qdot_u"][:, 0] = [0, 0]
    x_bounds["q_u"][0, -1] = -1.54
    x_bounds["q_u"][1, -1] = 0

    a_init = InitialGuessList()
    a_init.add("q_v", [0, 2])

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    # Only the rotations are controlled
    tau_variable_bimapping = BiMappingList()
    tau_variable_bimapping.add("tau", to_second=[0, None, None, 1], to_first=[0, 3])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2)
    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * 2)

    # Path Constraints
    constraints = ConstraintList()
    constraints.add(
        constraint_holonomic,
        phase=0,
        node=Node.ALL_SHOOTING,
        # penalty_type=PenaltyType.INTERNAL,
    )
    constraints.add(
        constraint_holonomic_end,
        phase=0,
        node=Node.END,
        # penalty_type=PenaltyType.INTERNAL,
    )

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
            a_init=a_init,
            objective_functions=objective_functions,
            variable_mappings=tau_variable_bimapping,
            constraints=constraints,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = ExampleUtils.folder + "/models/two_pendulums.bioMod"
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path)
    ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    print(sol.real_time_to_optimize)

    stepwise_q_u = sol.stepwise_states(to_merge=SolutionMerge.NODES)["q_u"]
    stepwise_q_v = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)["q_v"]
    q = ocp.nlp[0].model.state_from_partition(stepwise_q_u, stepwise_q_v).toarray()

    viewer = "pyorerun"
    if viewer == "bioviz":
        import bioviz

        viz = bioviz.Viz(model_path)
        viz.load_movement(q)
        viz.exec()

    if viewer == "pyorerun":
        from pyorerun import BiorbdModel as PyorerunBiorbdModel
        from pyorerun import PhaseRerun

        pyomodel = PyorerunBiorbdModel(model_path)
        viz = PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
        viz.add_animated_model(pyomodel, q=q)

        viz.rerun("double_pendulum")

    # --- Show results --- #
    sol.graphs()


if __name__ == "__main__":
    main()
