"""
This example is a trivial slider that goes from 0 to 1 and back to 0. The slider is actuated by a force applied on the
slider. The slider is constrained to move only on the x axis. This example is multi-phase optimal control problem.
"""

import platform

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    CostType,
    ControlType,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str = "models/slider.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    n_shooting: tuple = (20, 20, 20),
    phase_time: tuple = (0.2, 0.3, 0.5),
    control_type: ControlType = ControlType.CONSTANT,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolverBase
        The ode solve to use
    n_shooting: tuple
        The number of shooting points for each phase
    phase_time: tuple
        The time of each phase
    control_type: ControlType
        The type of control to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    # final_time = (0.2, 0.2, 0.2)
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, ode_solver=ode_solver)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, ode_solver=ode_solver)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, ode_solver=ode_solver)

    # Constraints
    constraints = ConstraintList()

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bio_model[2].bounds_from_ranges("qdot"), phase=2)

    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["qdot"][:, 0] = 0
    x_bounds[1]["q"][:, -1] = 0.5
    x_bounds[1]["qdot"][:, -1] = 0.5
    x_bounds[2]["q"][:, -1] = 0
    x_bounds[2]["qdot"][:, -1] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[1].nb_tau, phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[2].nb_tau, phase=2)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        control_type=control_type,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    n_shooting = (20, 30, 50)
    phase_time = (0.2, 0.3, 0.5)

    ocp = prepare_ocp(n_shooting=n_shooting, phase_time=phase_time)

    ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=platform.system() == "Linux")
    sol = ocp.solve(solver)
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
