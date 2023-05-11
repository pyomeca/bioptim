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
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    CostType,
    ControlType,
)


def prepare_ocp(
    biorbd_model_path: str = "models/slider.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    n_shooting: tuple = (20, 20, 20),
    phase_time: tuple = (0.2, 0.3, 0.5),
    control_type: ControlType = ControlType.CONSTANT,
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

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    # final_time = (0.2, 0.2, 0.2)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=True)

    # Constraints
    constraints = ConstraintList()

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds[0].min[:, 0] = 0
    x_bounds[0].max[:, 0] = 0
    x_bounds[1].min[0, -1] = 0.5
    x_bounds[1].max[0, -1] = 0.5
    x_bounds[2].min[:, -1] = 0
    x_bounds[2].max[:, -1] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        control_type=control_type,
        assume_phase_dynamics=True,
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
