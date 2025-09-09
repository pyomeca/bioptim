"""
A very simple optimal control program playing with a soft contact sphere rolling going from one point to another.

The soft contact sphere are hard to make converge and sensitive to parameters.
One could use ContactType.SOFT_IMPLICIT to ease the convergence.
"""

import numpy as np
from bioptim import (
    TorqueBiorbdModel,
    OptimalControlProgram,
    DynamicsOptions,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    Shooting,
    Solution,
    SolutionIntegrator,
    PhaseDynamics,
    SolutionMerge,
    ContactType,
)
from bioptim.optimization.vector_layout import VectorLayout


def prepare_single_shooting(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase,
    n_threads: int = 1,
    use_sx: bool = False,
) -> OptimalControlProgram:
    """
    Prepare the ss

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    bio_model = TorqueBiorbdModel(biorbd_model_path, contact_types=[ContactType.SOFT_EXPLICIT])

    # Dynamics
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
    )

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def initial_states_from_single_shooting(model, ns, tf, ode_solver):
    ocp = prepare_single_shooting(model, ns, tf, ode_solver)
    ocp.vector_layout = VectorLayout(ocp)

    # Find equilibrium
    dt = np.array([tf / ns])
    x = InitialGuessList()
    x["q"] = [0, 0.10, 0]
    x["qdot"] = [1e-10, 1e-10, 1e-10]
    u = InitialGuessList()
    u["tau"] = [0, 0, 0]
    p = InitialGuessList()
    a = InitialGuessList()

    sol_from_initial_guess = Solution.from_initial_guess(ocp, [dt, x, u, p, a])
    sol = sol_from_initial_guess.integrate(
        shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, to_merge=SolutionMerge.NODES
    )
    # s.animate()

    # Rolling Sphere at equilibrium
    x0 = sol["q"][:, -1]
    x = InitialGuessList()
    x["q"] = x0
    x["qdot"] = np.array([0] * 3)
    # u = InitialGuessList()
    # u["tau"] = [0, 0, -10]
    # p = InitialGuessList()
    # a = InitialGuessList()

    # sol_from_initial_guess = Solution.from_initial_guess(ocp, [dt, x, u, p, a])
    # sol2 = sol_from_initial_guess.integrate(
    #   shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, to_merge=SolutionMerge.NODES
    # )
    # sol2.animate()

    return x


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase,
    slack: float = 1e-4,
    n_threads: int = 8,
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    bio_model = TorqueBiorbdModel(biorbd_model_path, contact_types=[ContactType.SOFT_EXPLICIT])

    # Problem parameters
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SOFT_CONTACT_FORCES, weight=0.0001)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.START,
        first_marker="marker_point",
        second_marker="start",
        weight=10,
        axes=2,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="marker_point",
        second_marker="end",
        weight=10,
    )

    # Dynamics
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="marker_point", second_marker="start"
    )
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="marker_point", second_marker="end")

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    init = initial_states_from_single_shooting(biorbd_model_path, ns=n_shooting, tf=final_time, ode_solver=ode_solver)
    x_bounds["q"].min[:, 0] = (init["q"].init - slack)[:, 0]
    x_bounds["q"].max[:, 0] = (init["q"].init + slack)[:, 0]

    x_bounds["qdot"].min[:, 0] = -slack
    x_bounds["qdot"].max[:, 0] = slack

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = init["q"]
    x_init["qdot"] = init["qdot"]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    biorbd_model_path = "../torque_driven_ocp/models/soft_contact_sphere.bioMod"
    ode_solver = OdeSolver.RK8()

    # Prepare OCP to reach the second marker
    ocp = prepare_ocp(biorbd_model_path, 37, 0.3, ode_solver, slack=1e-4)
    # ocp.add_plot_penalty(CostType.ALL)
    # ocp.print(to_graph=True)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    sol = ocp.solve(solver)

    sol.print_cost()
    sol.graphs()

    # --- Show results --- #
    viewer = "pyorerun"
    if viewer == "pyorerun":
        from pyorerun import BiorbdModel, PhaseRerun

        # Model
        model = BiorbdModel(biorbd_model_path)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_floor = True

        # Visualization
        time = sol.decision_time(to_merge=SolutionMerge.NODES)
        q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        viz = PhaseRerun(time)
        viz.add_animated_model(model, q)
        viz.rerun_by_frame("Optimal solution")
    else:
        sol.animate()


if __name__ == "__main__":
    main()
