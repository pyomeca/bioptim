"""
A very simple optimal control program playing with a soft contact sphere rolling going from one point to another.

The soft contact sphere are hard to make converge and sensitive to parameters.
One could use soft_contacts_dynamics or implicit_dynamics to ease the convergence.
"""

import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
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
    SoftContactDynamics,
    RigidBodyDynamics,
    SolutionIntegrator,
)


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

    bio_model = BiorbdModel(biorbd_model_path)

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        soft_contacts_dynamics=SoftContactDynamics.ODE,
    )

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        assume_phase_dynamics=True,
    )


def initial_states_from_single_shooting(model, ns, tf, ode_solver):
    ocp = prepare_single_shooting(model, ns, tf, ode_solver)

    # Find equilibrium
    x = InitialGuessList()
    x["q"] = [0, 0.10, 0]
    x["qdot"] = [1e-10, 1e-10, 1e-10]
    u = InitialGuessList()
    u["tau"] = [0, 0, 0]
    p = InitialGuessList()
    s = InitialGuessList()

    sol_from_initial_guess = Solution(ocp, [x, u, p, s])
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP)
    # s.animate()

    # Rolling Sphere at equilibrium
    x0 = s.states["q"][:, -1]
    x = InitialGuessList()
    x["q"] = x0
    x["qdot"] = np.array([0] * 3)
    u = InitialGuessList()
    u["tau"] = [0, 0, -10]
    p = InitialGuessList()
    s = InitialGuessList()

    sol_from_initial_guess = Solution(ocp, [x, u, p, s])
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP)
    # s.animate()

    return x


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase,
    slack: float = 1e-4,
    n_threads: int = 8,
    use_sx: bool = False,
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    bio_model = BiorbdModel(biorbd_model_path)

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
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        soft_contacts_dynamics=SoftContactDynamics.ODE,
        expand=True,
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

    init = initial_states_from_single_shooting(biorbd_model_path, 100, 1, ode_solver)
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
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    model = "../torque_driven_ocp/models/soft_contact_sphere.bioMod"
    ode_solver = OdeSolver.RK8()

    # Prepare OCP to reach the second marker
    ocp = prepare_ocp(model, 37, 0.37, ode_solver, slack=1e-4)
    # ocp.add_plot_penalty(CostType.ALL)
    # ocp.print(to_graph=True)

    # --- Solve the program --- #
    solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solv.set_linear_solver("mumps")
    solv.set_maximum_iterations(500)
    sol = ocp.solve(solv)

    sol.animate()
    sol.print_cost()
    sol.graphs()


if __name__ == "__main__":
    main()
