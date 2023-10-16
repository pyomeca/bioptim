"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end. While doing so, a force pushes the box upward.
The solver must minimize the force needed to lift the box while reaching the marker in time.
It is designed to show how to use external forces. An example of external forces that depends on the state (for
example a spring) can be found at 'examples/torque_driven_ocp/spring_load.py'

Please note that the point of application of the external forces are defined from the name of the segment in the bioMod.
It is expected to act on a segment in the global_reference_frame. BiorbdBioptim expect a list of list[segment_name, vector]
where the vector is a 6x1 array (Mx, My, Mz, Fx, Fy, Fz)
"""
import platform

from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str = "models/cube_with_forces.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolverBase
        The ode solver to use
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        # This must be PhaseDynamics.ONE_PER_NODE since external forces change at each node within the phase
        DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=PhaseDynamics.ONE_PER_NODE
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # External forces. external_forces is of len 1 because there is only one phase.
    # The inner array is of len 30 since there is 30. At each node, two forces are added to the segments "Seg1" and
    # "Test" respectively and is of the format [Mx, My, Mz, Fx, Fy, Fz]
    external_forces = [[["Seg1", (0, 0, 0, 0, 0, -2)], ["Test", (0, 0, 0, 0, 0, 5)]] for _ in range(n_shooting)]
    # Change the values (index 1) of the 5th node (index 4) and 1st (index 0) and 2nd (index 1) forces
    external_forces[4][0][1] = (0, 0, 0, 0, 0, -22)
    external_forces[4][1][1] = (0, 0, 0, 0, 0, 52)
    # Wrap external_forces in a list so there is only one value for the phase
    external_forces = [external_forces]

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][3, [0, -1]] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:3, [0, -1]] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    tau_min, tau_max = -100, 100
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        external_forces=external_forces,
        ode_solver=ode_solver,
    )


def main():
    """
    Solve an ocp with external forces and animates the solution
    """

    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.graphs()


if __name__ == "__main__":
    main()
