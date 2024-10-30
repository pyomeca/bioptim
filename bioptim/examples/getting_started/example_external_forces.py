"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end. While doing so, a force pushes the box upward.
The solver must minimize the force needed to lift the box while reaching the marker in time.
It is designed to show how to use external forces. An example of external forces that depends on the state (for
example a spring) can be found at 'examples/torque_driven_ocp/spring_load.py'
"""

import platform

import numpy as np

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
    ExternalForceSetTimeSeries,
)


def prepare_ocp(
    biorbd_model_path: str = "models/cube_with_forces.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    expand_dynamics: bool = True,
    phase_dynamics: PhaseDynamics = PhaseDynamics.ONE_PER_NODE,
    # force_type: ExternalForceType = ExternalForceType.FORCE,
    # force_reference_frame: ReferenceFrame = ReferenceFrame.GLOBAL,
    use_point_of_applications: bool = False,
    # point_of_application_reference_frame: ReferenceFrame = ReferenceFrame.GLOBAL,
    n_threads: int = 1,
    use_sx: bool = False,
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
    phase_dynamics: PhaseDynamics
        The phase dynamics to use
    use_point_of_applications: bool
        If the external forces should be applied at the point of application or at the segment's origin
    n_threads: int
        The number of threads to use
    use_sx: bool
        If the code should be compiled with SX

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Problem parameters
    n_shooting = 30
    final_time = 2

    # Linear external forces
    # Seg1_force = np.zeros((3, n_shooting + 1))
    Seg1_force = np.zeros((3, n_shooting))
    Seg1_force[2, :] = -2
    Seg1_force[2, 4] = -22

    # Test_force = np.zeros((3, n_shooting + 1))
    Test_force = np.zeros((3, n_shooting))
    Test_force[2, :] = 5
    Test_force[2, 4] = 52

    if use_point_of_applications:
        # Seg1_point_of_application = np.zeros((3, n_shooting + 1))
        Seg1_point_of_application = np.zeros((3, n_shooting))
        Seg1_point_of_application[0, :] = 0.05
        Seg1_point_of_application[1, :] = -0.05
        Seg1_point_of_application[2, :] = 0.007

        # Test_point_of_application = np.zeros((3, n_shooting + 1))
        Test_point_of_application = np.zeros((3, n_shooting))
        Test_point_of_application[0, :] = -0.009
        Test_point_of_application[1, :] = 0.01
        Test_point_of_application[2, :] = -0.01
    else:
        Seg1_point_of_application = None
        Test_point_of_application = None
    #
    # external_forces = ExternalForces()
    # external_forces.add(
    #     key="Seg1",  # Name of the segment where the external force is applied
    #     data=Seg1_force,  # 3 x (n_shooting_points+1) array
    #     force_type=force_type,  # Type of the external force (ExternalForceType.FORCE)
    #     force_reference_frame=force_reference_frame,  # Reference frame of the external force (ReferenceFrame.GLOBAL)
    #     point_of_application=Seg1_point_of_application,  # Position of the point of application
    #     point_of_application_reference_frame=point_of_application_reference_frame,  # Reference frame of the point of application (ReferenceFrame.GLOBAL)
    # )
    # external_forces.add(
    #     key="Test",  # Name of the segment where the external force is applied
    #     data=Test_force,  # 3 x (n_shooting_points+1) array
    #     force_type=force_type,  # Type of the external force (ExternalForceType.FORCE)
    #     force_reference_frame=force_reference_frame,  # Reference frame of the external force (ReferenceFrame.GLOBAL)
    #     point_of_application=Test_point_of_application,  # Position of the point of application
    #     point_of_application_reference_frame=point_of_application_reference_frame,  # Reference frame of the point of application (ReferenceFrame.GLOBAL)
    # )

    external_force_set = ExternalForceSetTimeSeries(
        nb_frames=n_shooting,
    )
    external_force_set.add_translational_force("Seg1", Seg1_force, point_of_application=Seg1_point_of_application)
    external_force_set.add_translational_force("Test", Test_force, point_of_application=Test_point_of_application)

    bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}

    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_time_series,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

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
        ode_solver=ode_solver,
        n_threads=n_threads,
        use_sx=use_sx,
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
