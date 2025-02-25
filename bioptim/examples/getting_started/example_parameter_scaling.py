"""
This example is a clone of the getting_started/pendulum.py example with the difference that the
model now evolves in an environment where the gravity can be modified.
The goal of the solver it to find the optimal gravity (target = 8 N/kg), while performing the
pendulum balancing task
It is designed to show how one can define its own parameter objective functions if the provided ones are not
sufficient.
"""

from typing import Any

import numpy as np
from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    InterpolationType,
    ParameterList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    ParameterObjectiveList,
    PenaltyController,
    ObjectiveList,
    PhaseDynamics,
    VariableScaling,
    VariableScalingList,
    SolutionMerge,
)


def generate_dat_to_track(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
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
    The ocp ready to be solved
    """
    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path)
    n_tau = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    tau_min, tau_max = -300, 300
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
    )


def my_parameter_function(bio_model: BiorbdModel, value: MX):
    bio_model.set_gravity(value)


def my_target_function(controller: PenaltyController, key: str) -> MX:
    return controller.parameters[key].cx


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    min_g: np.ndarray,
    max_g: np.ndarray,
    q_to_track: np.ndarray,
    qdot_to_track: np.ndarray,
    tau_to_track: np.ndarray,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    min_g: np.ndarray
        The minimal value for the gravity
    max_g: np.ndarray
        The maximal value for the gravity
    q_to_track: np.ndarray
        The data to track
    qdot_to_track: np.ndarray
        The data to track
    tau_to_track: np.ndarray
        The data to track
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
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
    The ocp ready to be solved
    """

    # Define the parameter to optimize
    parameters = ParameterList(use_sx=use_sx)
    parameter_objectives = ParameterObjectiveList()
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    g_scaling = VariableScaling("gravity_xyz", np.array([1, 1, 1]))  # Works fine (output: 0.0,   1.0, -20.0)
    # g_scaling = VariableScaling("gravity_xyz", np.array([1, 1, 10])) # Does not converge to the right place
    #  "Optimal parameters unscaled: {'gravity_xyz': array([ 0.        ,  5.00000002, -4.9999999 ])}"
    #  "Optimal parameters scaled: {'gravity_xyz': array([ 0.        ,  5.00000002, -0.49999999])}"
    parameters.add(
        "gravity_xyz",  # The name of the parameter
        my_parameter_function,  # The function that modifies the biorbd model
        size=3,  # The number of elements this particular parameter vector has
        scaling=g_scaling,  # The scaling of the parameter
    )

    # Give the parameter some min and max bounds and initial conditions
    parameter_bounds.add("gravity_xyz", min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)
    parameter_init["gravity_xyz"] = np.array([0.0, 0.5, -15])

    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path, parameters=parameters)
    n_tau = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        state_continuity_weight=100,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", min_bound=q_to_track, max_bound=q_to_track, interpolation=InterpolationType.EACH_FRAME)
    x_bounds.add("qdot", min_bound=qdot_to_track, max_bound=qdot_to_track, interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=tau_to_track, max_bound=tau_to_track, interpolation=InterpolationType.EACH_FRAME)

    # Define initial guesses
    x_init = InitialGuessList()
    x_init.add("q", q_to_track, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", qdot_to_track, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add("tau", tau_to_track, interpolation=InterpolationType.EACH_FRAME)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        ode_solver=ode_solver,
        use_sx=use_sx,
    )


def main():
    """
    Solve and print the optimized value for the gravity and animate the solution
    """
    final_time = 1
    n_shooting = 100

    ocp_to_track = generate_dat_to_track(
        biorbd_model_path="models/pendulum_wrong_gravity.bioMod", final_time=final_time, n_shooting=n_shooting
    )
    sol_to_track = ocp_to_track.solve(Solver.IPOPT(show_online_optim=False))
    q_to_track = sol_to_track.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot_to_track = sol_to_track.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    tau_to_track = sol_to_track.decision_controls(to_merge=SolutionMerge.NODES)["tau"]

    ocp = prepare_ocp(
        biorbd_model_path="models/pendulum.bioMod",
        final_time=final_time,
        n_shooting=n_shooting,
        min_g=np.array([0, -5, -50]),
        max_g=np.array([0, 5, -5]),
        q_to_track=q_to_track,
        qdot_to_track=qdot_to_track,
        tau_to_track=tau_to_track,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Get the results --- #
    print(f"Optimal parameters unscaled: {sol.decision_parameters(scaled=False)}")
    print(f"Optimal parameters scaled: {sol.decision_parameters(scaled=True)}")

    # --- Show results --- #
    # sol.graphs()
    sol.animate(n_frames=200, viewer="pyorerun")


if __name__ == "__main__":
    main()
