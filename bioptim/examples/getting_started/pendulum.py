"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

import platform
import matplotlib
matplotlib.use('Qt5Agg')  # Use 'Qt5Agg' for PyQt5 or 'Qt6Agg' for PyQt6
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set the backend for Matplotlib to 'Qt5Agg'
matplotlib.use('Qt5Agg') # Use 'Qt5Agg' for PyQt5 compatibility, 'Qt6Agg' if using PyQt6


from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    BiorbdModel,
    ControlType,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0  # Start and end at 0...
    x_bounds["q"][1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    # Define control path constraint
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=400, n_threads=2)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp. Please note that online graphics only works with the Linux operating system --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    sol.print_cost()

    # --- Show the results (graph or animation) --- #
    sol.graphs(show_bounds=True)
    # sol.animate(n_frames=100)

    # Retrieve decision states from the solution object.
    decision_states = sol.decision_states()

    # Retrieve stepwise states from the solution object.
    stepwise_states = sol.stepwise_states()

    # Retrieve decision controls from the solution object.
    decision_controls = sol.decision_controls()

    # Retrieve stepwise controls from the solution object.
    stepwise_controls = sol.stepwise_controls()

    # Retrieve stepwise time
    stepwise_time = sol.stepwise_time()

    # Retrieve the final time of the phase in the optimal control problem.
    final_time = ocp.phase_time

    # Retrieve the decision time
    decision_time = sol.decision_time()

    # Extract the position (q) and velocity (qdot) states from the decision states.
    q_sol_decision_states = decision_states["q"]
    qdot_sol_decision_states = decision_states["qdot"]

    # Extract the position (q) and velocity (qdot) states from the stepwise states.
    q_sol_stepwise_states = stepwise_states["q"]
    qdot_sol_stepwise_states = stepwise_states["qdot"]

    # Extract the 'tau' control from the decision controls.
    tau_sol_decision_controls = decision_controls["tau"]

    # Extract the 'tau' control  from the stepwise controls.
    tau_sol_stepwise_controls = stepwise_controls["tau"]

    # # # # # # # # # # # #
    tau_decision_dof1 = [item[0][0] for item in tau_sol_decision_controls]
    tau_decision_dof2 = [item[1][0] for item in tau_sol_decision_controls]

    tau_stepwise_dof1 = [item[0][0] for item in tau_sol_stepwise_controls]
    tau_stepwise_dof2 = [item[1][0] for item in tau_sol_stepwise_controls]

    q_decision_dof1 = [item[0][0] for item in q_sol_decision_states]
    q_decision_dof2 = [item[1][0] for item in q_sol_decision_states]

    q_stepwise_dof1 = [item[0][0] for item in q_sol_stepwise_states]
    q_stepwise_dof2 = [item[1][0] for item in q_sol_stepwise_states]

    qdot_decision_dof1 = [item[0][0] for item in qdot_sol_decision_states]
    qdot_decision_dof2 = [item[1][0] for item in qdot_sol_decision_states]

    qdot_stepwise_dof1 = [item[0][0] for item in qdot_sol_stepwise_states]
    qdot_stepwise_dof2 = [item[1][0] for item in qdot_sol_stepwise_states]

    # Convert Casadi DM to NumPy and flatten
    decision_times_np = [np.array(dm.full()).flatten() for dm in decision_time]
    stepwise_times_np = [np.array(dm.full()).flatten() for dm in stepwise_time]

    stepwise_concatenated_times = np.concatenate(stepwise_times_np[:-1])
    # Extract the first element of each sub-array and concatenate into a new NumPy array
    decision_concatenated_times = np.array([sub_array[0] for sub_array in decision_times_np])

    # Repeating the values
    repeated_tau_stepwise_dof1 = np.repeat(tau_stepwise_dof1, 6)
    repeated_tau_stepwise_dof2 = np.repeat(tau_stepwise_dof2, 6)
    repeated_q_stepwise_dof1 = np.repeat(q_stepwise_dof1, 6)
    repeated_q_stepwise_dof2 = np.repeat(q_stepwise_dof2, 6)
    repeated_qdot_stepwise_dof1 = np.repeat(qdot_stepwise_dof1, 6)
    repeated_qdot_stepwise_dof2 = np.repeat(qdot_stepwise_dof2, 6)

    tau_decision_dof2.append(tau_decision_dof2[-1])
    tau_decision_dof1.append(tau_decision_dof1[-1])

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Plotting q (position) solutions for both DOFs
    axs[0, 0].plot(decision_concatenated_times, q_decision_dof1, label="Decision")
    axs[0, 0].step(stepwise_concatenated_times, repeated_q_stepwise_dof1[:len(stepwise_concatenated_times)], label="Stepwise")
    axs[0, 0].set_title("q Solution for DOF1")
    axs[0, 0].set_ylabel("q")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(decision_concatenated_times, q_decision_dof2, label="Decision")
    axs[0, 1].step(stepwise_concatenated_times, repeated_q_stepwise_dof2[:len(stepwise_concatenated_times)], label="Stepwise")
    axs[0, 1].set_title("q Solution for DOF2")
    axs[0, 1].set_ylabel("q")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plotting qdot (velocity) solutions for both DOFs
    axs[1, 0].plot(decision_concatenated_times, qdot_decision_dof1, label="Decision")
    axs[1, 0].step(stepwise_concatenated_times, repeated_qdot_stepwise_dof1[:len(stepwise_concatenated_times)], label="Stepwise")
    axs[1, 0].set_title("qdot Solution for DOF1")
    axs[1, 0].set_ylabel("qdot")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(decision_concatenated_times, qdot_decision_dof2, label="Decision")
    axs[1, 1].step(stepwise_concatenated_times, repeated_qdot_stepwise_dof2[:len(stepwise_concatenated_times)], label="Stepwise")
    axs[1, 1].set_title("qdot Solution for DOF2")
    axs[1, 1].set_ylabel("qdot")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # Plotting tau (control) solutions for both DOFs
    axs[2, 0].plot(decision_concatenated_times, tau_decision_dof1, label="Decision")
    axs[2, 0].step(stepwise_concatenated_times, repeated_tau_stepwise_dof1, label="Stepwise", where='post')
    axs[2, 0].set_title("tau Solution for DOF1")
    axs[2, 0].set_ylabel("tau")
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(decision_concatenated_times, tau_decision_dof2, label="Decision")
    axs[2, 1].step(stepwise_concatenated_times, repeated_tau_stepwise_dof2, label="Stepwise", where='post')
    axs[2, 1].set_title("tau Solution for DOF2")
    axs[2, 1].set_ylabel("tau")
    axs[2, 1].set_xlabel("Time")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    # Adjust layout for clarity and display the plot
    plt.tight_layout()
    plt.subplots_adjust(left=0.058, bottom=0.074, right=0.958, top=0.935, wspace=0.15, hspace=0.560)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
