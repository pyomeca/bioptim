"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""
import matplotlib.pyplot as plt
from casadi import DM, horzcat, Function, sum1
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    PlotType,
    ObjectiveList,
    Node,
)
def plot_objectives(ocp):
    def penalty_plot_count():
        number_of_plots = 0
        objective_names = []
        same_objectives = [[], []]
        number_of_same_objectives = 0
        for nlp in ocp.nlp:
            for j in nlp.J:
                if j.name in objective_names:
                    same_objectives[0].append(objective_names.index(j.name))
                    same_objectives[1].append(number_of_plots)
                    number_of_same_objectives += 1
                else:
                    objective_names.append(j.name)
                number_of_plots += 1
        return number_of_plots, number_of_same_objectives, same_objectives

    def penalty_color(number_of_plots, number_of_same_objectives):
        step_size = 1 / (number_of_plots - number_of_same_objectives)
        color = []
        unique_color = 0
        for i in range(number_of_plots):
            if i in same_objectives[1]:
                color += [plt.cm.viridis(step_size * same_objectives[0][same_objectives[1].index(i)])]
            else:
                color += [plt.cm.viridis(step_size * unique_color)]
                unique_color += 1
        return color

    def get_plotting_penalty_values(x, u, p, j, dt):
        plot_values_returned = DM()
        for i in range(np.shape(x)[1]):
            if j.target is not None:
                try:
                    plot_values = j.weighted_function(x[:, i], u, p, DM(j.weight), DM(j.target), DM(dt))
                except AttributeError:
                    print('here')
            else:
                try:
                    plot_values = j.weighted_function(x[:, i], u, p, DM(j.weight), [], DM(dt))
                except AttributeError:
                    print('here')
            plot_returned = horzcat(plot_values_returned, plot_values[:, 0])
            plot_values_combined = sum1(plot_returned)  # Est-ce que le quadratique est déjà dans la fonction ?
        return plot_values_combined

    number_of_plots, number_of_same_objectives, same_objectives = penalty_plot_count()
    color = penalty_color(number_of_plots, number_of_same_objectives)

    number_of_plots = 0
    for i_phase, nlp in enumerate(ocp.nlp):
        for j in nlp.J:
            if "time" in nlp.parameters.names:
                if j.name == 'MINIMIZE_TIME':
                    dt = Function("time", [nlp.parameters.cx], [j.dt])(nlp.parameters[nlp.parameters.names.index('time')].dt)
            else:
                dt = j.dt
            if j.type in ObjectiveFcn.Mayer:
                ocp.add_plot(f"Objectives", lambda x, u, p, j, dt: get_plotting_penalty_values(x, u, p, j, dt), plot_type=PlotType.POINT, phase=i_phase, j=j, dt=dt, color=color[number_of_plots], node_idx=j.node_idx, label=j.name)
            else:
                ocp.add_plot(f"Objectives", lambda x, u, p, j, dt: get_plotting_penalty_values(x, u, p, j, dt), plot_type=PlotType.INTEGRATED, phase=i_phase, j=j, dt=dt, color=color[number_of_plots], label=j.name)
            number_of_plots += 1

    return


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
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
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"))
    objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q"))
    objective_functions.add(Objective(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=0, key="q"))
    objective_functions.add(Objective(ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.MID, index=1, key="q"))

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
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
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, n_shooting=100)

    # Custom plots
    plot_objectives(ocp)

    # # --- Print ocp structure --- #
    # ocp.print(to_console=False, to_graph=True)

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results in a bioviz animation --- #
    sol.animate()


if __name__ == "__main__":
    main()
