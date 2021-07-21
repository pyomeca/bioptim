"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""
import casadi as cas
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
    PlotType,
)


def plot_objectives(ocp, x, u, p):

    for nlp in ocp.nlp:
        for j in nlp.J:
            if "time" in nlp.parameters.names:
                dt = cas.Function("time", [nlp.parameters.cx], [j.dt])(j.parameters["time"])
            else:
                dt = j.dt

            # if j.plot_target:
            #     Plot = j.weighted_function - j.target_to_plot
            # else:
            if np.shape(x)[1] == 1:
                Plot = j.weighted_function(x, u, p, cas.DM(j.weight), cas.DM(j.weight), cas.DM(dt))
                plot_returned = Plot[:, 0]
            else:
                plot_returned = cas.DM()
                for i in range(np.shape(x)[1]):
                    Plot = j.weighted_function(x[:, i], u, p, cas.DM(j.weight), cas.DM(j.weight), cas.DM(dt))
                    plot_returned = cas.horzcat(plot_returned, Plot[:, 0])
    return plot_returned


def prepare_ocp(biorbd_model_path: str, final_time: float, n_shooting: int) -> OptimalControlProgram:
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

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

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
        use_sx=True,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, n_shooting=100)

    # Custom plots
    ocp.add_plot("Objective", lambda x, u, p : plot_objectives(ocp, x, u, p), plot_type=PlotType.INTEGRATED) # legend, target

    # --- Print ocp structure --- #
    # ocp.print(to_console=False, to_graph=True)

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results in a bioviz animation --- #
    sol.print()
    sol.animate()


if __name__ == "__main__":
    main()
