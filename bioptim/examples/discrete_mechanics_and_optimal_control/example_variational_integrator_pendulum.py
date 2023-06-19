"""
This example presents how to implement an optimal control program with the variational integrator.
The simulation is a pendulum simulation, its behaviour should be the same as the one in
bioptim/examples/getting_started/pendulum.py
"""
from bioptim import (
    Bounds,
    InitialGuess,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    Solver,
)
import numpy as np

from bioptim.examples.discrete_mechanics_and_optimal_control.biorbd_model_holonomic import BiorbdModelCustomHolonomic
from bioptim.examples.discrete_mechanics_and_optimal_control.variational_optimal_control_program import (
    VariationalOptimalControlProgram,
)


def prepare_ocp(
    bio_model_path: str,
    final_time: float,
    n_shooting: int,
    use_sx: bool = True,
) -> VariationalOptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    bio_model_path: str
        The path to the biorbd model.
    final_time: float
        The time in second required to perform the task.
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program.
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM).

    Returns
    -------
    The OptimalControlProgram ready to be solved.
    """

    bio_model = BiorbdModelCustomHolonomic(bio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14
    # Initial guess
    n_q = bio_model.nb_q
    x_0_guess = np.zeros(n_shooting + 1)
    x_linear_guess = np.linspace(0, np.pi, n_shooting + 1)
    x_init = InitialGuess([x_0_guess, x_linear_guess], interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate
    # Initial guess
    u_init = InitialGuess([0] * n_q)

    # Give the initial and final some min and max bounds
    qdot_start_bounds = Bounds([0.0, 0.0], [0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    qdot_end_bounds = Bounds([0.0, 0.0], [0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    # And an initial guess
    qdot_start_init = InitialGuess([0] * n_q)
    qdot_end_init = InitialGuess([0] * n_q)

    return VariationalOptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        q_init=x_init,
        u_init=u_init,
        q_bounds=x_bounds,
        u_bounds=u_bounds,
        qdot_start_init=qdot_start_init,
        qdot_start_bounds=qdot_start_bounds,
        qdot_end_init=qdot_end_init,
        qdot_end_bounds=qdot_end_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """
    n_shooting = 100

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(bio_model_path="models/pendulum.bioMod", final_time=1, n_shooting=n_shooting)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show the results in a bioviz animation --- #
    sol.detailed_cost_values()  # /!\ Since the last controls are nan the costs are not accurate /!\
    sol.print_cost()
    sol.animate()

    # --- Show the graph results --- #
    # The states are displayed piecewise constant, but actually they are not.
    sol.graphs()

    print(f"qdot0 :{sol.parameters['qdot0'].squeeze()}")
    print(f"qdotN :{sol.parameters['qdotN'].squeeze()}")


if __name__ == "__main__":
    main()
