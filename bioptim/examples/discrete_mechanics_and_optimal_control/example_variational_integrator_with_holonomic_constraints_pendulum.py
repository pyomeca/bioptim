"""
This example presents how to implement a holonomic constraint with the variational integrator.
The simulation is a pendulum simulation, the model has been freed in translation on the z-axis. A holonomic constraint
constrains the z-distance between two markers to remain null.
The behaviour of the pendulum should be the same as the one in bioptim/examples/getting_started/pendulum.py
"""
from bioptim import (
    BoundsList,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    Solver,
    VariationalBiorbdModel,
    VariationalOptimalControlProgram,
)
import numpy as np


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

    bio_model = VariationalBiorbdModel(bio_model_path)
    # Holonomic constraints: The pendulum must not move on the z axis
    (
        constraints_func,
        constraints_jacobian_func,
        constraints_double_derivative_func,
    ) = HolonomicConstraintsFcn.superimpose_markers(bio_model, marker_1="marker_1", index=slice(2, 3))
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "holonomic_constraints", constraints_func, constraints_jacobian_func, constraints_double_derivative_func
    )
    bio_model.set_dependencies(holonomic_constraints)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = [-1, -1, -2 * np.pi], [5, 5, 2 * np.pi]
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][2, -1] = 3.14
    x_bounds["lambdas"] = [-1000], [1000]

    # Initial guess
    n_q = bio_model.nb_q
    x_0_guess = np.zeros(n_shooting + 1)
    x_linear_guess = np.linspace(0, np.pi, n_shooting + 1)
    x_init = InitialGuessList()
    x_init.add("q", [x_0_guess, x_0_guess, x_linear_guess], interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively impose forces on z axis
    u_bounds["tau"][2, :] = 0  # Prevent the model from actively rotate

    # Make sure all are declared
    qdot_bounds = BoundsList()
    # Start and finish with zero velocity
    qdot_bounds.add("qdot_start", min_bound=[0] * n_q, max_bound=[0] * n_q, interpolation=InterpolationType.CONSTANT)
    qdot_bounds.add("qdot_end", min_bound=[0] * n_q, max_bound=[0] * n_q, interpolation=InterpolationType.CONSTANT)

    return VariationalOptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        q_init=x_init,
        q_bounds=x_bounds,
        u_bounds=u_bounds,
        qdot_bounds=qdot_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """
    n_shooting = 100

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(bio_model_path="models/pendulum_holonomic.bioMod", final_time=1, n_shooting=n_shooting)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show the results in a bioviz animation --- #
    sol.print_cost()  # /!\ Since the last controls are nan the costs are not accurate /!\
    sol.animate()

    # --- Show the graph results --- #
    # The states (q and lambdas) are displayed piecewise constant, but actually they are not.
    sol.graphs()

    print(f"qdot0 :{sol.parameters['qdot_start'].squeeze()}")
    print(f"qdot_end :{sol.parameters['qdot_end'].squeeze()}")


if __name__ == "__main__":
    main()
