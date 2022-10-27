"""
An example of how to use multi-start to find local minima from different initial guesses.
This example is a variation of the pendulum example in getting_started/pendulum.py.
"""
import pickle
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
    CostType,
    Solver,
    NoisedInitialGuess,
    InterpolationType,
    MultiStart,
    Solution,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    seed: int = 0,
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
    x_init = NoisedInitialGuess(
        [0] * (n_q + n_qdot),
        interpolation=InterpolationType.CONSTANT,
        bounds=x_bounds,
        noise_magnitude=0.5,
        n_shooting=n_shooting,
        seed=seed,
    )

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    # TODO: To be changed for n_shooting
    # u_init = NoisedInitialGuess(
    #     [0] * n_tau,
    #     interpolation=InterpolationType.CONSTANT,
    #     bounds=u_bounds,
    #     noise_magnitude=0.5,
    #     n_shooting=n_shooting-1,
    #     seed=seed,
    # )
    u_init = InitialGuess([tau_init] * n_tau, interpolation=InterpolationType.CONSTANT)

    ocp = OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        n_threads=1,  # You cannot use multi-threading for the resolution of the ocp with multi-start
    )

    ocp.add_plot_penalty(CostType.ALL)

    return ocp


def save_results(sol: Solution, biorbd_model_path: str, final_time: float, n_shooting: int, seed: int):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    seed: int
        The seed to use for the random initial guess
    """
    # OptimalControlProgram.save(sol, f"solutions/pendulum_multi_start_random{seed}.bo", stand_alone=True)
    states = sol.states["all"]
    with open(f"pendulum_multi_start_random_states_{seed}.pkl", "wb") as file:
        pickle.dump(states, file)


def prepare_multi_start(biorbd_model_path: list, final_time: list, n_shooting: list, seed: list) -> MultiStart:
    """
    The initialization of the multi-start
    """
    return MultiStart(
        prepare_ocp,
        solver=Solver.IPOPT(show_online_optim=False),  # You cannot use show_online_optim with multi-start
        callback_function=save_results,
        n_pools=4,
        biorbd_model_path=biorbd_model_path,
        final_time=final_time,
        n_shooting=n_shooting,
        seed=seed,
    )


def main():

    # --- Prepare the multi-start and run it --- #
    multi_start = prepare_multi_start(
        biorbd_model_path=["models/pendulum.bioMod"], final_time=[1], n_shooting=[30, 40, 50], seed=[0, 1, 2, 3]
    )
    multi_start.solve()


if __name__ == "__main__":
    main()
