"""
An example of how to use multi-start to find local minima from different initial guesses.
This example is a variation of the pendulum example in getting_started/pendulum.py.
"""

import pickle
import os
import shutil

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    CostType,
    Solver,
    MultiStart,
    Solution,
    MagnitudeType,
    PhaseDynamics,
    SolutionMerge,
)


def prepare_ocp(
    bio_model_path: str,
    final_time: float,
    n_shooting: int,
    seed: int = 0,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    bio_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    seed: int
        The seed to use for the random initial guess
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should alos be used when multi-node penalties are added to the OCP.

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(bio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuessList()
    x_init["q"] = [0] * n_q
    x_init["qdot"] = [0] * n_qdot
    x_init.add_noise(  # Alternatively one can call add_noise to individual element as well (e.g. x_init["q"].add_noise)
        bounds=x_bounds,
        magnitude=0.5,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=[n_shooting + 1],
        seed=seed,
    )

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau
    u_init["tau"].add_noise(
        bounds=u_bounds["tau"],
        magnitude=0.5,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=n_shooting,
        seed=seed,
    )

    ocp = OptimalControlProgram(
        bio_model,
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


def construct_filepath(save_path, n_shooting, seed):
    return f"{save_path}/pendulum_multi_start_random_states_{n_shooting}_{seed}.pkl"


def save_results(
    sol: Solution,
    *combinatorial_parameters,
    **extra_parameters,
) -> None:
    """
    Callback of the post_optimization_callback, this can be used to save the results

    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    combinatorial_parameters:
        The current values of the combinatorial_parameters being treated
    extra_parameters:
        All the non-combinatorial parameters sent by the user
    """

    bio_model_path, final_time, n_shooting, seed = combinatorial_parameters
    save_folder = extra_parameters["save_folder"]

    file_path = construct_filepath(save_folder, n_shooting, seed)
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    with open(file_path, "wb") as file:
        pickle.dump(states, file)


def should_solve(*combinatorial_parameters, **extra_parameters):
    """
    Callback of the should_solve_callback, this allows the user to instruct bioptim

    Parameters
    ----------
    combinatorial_parameters:
        The current values of the combinatorial_parameters being treated
    extra_parameters:
        All the non-combinatorial parameters sent by the user
    """

    bio_model_path, final_time, n_shooting, seed = combinatorial_parameters
    save_folder = extra_parameters["save_folder"]

    file_path = construct_filepath(save_folder, n_shooting, seed)
    return not os.path.exists(file_path)


def prepare_multi_start(
    combinatorial_parameters: dict,
    save_folder: str = None,
    n_pools: int = 1,
) -> MultiStart:
    """
    The initialization of the multi-start
    """
    if not isinstance(save_folder, str):
        raise ValueError("save_folder must be an str")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    return MultiStart(
        combinatorial_parameters=combinatorial_parameters,
        prepare_ocp_callback=prepare_ocp,
        post_optimization_callback=(save_results, {"save_folder": save_folder}),
        should_solve_callback=(should_solve, {"save_folder": save_folder}),
        solver=Solver.IPOPT(show_online_optim=False),  # You cannot use show_online_optim with multi-start
        n_pools=n_pools,
    )


def main():
    # --- Prepare the multi-start and run it --- #

    bio_model_path = ["models/pendulum.bioMod"]
    final_time = [1]
    n_shooting = [30, 40, 50]
    seed = [0, 1, 2, 3]

    combinatorial_parameters = {
        "bio_model_path": bio_model_path,
        "final_time": final_time,
        "n_shooting": n_shooting,
        "seed": seed,
    }

    save_folder = "./temporary_results"
    multi_start = prepare_multi_start(
        combinatorial_parameters=combinatorial_parameters,
        save_folder=save_folder,
        n_pools=2,
    )

    multi_start.solve()

    # Delete the solutions
    shutil.rmtree(save_folder)


if __name__ == "__main__":
    main()
