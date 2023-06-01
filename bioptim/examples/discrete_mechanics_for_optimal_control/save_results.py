"""
Save the results of the ocp for comparison purposes.
"""
import pickle


def save_results(sol, c3d_file_path):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
    The solution to the ocp at the current pool
    c3d_file_path: str
    The path to the c3d file of the task
    """
    data = dict(
        states=sol.states,
        states_no_intermediate=sol.states_scaled_no_intermediate,
        controls=sol.controls,
        parameters=sol.parameters,
        iterations=sol.iterations,
        cost=sol.cost,
        real_time_to_optimize=sol.real_time_to_optimize,
        status=sol.status,
        time=sol.time,
    )
    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)
