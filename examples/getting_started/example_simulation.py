"""
The first part of this example of a single shooting simulation from initial guesses.
It is NOT an optimal control program. It is merely the simulation of values, that is applying the dynamics.
The main goal of this kind of simulation is to get a sens of the initial guesses passed to the solver

The second part of the example is to actually solve the program and then simulate the results from this solution.
The main goal of this kind of simulation, especially in single shooting (that is not resetting the states at each node)
is to validate the dynamics of multiple shooting. If they both are equal, it usually means that a great confidence
can be held in the solution. Another goal would be to reload fast a previously saved optimized solution
"""

from bioptim import InitialGuess, Solution, Shooting, InterpolationType
import numpy as np
import pendulum


if __name__ == "__main__":
    # --- Load pendulum --- #
    ocp = pendulum.prepare_ocp(
        biorbd_model_path="pendulum.bioMod",
        final_time=2,
        n_shooting=10,
    )

    # Simulation the Initial Guess
    # Interpolation: Constant
    X = InitialGuess([0, 0, 0, 0])
    U = InitialGuess([-1, 1])

    sol_from_initial_guess = Solution(ocp, [X, U])
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS)
    print(f"Final position of q from single shooting of initial guess = {s.states['q'][:, -1]}")

    # Interpolation: Each frame (for instance, values from a previous optimization or from measured data)
    X = np.random.rand(4, 11)
    X = InitialGuess(X, interpolation=InterpolationType.EACH_FRAME)
    U = np.random.rand(2, 11)
    U = InitialGuess(U, interpolation=InterpolationType.EACH_FRAME)

    sol_from_initial_guess = Solution(ocp, [X, U])
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS)
    print(f"Final position of q from single shooting of initial guess = {s.states['q'][:, -1]}")
    # Uncomment the next line to animate the integration
    # s.animate()

    # Uncomment the following lines to graph the solution from initial guesses
    # sol_from_initial_guess.graphs(shooting_type=Shooting.SINGLE_CONTINUOUS)
    # sol_from_initial_guess.graphs(shooting_type=Shooting.MULTIPLE)

    # Simulation of the solution. It is not the graph of the solution, it is the graph of a Runge Kutta from the solution
    sol = ocp.solve()
    s_single = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS)
    # Uncomment the next line to animate the integration
    # s_single.animate()
    print(f"Final position of q from single shooting of the solution = {s_single.states['q'][:, -1]}")
    s_multiple = sol.integrate(shooting_type=Shooting.MULTIPLE, keepdims=False)
    print(f"Final position of q from multiple shooting of the solution = {s_multiple.states['q'][:, -1]}")

    # Uncomment the following lines to graph the solution from the actual solution
    # sol.graphs(shooting_type=Shooting.SINGLE_CONTINUOUS)
    # sol.graphs(shooting_type=Shooting.MULTIPLE)
