"""
A pendulum simulation copying the example from bioptim/examples/getting_started/pendulum.py but integrated by the
variational integrator.
"""
import pickle
import numpy as np

from bioptim import (
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    Solver,
    BiorbdModel,
    DynamicsList,
    ParameterList,
    InterpolationType,
)

from variational_integrator import *
from save_results import save_results


def custom_configure_constrained(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, bio_model, constraints, jac, expand: bool = True
):
    """
    As we are here in a constrained problem, the lambdas are added as states.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp.
    nlp: NonLinearProgram
        A reference to the phase.
    bio_model: BiorbdModel
        The biorbd model.
    constraints: Function
        The constraints function.
    jac: Function
        The jacobian of the constraints.
    expand: bool
        If the dynamics should be expanded with casadi.
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_new_variable(
        "lambdas",
        ["Seg1_TransZ"],  # Note: to be generalized when more constraints are added
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=False,
    )
    custom_dynamics_function(ocp, nlp, bio_model, constraints, jac, expand)


def prepare_ocp(
    bio_model_path: str,
    final_time: float,
    n_shooting: int,
    use_sx: bool = True,
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
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
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node.

    Returns
    -------
    The OptimalControlProgram ready to be solved.
    """

    bio_model = BiorbdModel(bio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Path constraint
    x_bounds = Bounds([-1, -1, -2 * np.pi, -20], [5, 5, 2 * np.pi, 20])
    x_bounds[:3, [0, -1]] = 0
    x_bounds[2, -1] = 3.14
    # Initial guess
    n_q = bio_model.nb_q
    x_0_guess = np.zeros(n_shooting + 1)
    x_linear_guess = np.linspace(0, np.pi, n_shooting + 1)
    x_init = InitialGuess([x_0_guess, x_0_guess, x_linear_guess, x_0_guess], interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively impose forces on z axis
    u_bounds[2, :] = 0  # Prevent the model from actively rotate
    # Initial guess
    u_init = InitialGuess([tau_init] * n_q)

    # Declare parameters for the initial and final velocities
    parameters = ParameterList()
    # Give the parameter some min and max bounds
    qdot0_bounds = Bounds([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    qdotN_bounds = Bounds([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], interpolation=InterpolationType.CONSTANT)
    # And an initial guess
    qdot0_init = InitialGuess([0] * n_q)
    qdotN_init = InitialGuess([0] * n_q)

    parameters.add(
        "qdot0",  # The name of the parameter
        function=qdot_function,  # The function that modifies the biorbd model
        initial_guess=qdot0_init,  # The initial guess
        bounds=qdot0_bounds,  # The bounds
        size=n_q,  # The number of elements this particular parameter vector has
    )
    parameters.add(
        "qdotN",  # The name of the parameter
        function=qdot_function,  # The function that modifies the biorbd model
        initial_guess=qdotN_init,  # The initial guess
        bounds=qdotN_bounds,  # The bounds
        size=n_q,  # The number of elements this particular parameter vector has
    )

    # Holonomic constraints: The pendulum must not move on the z axis
    q_sym = MX.sym("q", (n_q, 1))
    constraints = Function("constraint", [q_sym], [q_sym[1]], ["q"], ["constraint"]).expand()
    jac = Function("jacobian", [q_sym], [jacobian(q_sym[1], q_sym)], ["q"], ["jacobian"]).expand()

    # Dynamics
    dynamics = DynamicsList()
    expand = True
    dynamics.add(custom_configure_constrained, bio_model=bio_model, constraints=constraints, jac=jac, expand=expand)

    multinode_constraints = variational_continuity(n_shooting, use_constraints=True)

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
        use_sx=use_sx,
        assume_phase_dynamics=assume_phase_dynamics,
        skip_continuity=True,
        parameters=parameters,
        multinode_constraints=multinode_constraints,
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
    sol.detailed_cost_values()  # /!\ Since the last controls are nan the costs are not accurate /!\
    sol.print_cost()
    sol.animate()

    save_results(sol, f"results/varint_{n_shooting}_nodes_holonomic")

    with open(f"results/varint_{n_shooting}_nodes", "rb") as f:
        data = pickle.load(f)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].set_title("q_Seg1_TransY-0")
    axs[0, 0].plot(sol.time, sol.states["q"][0], "purple")
    axs[0, 0].plot(data["time"], data["states"]["q"][0], "--m")
    axs[0, 1].set_title("q_Seg1_TransZ-0")
    axs[0, 1].plot(sol.time, sol.states["q"][1], "purple")
    axs[0, 2].set_title("q_Seg1_RotX-0")
    axs[0, 2].plot(sol.time, sol.states["q"][2], "purple")
    axs[0, 2].plot(data["time"], data["states"]["q"][1], "--m")
    axs[1, 0].set_title("tau_Seg1_TransY-0")
    axs[1, 0].step(sol.time, sol.controls["tau"][0], "orange")
    axs[1, 0].step(data["time"], data["controls"]["tau"][0], "--y")
    axs[1, 1].set_title("tau_Seg1_TransZ-0")
    axs[1, 1].step(sol.time, sol.controls["tau"][1], "orange")
    axs[1, 2].set_title("tau_Seg1_RotX-0")
    axs[1, 2].step(sol.time, sol.controls["tau"][2], "orange")
    axs[1, 2].step(data["time"], data["controls"]["tau"][1], "--y")

    for i in range(2):
        for j in [0, 2]:
            axs[i, j].set_xlabel("Time (s)")
            axs[i, j].legend(["With holonomic constraint", "Without holonomic constraint"])
        axs[i, 1].set_xlabel("Time (s)")
        axs[i, 1].legend(["With holonomic constraint"])

    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Constraint force (N)")
    plt.title("Constraint force on q_Seg1_TransZ-0")
    dt = sol.time[1] - sol.time[0]
    plt.plot(sol.time, sol.states["lambdas"][0] / dt, "purple")

    plt.show()


if __name__ == "__main__":
    main()
