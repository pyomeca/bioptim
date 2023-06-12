"""
This example shows how to use multinode_objectives.
It replicates the results from getting_started/pendulum.py
"""
import platform
from casadi import MX, sum1, sum2

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    InitialGuess,
    OdeSolver,
    OdeSolverBase,
    Solver,
    BiorbdModel,
    PenaltyController,
    MultinodeObjectiveList,
)


def multinode_min_controls(controllers: list[PenaltyController]) -> MX:
    """
    This function mimics the ObjectiveFcn.MINIMIZE_CONTROLS with a multinode objective.
    Note that it is better to use ObjectiveFcn.MINIMIZE_CONTROLS, here is juste a toy example.
    """
    dt = controllers[0].tf / controllers[0].ns
    out = 0
    for i, ctrl in enumerate(controllers):
        out += sum1(ctrl.controls["tau"].cx_start ** 2) * dt
    return out


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    assume_phase_dynamics: bool = False,
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

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(
        multinode_min_controls,
        nodes_phase=[0 for _ in range(n_shooting)],
        nodes=[i for i in range(n_shooting)],
        weight=10,
        quadratic=False,
        expand=False,
    )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        multinode_objectives=multinode_objectives,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,  # This has to be set to 1 by definition.
        assume_phase_dynamics=assume_phase_dynamics,  # This has to be set to False by definition.
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    n_shooting = 30
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=n_shooting)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # show_online_optim=platform.system() == "Linux"

    # --- Show the results in a bioviz animation --- #
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
