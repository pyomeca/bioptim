"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways. Fatigue
is applied on torques.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses.

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution.
"""

import platform

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    InitialGuessList,
    Objective,
    BoundsList,
    FatigueBounds,
    FatigueInitialGuess,
    FatigueList,
    XiaFatigue,
    XiaFatigueStabilized,
    XiaTauFatigue,
    MichaudFatigue,
    MichaudTauFatigue,
    EffortPerception,
    TauEffortPerception,
    ObjectiveFcn,
    VariableType,
    Solver,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    fatigue_type: str,
    split_controls: bool,
    use_sx: bool = True,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
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
    fatigue_type: str
        The type of dynamics to apply ("xia" or "michaud")
    split_controls: bool
        If the tau should be split into minus and plus or a if_else should be used
    use_sx: bool
        If the program should be built from SX (True) or MX (False)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", expand=expand_dynamics)

    # Fatigue parameters
    fatigue_dynamics = FatigueList()
    for i in range(n_tau):
        if fatigue_type == "xia":
            fatigue_dynamics.add(
                XiaTauFatigue(
                    XiaFatigue(LD=100, LR=100, F=5, R=10, scaling=tau_min),
                    XiaFatigue(LD=100, LR=100, F=5, R=10, scaling=tau_max),
                    state_only=False,
                    split_controls=split_controls,
                ),
            )
        elif fatigue_type == "xia_stabilized":
            fatigue_dynamics.add(
                XiaTauFatigue(
                    XiaFatigueStabilized(LD=100, LR=100, F=5, R=10, stabilization_factor=10, scaling=tau_min),
                    XiaFatigueStabilized(LD=100, LR=100, F=5, R=10, stabilization_factor=10, scaling=tau_max),
                    state_only=False,
                    split_controls=split_controls,
                ),
            )
        elif fatigue_type == "michaud":
            fatigue_dynamics.add(
                MichaudTauFatigue(
                    MichaudFatigue(
                        LD=100,
                        LR=100,
                        F=0.005,
                        R=0.005,
                        effort_threshold=0.2,
                        effort_factor=0.001,
                        stabilization_factor=10,
                        scaling=tau_min,
                    ),
                    MichaudFatigue(
                        LD=100,
                        LR=100,
                        F=0.005,
                        R=0.005,
                        effort_threshold=0.2,
                        effort_factor=0.001,
                        stabilization_factor=10,
                        scaling=tau_max,
                    ),
                    state_only=False,
                    split_controls=split_controls,
                ),
            )
        elif fatigue_type == "effort":
            fatigue_dynamics.add(
                TauEffortPerception(
                    EffortPerception(effort_threshold=0.2, effort_factor=0.001, scaling=tau_min),
                    EffortPerception(effort_threshold=0.2, effort_factor=0.001, scaling=tau_max),
                    split_controls=split_controls,
                )
            )
        else:
            raise ValueError("fatigue_type not implemented")

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        fatigue=fatigue_dynamics,
        phase_dynamics=phase_dynamics,
        expand_dynamics=expand_dynamics,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    x_bounds.concatenate(FatigueBounds(fatigue_dynamics, fix_first_frame=True))
    if fatigue_type != "effort":
        x_bounds["tau_minus_ma"][1, 0] = 0  # The rotation dof is passive (fatigue_ma = 0)
        x_bounds["tau_plus_ma"][1, 0] = 0  # The rotation dof is passive (fatigue_ma = 0)
        if fatigue_type == "xia":
            x_bounds["tau_minus_mr"][1, 0] = 1  # The rotation dof is passive (fatigue_mr = 1)
            x_bounds["tau_plus_mr"][1, 0] = 1  # The rotation dof is passive (fatigue_mr = 1)

    # Initial guess
    x_init = InitialGuessList()
    x_init.concatenate(FatigueInitialGuess(fatigue_dynamics))

    # Define control path constraint
    u_bounds = FatigueBounds(fatigue_dynamics, variable_type=VariableType.CONTROLS)
    if split_controls:
        u_bounds["tau_minus"][1, :] = 0  # The rotation dof is passive
        u_bounds["tau_plus"][1, :] = 0  # The rotation dof is passive
    else:
        u_bounds["tau"][1, :] = 0  # The rotation dof is passive
    u_init = FatigueInitialGuess(fatigue_dynamics, variable_type=VariableType.CONTROLS)

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
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path="models/pendulum.bioMod",
        final_time=1,
        split_controls=False,
        n_shooting=30,
        fatigue_type="effort",
    )

    # --- Print ocp structure --- #
    ocp.add_plot_penalty()
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show the results in a bioviz animation --- #
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
