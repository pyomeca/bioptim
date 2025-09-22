"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
Fatigue is applied on muscles. The arms must reach a marker placed upward in front while minimizing
the muscles activity and fatigue.
Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import platform

from bioptim import (
    MusclesBiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsOptions,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Constraint,
    ConstraintFcn,
    FatigueList,
    FatigueBounds,
    FatigueInitialGuess,
    BoundsList,
    XiaFatigue,
    XiaFatigueStabilized,
    XiaTauFatigue,
    MichaudFatigue,
    MichaudTauFatigue,
    EffortPerception,
    TauEffortPerception,
    Node,
    Axis,
    VariableType,
    Solver,
    PhaseDynamics,
)
from bioptim.examples.utils import ExampleUtils


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    fatigue_type: str,
    ode_solver: OdeSolverBase = OdeSolver.COLLOCATION(),
    torque_level: int = 0,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    n_threads: int = 8,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    fatigue_type: str
        The type of dynamics to apply ("xia" or "michaud")
    ode_solver: OdeSolverBase
        The ode solver to use
    torque_level: int
        0 no residual torque, 1 with residual torque, 2 with fatigable residual torque
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    n_threads: int
        Number ot threads to use
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    n_tau = 2
    n_muscles = 6
    tau_min, tau_max = -10, 10

    # Define fatigue parameters for each muscle and residual torque
    fatigue_dynamics = FatigueList()
    for i in range(n_muscles):
        if fatigue_type == "xia":
            fatigue_dynamics.add(XiaFatigue(LD=10, LR=10, F=0.01, R=0.002), state_only=False)
        elif fatigue_type == "xia_stabilized":
            fatigue_dynamics.add(
                XiaFatigueStabilized(LD=10, LR=10, F=0.01, R=0.002, stabilization_factor=10), state_only=False
            )
        elif fatigue_type == "michaud":
            fatigue_dynamics.add(
                MichaudFatigue(
                    LD=100, LR=100, F=0.005, R=0.005, effort_threshold=0.2, effort_factor=0.001, stabilization_factor=10
                ),
                state_only=True,
            )
        elif fatigue_type == "effort":
            fatigue_dynamics.add(EffortPerception(effort_threshold=0.2, effort_factor=0.001))
        else:
            raise ValueError("fatigue_type not implemented")
    if torque_level >= 2:
        for i in range(n_tau):
            if fatigue_type == "xia":
                fatigue_dynamics.add(
                    XiaTauFatigue(
                        XiaFatigue(LD=10, LR=10, F=5, R=10, scaling=tau_min),
                        XiaFatigue(LD=10, LR=10, F=5, R=10, scaling=tau_max),
                    ),
                    state_only=False,
                )
            elif fatigue_type == "xia_stabilized":
                fatigue_dynamics.add(
                    XiaTauFatigue(
                        XiaFatigueStabilized(LD=10, LR=10, F=5, R=10, stabilization_factor=10, scaling=tau_min),
                        XiaFatigueStabilized(LD=10, LR=10, F=5, R=10, stabilization_factor=10, scaling=tau_max),
                    ),
                    state_only=False,
                )
            elif fatigue_type == "michaud":
                fatigue_dynamics.add(
                    MichaudTauFatigue(
                        MichaudFatigue(
                            LD=10, LR=10, F=5, R=10, effort_threshold=0.15, effort_factor=0.07, scaling=tau_min
                        ),
                        MichaudFatigue(
                            LD=10, LR=10, F=5, R=10, effort_threshold=0.15, effort_factor=0.07, scaling=tau_max
                        ),
                    ),
                    state_only=False,
                )
            elif fatigue_type == "effort":
                fatigue_dynamics.add(
                    TauEffortPerception(
                        EffortPerception(effort_threshold=0.15, effort_factor=0.001, scaling=tau_min),
                        EffortPerception(effort_threshold=0.15, effort_factor=0.001, scaling=tau_max),
                    ),
                    state_only=False,
                )
            else:
                raise ValueError("fatigue_type not implemented")

    bio_model = MusclesBiorbdModel(
        biorbd_model_path,
        with_residual_torque=torque_level > 0,
        fatigue=fatigue_dynamics,
    )

    # DynamicsOptions
    dynamics = DynamicsOptions(
        expand_dynamics=expand_dynamics,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    if torque_level > 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=100)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker="target", second_marker="COM_hand", weight=0.01
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="muscles", weight=1000)

    # Constraint
    constraint = Constraint(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="target",
        second_marker="COM_hand",
        node=Node.END,
        axes=[Axis.X, Axis.Y],
    )

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = (0.07, 1.4)
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0
    x_bounds.concatenate(FatigueBounds(fatigue_dynamics, fix_first_frame=True))

    x_init = InitialGuessList()
    x_init["q"] = [1.57] * bio_model.nb_q
    x_init.concatenate(FatigueInitialGuess(fatigue_dynamics))

    # Define control path constraint
    u_bounds = BoundsList()
    if torque_level == 1:
        u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds.concatenate(FatigueBounds(fatigue_dynamics, variable_type=VariableType.CONTROLS))
    u_init = InitialGuessList()
    u_init.concatenate(FatigueInitialGuess(fatigue_dynamics, variable_type=VariableType.CONTROLS))

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraint,
        use_sx=False,
        n_threads=n_threads,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """
    biorbd_model_path = ExampleUtils.folder + "/models/arm26_constant.bioMod"
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        final_time=0.8,
        n_shooting=50,
        fatigue_type="effort",
        torque_level=1,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=platform.system() == "Linux")
    solver.set_hessian_approximation("exact")
    sol = ocp.solve(solver)
    sol.print_cost()

    # --- Show results --- #
    sol.graphs()
    sol.animate(show_meshes=True)


if __name__ == "__main__":
    main()
