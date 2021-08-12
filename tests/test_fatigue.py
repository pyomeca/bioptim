import pytest
import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils

import biorbd_casadi as biorbd

from bioptim.misc.enums import Fatigue
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    Dynamics,
    DynamicsFcn,
    QAndQDotBounds,
    InitialGuess,
    Objective,
    OdeSolver,
    Bounds,
    Constraint,
    ConstraintFcn,
    XiaTorqueFatigue,
    XiaFatigueDynamicsList,
    XiaFatigueStateInitialGuess,
    XiaFatigueStateBounds,
    XiaFatigueControlsBounds,
    XiaFatigueControlsInitialGuess,
    Node,
    Axis,
)


def prepare_ocp_static_arm(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    with_torque: bool = False,
    fatigue: list = None,
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
    ode_solver: OdeSolver
        The ode solver to use
    with_torque: bool
        True if we use residual torque
    fatigue: list
        The type of fatigue applied on the system
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    n_muscles = biorbd_model.nbMuscleTotal()
    muscle_min, muscle_max, muscle_init = 0, 1, 0.3

    tau_min, tau_max, tau_init = -1, 1, 0

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.MUSCLE_DRIVEN, expand=False, fatigue=fatigue, with_torque=with_torque
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker="target", second_marker="COM_hand", weight=0.01
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles_mf", weight=10000)  # Minimize fatigue

    # Constraint
    constraint = Constraint(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="target",
        second_marker="COM_hand",
        node=Node.END,
        axes=[Axis.X, Axis.Y],
    )

    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, 0] = (0.07, 1.4, 0, 0)
    x_bounds.concatenate(XiaFatigueStateBounds(biorbd_model, has_muscles=True, has_torque=False))

    x_init = InitialGuess([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())
    x_init.concatenate(
        XiaFatigueStateInitialGuess(
            biorbd_model,
            has_muscles=True,
            has_torque=False,
            muscle_init=muscle_init,
            tau_init=tau_init,
            tau_max=tau_max,
        )
    )

    # Define control path constraint
    muscle_bounds = Bounds([muscle_min] * n_muscles, [muscle_max] * n_muscles)
    muscle_init = InitialGuess([muscle_init] * n_muscles)
    u_bounds = XiaFatigueControlsBounds(biorbd_model, muscles=muscle_bounds)
    u_init = XiaFatigueControlsInitialGuess(biorbd_model, muscles=muscle_init)

    # Define fatigue parameters for each muscle and residual torque
    fatigue_dynamics = XiaFatigueDynamicsList()
    for _ in range(n_muscles):
        fatigue_dynamics.add_muscle(LD=10, LR=10, F=0.01, R=0.002)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraint,
        ode_solver=ode_solver,
        fatigue_dynamics=fatigue_dynamics,
        use_sx=False,
        n_threads=8,
    )


def prepare_ocp_pendulum(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    fatigue: list = None,
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
    fatigue: list
        The type of fatigue applied on the system

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, fatigue=[Fatigue.TAU])

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14
    x_bounds.concatenate(XiaFatigueStateBounds(biorbd_model, has_muscles=False, has_torque=True))

    # Initial guess
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))
    x_init.concatenate(
        XiaFatigueStateInitialGuess(
            biorbd_model,
            has_torque=True,
            tau_init=tau_init,
            tau_max=tau_max,
        )
    )

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0
    u_bounds = XiaFatigueControlsBounds(biorbd_model, torque=u_bounds)

    u_init = InitialGuess([tau_init] * n_tau)
    u_init = XiaFatigueControlsInitialGuess(biorbd_model, torque=u_init)

    # Fatigue parameters
    fatigue_dynamics = XiaFatigueDynamicsList()
    for i in range(n_tau):
        fatigue_dynamics.add_torque(
            XiaTorqueFatigue(LD=100, LR=100, F=0.9, R=0.01, tau_max=tau_min),
            XiaTorqueFatigue(LD=100, LR=100, F=0.9, R=0.01, tau_max=tau_max),
            index=i,
        )

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=True,
        fatigue_dynamics=fatigue_dynamics,
    )


def test_fatigable_muscles():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/fatigue/arm26_constant.bioMod"
    prepare_ocp_static_arm(
        biorbd_model_path=model_path,
        final_time=3,
        n_shooting=50,
        with_torque=False,
        fatigue=[Fatigue.MUSCLES_STATE_ONLY],
    )


def test_fatigable_torque():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/fatigue/pendulum.bioMod"
    prepare_ocp_pendulum(biorbd_model_path=model_path, final_time=3, n_shooting=100, fatigue=[Fatigue.TAU_STATE_ONLY])
