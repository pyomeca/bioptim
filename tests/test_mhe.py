import pytest
import os
import shutil
from sys import platform

import numpy as np
import biorbd
from bioptim import (
    Solver,
    MovingHorizonEstimator,
    Dynamics,
    DynamicsFcn,
    InterpolationType,
    InitialGuess,
    Bounds,
)

from .utils import TestUtils


@pytest.mark.parametrize("solver", [Solver.ACADOS, Solver.IPOPT])
def test_mhe(solver):
    if platform == "win32" and solver == Solver.ACADOS:
        print("Test for ACADOS on Windows is skipped")
        return
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    pendulum = TestUtils.load_module(root_folder + "mhe.py")
    biorbd_model = biorbd.Model(root_folder + "cart_pendulum.bioMod")
    nq = biorbd_model.nbQ()
    torque_max = 5  # Give a bit of slack on the max torque

    n_cycles = 5 if solver == Solver.ACADOS else 1
    n_frame_by_cycle = 20
    window_len = 5
    window_duration = 0.2

    final_time = window_duration / window_len * n_cycles * n_frame_by_cycle
    x_init = np.zeros((nq * 2, window_len + 1))
    u_init = np.zeros((nq, window_len))

    target_q, _, _, _ = pendulum.generate_data(
        biorbd_model, final_time, [0, np.pi / 2, 0, 0], torque_max, n_cycles * n_frame_by_cycle, 0
    )
    target = pendulum.states_to_markers(biorbd_model, target_q)

    def update_functions(mhe, t, _):
        def target_func(i: int):
            return target[:, :, i : i + window_len + 1]

        mhe.update_objectives_target(target=target_func(t), list_index=0)
        return t < n_frame_by_cycle * n_cycles - window_len - 1

    biorbd_model = biorbd.Model(root_folder + "cart_pendulum.bioMod")
    sol = pendulum.prepare_mhe(
        biorbd_model=biorbd_model,
        window_len=window_len,
        window_duration=window_duration,
        max_torque=torque_max,
        x_init=x_init,
        u_init=u_init,
    ).solve(update_functions, **pendulum.get_solver_options(solver))

    # Clean test folder
    if solver == Solver.ACADOS:
        # Compare the position on the first few frames (only ACADOS, since IPOPT is not precise with current options)
        np.testing.assert_almost_equal(
            sol.states["q"][:, : -2 * window_len], target_q[:nq, : -3 * window_len - 1], decimal=3
        )
        os.remove(f"./acados_ocp.json")
        shutil.rmtree(f"./c_generated_code/")


def test_mhe_redim_xbounds_and_init():
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    biorbd_model = biorbd.Model(root_folder + "cart_pendulum.bioMod")
    nq = biorbd_model.nbQ()
    ntau = biorbd_model.nbGeneralizedTorque()

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    x_init = InitialGuess(np.zeros((nq * 2, 1)), interpolation=InterpolationType.CONSTANT)
    u_init = InitialGuess(np.zeros((ntau, 1)), interpolation=InterpolationType.CONSTANT)
    x_bounds = Bounds(np.zeros((nq * 2, 1)), np.zeros((nq * 2, 1)), interpolation=InterpolationType.CONSTANT)
    u_bounds = Bounds(np.zeros((ntau, 1)), np.zeros((ntau, 1)))

    mhe = MovingHorizonEstimator(
        biorbd_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        window_len,
        window_duration,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=4,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    mhe.solve(update_functions, Solver.IPOPT)


def test_mhe_redim_xbounds_not_implemented():
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    biorbd_model = biorbd.Model(root_folder + "cart_pendulum.bioMod")
    nq = biorbd_model.nbQ()
    ntau = biorbd_model.nbGeneralizedTorque()

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    x_init = InitialGuess(np.zeros((nq * 2, 1)), interpolation=InterpolationType.CONSTANT)
    u_init = InitialGuess(np.zeros((ntau, 1)), interpolation=InterpolationType.CONSTANT)
    x_bounds = Bounds(
        np.zeros((nq * 2, window_len + 1)),
        np.zeros((nq * 2, window_len + 1)),
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_bounds = Bounds(np.zeros((ntau, 1)), np.zeros((ntau, 1)))

    mhe = MovingHorizonEstimator(
        biorbd_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        window_len,
        window_duration,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=4,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    with pytest.raises(
        NotImplementedError,
        match="The MHE is not implemented yet for x_bounds not being "
        "CONSTANT or CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT",
    ):
        mhe.solve(update_functions, Solver.IPOPT)
