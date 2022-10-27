import pytest
import os
import shutil
from sys import platform

import numpy as np
import biorbd_casadi as biorbd
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
from bioptim.misc.enums import SolverType


@pytest.mark.parametrize("solver", [Solver.ACADOS, Solver.IPOPT])
def test_mhe(solver):
    solver = solver()
    if platform == "win32" and solver.type == SolverType.ACADOS:
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.moving_horizon_estimation import mhe as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    biorbd_model = biorbd.Model(bioptim_folder + "/models/cart_pendulum.bioMod")
    nq = biorbd_model.nbQ()
    torque_max = 5  # Give a bit of slack on the max torque

    n_cycles = 5 if solver.type == SolverType.ACADOS else 1
    n_frame_by_cycle = 20
    window_len = 5
    window_duration = 0.2

    final_time = window_duration / window_len * n_cycles * n_frame_by_cycle
    x_init = np.zeros((nq * 2, window_len + 1))
    u_init = np.zeros((nq, window_len))

    target_q, _, _, _ = ocp_module.generate_data(
        biorbd_model, final_time, [0, np.pi / 2, 0, 0], torque_max, n_cycles * n_frame_by_cycle, 0
    )
    target = ocp_module.states_to_markers(biorbd_model, target_q)

    def update_functions(mhe, t, _):
        def target_func(i: int):
            return target[:, :, i : i + window_len + 1]

        mhe.update_objectives_target(target=target_func(t), list_index=0)
        return t < n_frame_by_cycle * n_cycles - window_len - 1

    biorbd_model = biorbd.Model(bioptim_folder + "/models/cart_pendulum.bioMod")
    sol = ocp_module.prepare_mhe(
        biorbd_model=biorbd_model,
        window_len=window_len,
        window_duration=window_duration,
        max_torque=torque_max,
        x_init=x_init,
        u_init=u_init,
    ).solve(update_functions, **ocp_module.get_solver_options(solver))

    if solver.type == SolverType.ACADOS:
        # Compare the position on the first few frames (only ACADOS, since IPOPT is not precise with current options)
        np.testing.assert_almost_equal(
            sol.states["q"][:, : -2 * window_len], target_q[:nq, : -3 * window_len - 1], decimal=3
        )
        # Clean test folder
        os.remove(f"./acados_ocp.json")
        shutil.rmtree(f"./c_generated_code/")


def test_mhe_redim_xbounds_and_init():
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    biorbd_model = biorbd.Model(root_folder + "models/cart_pendulum.bioMod")

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

    mhe.solve(update_functions, Solver.IPOPT())


def test_mhe_redim_xbounds_not_implemented():
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    biorbd_model = biorbd.Model(root_folder + "models/cart_pendulum.bioMod")
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
        mhe.solve(update_functions, Solver.IPOPT())
