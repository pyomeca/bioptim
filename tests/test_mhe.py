import pytest
import os
import shutil
from sys import platform

import numpy as np
from bioptim import (
    BiorbdModel,
    Solver,
    MovingHorizonEstimator,
    Dynamics,
    DynamicsFcn,
    InterpolationType,
    InitialGuessList,
    BoundsList,
)

from .utils import TestUtils
from bioptim.misc.enums import SolverType


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("solver", [Solver.ACADOS, Solver.IPOPT])
def test_mhe(solver, assume_phase_dynamics):
    solver = solver()
    if solver.type == SolverType.ACADOS:
        if platform == "win32":
            # ACADOS is not installed on the CI for Windows
            return
        if not assume_phase_dynamics:
            return

    from bioptim.examples.moving_horizon_estimation import mhe as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    bio_model = BiorbdModel(bioptim_folder + "/models/cart_pendulum.bioMod")
    nq = bio_model.nb_q
    torque_max = 5  # Give a bit of slack on the max torque

    n_cycles = 5 if solver.type == SolverType.ACADOS else 1
    n_frame_by_cycle = 20
    window_len = 5
    window_duration = 0.2

    final_time = window_duration / window_len * n_cycles * n_frame_by_cycle
    x_init = np.zeros((nq * 2, window_len + 1))
    u_init = np.zeros((nq, window_len))

    target_q, _, _, _ = ocp_module.generate_data(
        bio_model, final_time, [0, np.pi / 2, 0, 0], torque_max, n_cycles * n_frame_by_cycle, 0
    )
    target = ocp_module.states_to_markers(bio_model, target_q)

    def update_functions(mhe, t, _):
        def target_func(i: int):
            return target[:, :, i : i + window_len + 1]

        mhe.update_objectives_target(target=target_func(t), list_index=0)
        return t < n_frame_by_cycle * n_cycles - window_len - 1

    bio_model = BiorbdModel(bioptim_folder + "/models/cart_pendulum.bioMod")
    sol = ocp_module.prepare_mhe(
        bio_model=bio_model,
        window_len=window_len,
        window_duration=window_duration,
        max_torque=torque_max,
        x_init=x_init,
        u_init=u_init,
        assume_phase_dynamics=assume_phase_dynamics,
    ).solve(update_functions, **ocp_module.get_solver_options(solver))

    if solver.type == SolverType.ACADOS:
        # Compare the position on the first few frames (only ACADOS, since IPOPT is not precise with current options)
        np.testing.assert_almost_equal(
            sol.states["q"][:, : -2 * window_len], target_q[:nq, : -3 * window_len - 1], decimal=3
        )
        # Clean test folder
        os.remove(f"./acados_ocp.json")
        shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_mhe_redim_xbounds_and_init(assume_phase_dynamics):
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    bio_model = BiorbdModel(root_folder + "models/cart_pendulum.bioMod")

    nq = bio_model.nb_q
    ntau = bio_model.nb_tau

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    x_init = InitialGuess(np.zeros((nq * 2, 1)), interpolation=InterpolationType.CONSTANT)
    u_init = InitialGuess(np.zeros((ntau, 1)), interpolation=InterpolationType.CONSTANT)
    x_bounds = Bounds(np.zeros((nq * 2, 1)), np.zeros((nq * 2, 1)), interpolation=InterpolationType.CONSTANT)
    u_bounds = Bounds(np.zeros((ntau, 1)), np.zeros((ntau, 1)))

    mhe = MovingHorizonEstimator(
        bio_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        window_len,
        window_duration,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=4,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    mhe.solve(update_functions, Solver.IPOPT())


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_mhe_redim_xbounds_not_implemented(assume_phase_dynamics):
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    bio_model = BiorbdModel(root_folder + "models/cart_pendulum.bioMod")
    nq = bio_model.nb_q
    ntau = bio_model.nb_tau

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
        bio_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        window_len,
        window_duration,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=4,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    with pytest.raises(
        NotImplementedError,
        match="The MHE is not implemented yet for x_bounds not being "
        "CONSTANT or CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT",
    ):
        mhe.solve(update_functions, Solver.IPOPT())
