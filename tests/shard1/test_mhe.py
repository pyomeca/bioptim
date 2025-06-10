import pytest
import os
import shutil
from sys import platform

from bioptim.misc.enums import SolverType
from bioptim import (
    BiorbdModel,
    Solver,
    MovingHorizonEstimator,
    Dynamics,
    InterpolationType,
    BoundsList,
    PhaseDynamics,
    SolutionMerge,
)
import numpy as np
import numpy.testing as npt

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("solver", [Solver.ACADOS, Solver.IPOPT])
def test_mhe(solver, phase_dynamics):
    solver = solver()
    if solver.type == SolverType.ACADOS:
        if platform == "win32":
            # ACADOS is not installed on the CI for Windows
            return
        if phase_dynamics == PhaseDynamics.ONE_PER_NODE:
            return

    from bioptim.examples.moving_horizon_estimation import mhe as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

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
        n_threads=4 if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE else 1,
        expand_dynamics=True,
    ).solve(update_functions, **ocp_module.get_solver_options(solver))

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    if solver.type == SolverType.ACADOS:
        # Compare the position on the first few frames (only ACADOS, since IPOPT is not precise with current options)
        npt.assert_almost_equal(states["q"][:, : -2 * window_len], target_q[:nq, : -3 * window_len], decimal=3)
        # Clean test folder
        os.remove(f"./acados_ocp.json")
        shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_mhe_redim_xbounds_and_init(phase_dynamics):
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    bio_model = TorqueBiorbdModel(root_folder + "models/cart_pendulum.bioMod")

    nq = bio_model.nb_q
    ntau = bio_model.nb_tau

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    x_bounds = BoundsList()
    x_bounds.add(
        "q", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)), interpolation=InterpolationType.CONSTANT
    )
    x_bounds.add(
        "qdot", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)), interpolation=InterpolationType.CONSTANT
    )
    u_bounds = BoundsList()
    u_bounds["tau"] = np.zeros((ntau, 1)), np.zeros((ntau, 1))

    mhe = MovingHorizonEstimator(
        bio_model,
        Dynamics(phase_dynamics=phase_dynamics),
        window_len,
        window_duration,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=8 if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE else 1,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    mhe.solve(update_functions, Solver.IPOPT())


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_mhe_redim_xbounds_not_implemented(phase_dynamics):
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    bio_model = TorqueBiorbdModel(root_folder + "models/cart_pendulum.bioMod")
    nq = bio_model.nb_q
    ntau = bio_model.nb_tau

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    x_bounds = BoundsList()
    x_bounds.add(
        "q",
        min_bound=np.zeros((nq, window_len + 1)),
        max_bound=np.zeros((nq, window_len + 1)),
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_bounds.add(
        "qdot",
        min_bound=np.zeros((nq, window_len + 1)),
        max_bound=np.zeros((nq, window_len + 1)),
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_bounds = BoundsList()
    u_bounds["tau"] = np.zeros((ntau, 1)), np.zeros((ntau, 1))

    mhe = MovingHorizonEstimator(
        bio_model,
        Dynamics(phase_dynamics=phase_dynamics),
        window_len,
        window_duration,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=8 if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE else 1,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    with pytest.raises(
        NotImplementedError,
        match="The MHE is not implemented yet for x_bounds not being "
        "CONSTANT or CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT",
    ):
        mhe.solve(update_functions, Solver.IPOPT())
