from sys import platform
import numpy as np
import numpy.testing as npt
import pytest

from bioptim import (
    PhaseDynamics,
    SolutionMerge,
    Solver,
)
from ..utils import TestUtils


@pytest.mark.parametrize(
    "phase_dynamics",
    [
        PhaseDynamics.SHARED_DURING_THE_PHASE,
        PhaseDynamics.ONE_PER_NODE,
    ],
)
@pytest.mark.parametrize(
    "use_sx",
    [
        True,
        False,
    ],
)
def test_example_param_obj_and_param_scaling(
    phase_dynamics,
    use_sx,
):

    if platform == "darwin" or platform == "win32":
        pytest.skip("This test is not working on MacOS or Windows")

    from bioptim.examples.toy_examples.feature_examples import example_parameter_scaling as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    final_time = 1
    n_shooting = 10

    ocp_to_track = ocp_module.generate_dat_to_track(
        biorbd_model_path=bioptim_folder + "/../../models/pendulum_wrong_gravity.bioMod",
        final_time=final_time,
        n_shooting=n_shooting,
    )
    sol_to_track = ocp_to_track.solve(Solver.IPOPT(show_online_optim=False))
    q_to_track = sol_to_track.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot_to_track = sol_to_track.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    tau_to_track = sol_to_track.decision_controls(to_merge=SolutionMerge.NODES)["tau"]

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=final_time,
        n_shooting=n_shooting,
        min_g=np.array([0, -5, -50]),
        max_g=np.array([0, 5, -5]),
        q_to_track=q_to_track,
        qdot_to_track=qdot_to_track,
        tau_to_track=tau_to_track,
        use_sx=use_sx,
        phase_dynamics=phase_dynamics,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(sum([cost["cost_value_weighted"] for cost in sol.detailed_cost]), f[0, 0])

    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (0, 1))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]), decimal=5)
    npt.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0]), decimal=5)

    npt.assert_almost_equal(q[:, 0], np.array([0.0, 0.0]), decimal=5)
    npt.assert_almost_equal(q[:, 5], np.array([-0.26673, 2.53154]), decimal=5)
    npt.assert_almost_equal(q[:, -1], np.array([0.0, 3.14]), decimal=5)

    param_not_scaled = sol.decision_parameters(scaled=False)
    param_scaled = sol.decision_parameters(scaled=True)

    npt.assert_almost_equal(param_not_scaled["gravity_xyz"], np.array([0.0, 3.89005766, -14.71310026]), decimal=5)
    npt.assert_almost_equal(param_scaled["gravity_xyz"], np.array([0.0, 3.89005766, -1.47131003]), decimal=5)
