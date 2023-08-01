import numpy as np
import pytest

from .utils import TestUtils


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_continuity_as_objective(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_continuity_as_objective as ocp_module

    np.random.seed(42)
    model_path = TestUtils.bioptim_folder() + "/examples/getting_started/models/pendulum_maze.bioMod"

    ocp = ocp_module.prepare_ocp_first_pass(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=3,
        state_continuity_weight=1000000,
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=True,
    )
    sol = ocp.solve()

    expected = np.array([-0.1376, 2.9976372])
    np.testing.assert_almost_equal(sol.states["q"][:, -1], expected)
