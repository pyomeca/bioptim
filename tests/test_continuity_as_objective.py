import numpy as np
from .utils import TestUtils


def test_continuity_as_objective():
    from bioptim.examples.getting_started import example_continuity_as_objective as ocp_module

    np.random.seed(42)
    model_path = TestUtils.bioptim_folder() + "/examples/getting_started/models/pendulum_maze.bioMod"

    ocp = ocp_module.prepare_ocp_first_pass(biorbd_model_path=model_path, final_time=1, n_shooting=30)
    sol = ocp.solve()

    expected = np.array(
        [
            0.0,
            -0.64973982,
            -1.04586342,
            -1.23039963,
            -1.28609966,
            -1.27453533,
            -1.17923724,
            -1.00388495,
            -0.74049525,
            -0.36559256,
            0.11697346,
            0.2119651,
            0.12688539,
            -0.39439146,
            -0.8021513,
            -1.08896348,
            -1.24678877,
            -1.29886843,
            -1.30565125,
            -1.27290884,
            -1.20026907,
            -1.0952026,
            -0.97125994,
            -0.8622114,
            -0.78413156,
            -0.68181541,
            -0.46349023,
            0.59339796,
            1.23987143,
            1.89697298,
            2.9976372,
        ]
    )
    np.testing.assert_almost_equal(sol.states["q"][-1], expected)
