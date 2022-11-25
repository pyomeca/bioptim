import os
import numpy as np
from bioptim import (
    BiorbdModel,
    Solver,
    )


def test_custom_model():
    from bioptim.examples.custom_model import main as ocp_module
    from bioptim.examples.custom_model import my_model as model
    from bioptim.examples.custom_model import custom_dynamics as dynamics

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        model=model.MyModel(),
        final_time=1,
        n_shooting=30,
        configure_dynamics=dynamics.custom_configure_my_dynamics,
        dynamics=dynamics.custom_dynamics,
    )

    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_q(), 1)
    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_qdot(), 1)
    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_qddot(), 1)
    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_generalized_torque(), 1)
    assert ocp.nlp[0].model.nb_quat() == 0
    np.testing.assert_almost_equal(ocp.nlp[0].model.mass(), 1)
    assert ocp.nlp[0].model.name_dof() == ["rotx"]
    assert ocp.nlp[0].model.path() is None

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(2)
    sol = ocp.solve(solver=solver)

    np.testing.assert_almost_equal(sol.states["q"][0,0], np.array([0]))
    np.testing.assert_almost_equal(sol.states["q"][0,-1], np.array([3.14]))
    np.testing.assert_almost_equal(sol.states["qdot"][0,0], np.array([0]))
    np.testing.assert_almost_equal(sol.states["qdot"][0,-1], np.array([0]))







