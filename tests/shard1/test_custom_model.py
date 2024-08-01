import pytest
import numpy as np
import numpy.testing as npt
from bioptim import Solver, PhaseDynamics, SolutionMerge


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_custom_model(phase_dynamics):
    from bioptim.examples.custom_model import main as ocp_module
    from bioptim.examples.custom_model.custom_package import my_model as model
    from bioptim.examples.custom_model.custom_package import custom_configure_my_dynamics as configure_dynamics

    ocp = ocp_module.prepare_ocp(
        model=model.MyModel(),
        final_time=1,
        n_shooting=30,
        configure_dynamics=configure_dynamics,
        phase_dynamics=phase_dynamics,
        n_threads=1,
        expand_dynamics=True,
    )

    npt.assert_almost_equal(ocp.nlp[0].model.nb_q, 1)
    npt.assert_almost_equal(ocp.nlp[0].model.nb_qdot, 1)
    npt.assert_almost_equal(ocp.nlp[0].model.nb_qddot, 1)
    npt.assert_almost_equal(ocp.nlp[0].model.nb_tau, 1)
    assert ocp.nlp[0].model.nb_quaternions == 0  # added by the ocp because it must be in any BioModel
    npt.assert_almost_equal(ocp.nlp[0].model.mass, 1)
    assert ocp.nlp[0].model.name_dof == ["rotx"]

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(2)
    sol = ocp.solve(solver=solver)

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    npt.assert_almost_equal(states["q"][0, 0], np.array([0]))
    npt.assert_almost_equal(states["q"][0, -1], np.array([3.14]))
    npt.assert_almost_equal(states["qdot"][0, 0], np.array([0]))
    npt.assert_almost_equal(states["qdot"][0, -1], np.array([0]))
