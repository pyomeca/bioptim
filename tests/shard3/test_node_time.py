import os

import pytest
import numpy as np
from casadi import Function, vertcat, DM

from bioptim import OdeSolver, Solver, PhaseDynamics, SolutionMerge


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_node_time(ode_solver, phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    sol = ocp.solve(solver=solver)
    all_node_time = np.array([ocp.node_time(0, i) for i in range(ocp.nlp[0].ns + 1)])

    computed_t = Function("time", [nlp.dt for nlp in ocp.nlp], [vertcat(all_node_time)])(sol.t_span[0][-1])
    time = sol.decision_time()
    expected_t = DM([0] + [t[-1] for t in time][:-1])
    np.testing.assert_almost_equal(np.array(computed_t), np.array(expected_t))
