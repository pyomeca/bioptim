# import os
#
# import pytest
# import numpy as np
#
# from bioptim import OdeSolver, Solver, PhaseDynamics
#
#
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
# def test_node_time(ode_solver, phase_dynamics):
#     # Load pendulum
#     from bioptim.examples.getting_started import pendulum as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
#         final_time=2,
#         n_shooting=10,
#         ode_solver=ode_solver(),
#         phase_dynamics=phase_dynamics,
#         expand_dynamics=ode_solver != OdeSolver.IRK,
#     )
#     solver = Solver.IPOPT(show_online_optim=False)
#     solver.set_maximum_iterations(0)
#     solver.set_print_level(0)
#
#     sol = ocp.solve(solver=solver)
#     if ode_solver == OdeSolver.RK4:
#         all_node_time = np.array([ocp.node_time(0, i) for i in range(ocp.nlp[0].ns + 1)])
#         np.testing.assert_almost_equal(sol.time, all_node_time)
#     else:
#         time_at_specific_nodes = np.array([sol.time[i] for i in range(0, 51, 5)])
#         all_node_time = np.array([ocp.node_time(0, i) for i in range(ocp.nlp[0].ns + 1)])
#         np.testing.assert_almost_equal(time_at_specific_nodes, all_node_time)
