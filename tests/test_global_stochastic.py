
import os
import pickle
import re
import sys
import shutil
import platform

import pytest
import numpy as np
from casadi import sum1, sum2
from bioptim import InterpolationType, OdeSolver, MultinodeConstraintList, MultinodeConstraintFcn, Node, Solver

from .utils import TestUtils


def test_arm_reaching_muscle_driven():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_muscle_driven as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    final_time = 0.8
    n_shooting = 4
    ee_initial_position = np.array([0.0, 0.2742])
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    problem_type = "CIRCLE"
    force_field_magnitude = 0

    ocp = ocp_module.prepare_socp(
            biorbd_model_path=bioptim_folder + "/models/LeuvenArmModel.bioMod",
            final_time=final_time,
            n_shooting=n_shooting,
            ee_final_position=ee_final_position,
            problem_type=problem_type,
            force_field_magnitude=force_field_magnitude
        )

    # ocp.print(to_console=True, to_graph=False)  #TODO: check to adjust the print method

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    solver.set_nlp_scaling_method('none')

    sol = ocp.solve(solver)

    # TODO: adjust the values as sol, gets updated
    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 8.444016053093858)

    # detailed cost values
    # sol.detailed_cost_values()
    # np.testing.assert_almost_equal(sol.detailed_cost[2]["cost_value_weighted"], 41.57063948309302)
    # np.testing.assert_almost_equal(sol.detailed_cost[3]["cost_value_weighted"], 41.57063948309302)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (426, 1))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, mus_activations, mus_excitations = states["q"], states["qdot"], states["muscles"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
    np.testing.assert_almost_equal(q[:, -2], np.array([0.95993109, 1.15939485]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -2], np.array((0, 0)))
    np.testing.assert_almost_equal(mus_activations[:, 0], np.array([0.03414132, 0.03292284, 0.0126227 , 0.01565839, 0.00676471,
       0.02404206]))
    np.testing.assert_almost_equal(mus_activations[:, -2], np.array([0.04160946, 0.07306185, 0.01894845, 0.02188286, 0.00068182,
       0.0253038]))
    np.testing.assert_almost_equal(mus_excitations[:, 0], np.array([0.02816048, 0.0712188 , 0.04627442, 0.0034365 , 0.00025384,
       0.03239987]))
    np.testing.assert_almost_equal(mus_excitations[:, -2], np.array([0.01826304, 0.04932192, 0.02884916, 0.01225839, 0.00113569,
       0.00867345]))

    # TODO: test the stochastic variables + update values

    # simulate
    # TestUtils.simulate(sol)  # TODO: check to adjust the simulate method


#iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
# 0  5.2443422e-01 2.05e+03 1.19e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
# 1  1.7949339e+00 1.99e+03 1.29e+03  -1.0 3.73e+01   2.0 1.08e-02 2.98e-02h  1
# 2  1.9623501e+00 1.98e+03 1.29e+03  -1.0 4.68e+00   3.3 1.13e-01 4.06e-03h  1
# 3  4.0176536e+00 1.88e+03 1.24e+03  -1.0 4.55e+00   2.9 3.66e-01 5.30e-02h  1
# 4  8.4440161e+00 1.65e+03 1.78e+03  -1.0 3.35e+00   3.3 2.20e-01 1.19e-01h  1