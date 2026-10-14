"""
Tests of OdeSolver.VARIATIONAL, that owns the discrete Euler-Lagrange functions of the variational integrator.
"""

import numpy as np
import pytest
from bioptim import (
    BoundsList,
    InterpolationType,
    VariationalOptimalControlProgram,
    VariationalTorqueBiorbdModel,
)

from ..utils import TestUtils


def _prepare_variational_ocp(n_shooting: int = 5, use_sx: bool = False) -> VariationalOptimalControlProgram:
    bio_model = VariationalTorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")
    n_q = bio_model.nb_q

    q_bounds = BoundsList()
    q_bounds["q"] = bio_model.bounds_from_ranges("q")
    q_bounds["q"][:, [0, -1]] = 0
    q_bounds["q"][1, -1] = 3.14

    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_q, [100] * n_q
    u_bounds["tau"][1, :] = 0

    qdot_bounds = BoundsList()
    qdot_bounds.add("qdot_start", min_bound=[0] * n_q, max_bound=[0] * n_q, interpolation=InterpolationType.CONSTANT)
    qdot_bounds.add("qdot_end", min_bound=[0] * n_q, max_bound=[0] * n_q, interpolation=InterpolationType.CONSTANT)

    return VariationalOptimalControlProgram(
        bio_model,
        n_shooting=n_shooting,
        final_time=1,
        q_bounds=q_bounds,
        u_bounds=u_bounds,
        qdot_bounds=qdot_bounds,
        use_sx=use_sx,
    )


@pytest.mark.parametrize("use_sx", [False, True])
def test_variational_functions_are_owned_by_the_ode_solver(use_sx):
    nlp = _prepare_variational_ocp(use_sx=use_sx).nlp[0]

    assert nlp.variational_three_nodes_func.name() == "ThreeNodesIntegration"
    assert nlp.variational_first_node_func.name() == "TwoFirstNodesIntegration"
    assert nlp.variational_last_node_func.name() == "TwoLastNodesIntegration"

    # dt, q at nodes i-1, i, i+1 and tau at the same nodes
    assert nlp.variational_three_nodes_func.n_in() == 7
    assert nlp.variational_first_node_func.n_in() == 6
    assert nlp.variational_last_node_func.n_in() == 6
    assert nlp.variational_three_nodes_func.size_out(0) == (nlp.model.nb_q, 1)


def test_variational_functions_do_not_hijack_the_dynamics_defects():
    nlp = _prepare_variational_ocp().nlp[0]

    assert nlp.dynamics_defects_func is not nlp.variational_three_nodes_func
    assert nlp.extra_dynamics_defects_func == []
