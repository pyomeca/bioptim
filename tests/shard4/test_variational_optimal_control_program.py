"""
Tests of the construction of VariationalOptimalControlProgram (no solving involved).
"""

import numpy.testing as npt
from bioptim import (
    BoundsList,
    ConstraintFcn,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ParameterConstraintList,
    ParameterObjectiveList,
    VariationalOptimalControlProgram,
    VariationalTorqueBiorbdModel,
)

from ..utils import TestUtils


def _prepare_variational_ocp(**kwargs) -> VariationalOptimalControlProgram:
    bio_model = VariationalTorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")
    n_q = bio_model.nb_q

    q_bounds = BoundsList()
    q_bounds["q"] = bio_model.bounds_from_ranges("q")

    qdot_bounds = BoundsList()
    qdot_bounds.add("qdot_start", min_bound=[-1] * n_q, max_bound=[1] * n_q, interpolation=InterpolationType.CONSTANT)
    qdot_bounds.add("qdot_end", min_bound=[-1] * n_q, max_bound=[1] * n_q, interpolation=InterpolationType.CONSTANT)

    return VariationalOptimalControlProgram(
        bio_model,
        n_shooting=5,
        final_time=1,
        q_bounds=q_bounds,
        qdot_bounds=qdot_bounds,
        **kwargs,
    )


def test_variational_qdot_init_is_forwarded_to_the_matching_parameter():
    qdot_init = InitialGuessList()
    qdot_init.add("qdot_start", [1, 2])
    qdot_init.add("qdot_end", [3, 4])

    ocp = _prepare_variational_ocp(qdot_init=qdot_init)

    npt.assert_equal(ocp.parameter_init["qdot_start"].init, [[1], [2]])
    npt.assert_equal(ocp.parameter_init["qdot_end"].init, [[3], [4]])


def test_variational_parameter_penalties_are_forwarded_to_the_ocp():
    parameter_objectives = ParameterObjectiveList()
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="qdot_start", weight=1)
    parameter_constraints = ParameterConstraintList()
    parameter_constraints.add(
        ConstraintFcn.TRACK_PARAMETER, key="qdot_end", node=Node.START, min_bound=-0.5, max_bound=0.5
    )

    ocp = _prepare_variational_ocp(
        parameter_objectives=parameter_objectives, parameter_constraints=parameter_constraints
    )

    assert [(j.name, j.extra_parameters["key"]) for j in ocp.J] == [("MINIMIZE_PARAMETER", "qdot_start")]
    assert [(g.name, g.extra_parameters["key"]) for g in ocp.nlp[0].g if g] == [("TRACK_PARAMETER", "qdot_end")]
