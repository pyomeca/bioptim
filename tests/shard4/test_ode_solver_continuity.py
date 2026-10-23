"""
Tests of the continuity of the states declared by the ode solvers.
"""

import pytest
from bioptim import (
    ControlType,
    DynamicsOptions,
    OdeSolver,
    OptimalControlProgram,
    TorqueBiorbdModel,
)

from ..utils import TestUtils


def _pendulum_ocp(ode_solver, control_type=ControlType.CONSTANT, **dynamics_kwargs) -> OptimalControlProgram:
    return OptimalControlProgram(
        TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"),
        n_shooting=4,
        phase_time=1,
        dynamics=DynamicsOptions(ode_solver=ode_solver, **dynamics_kwargs),
        control_type=control_type,
    )


def _names(penalties) -> list[str]:
    return [penalty.name for penalty in penalties if penalty]


@pytest.mark.parametrize(
    "ode_solver, control_type",
    [
        (OdeSolver.RK4(), ControlType.CONSTANT),
        (OdeSolver.TRAPEZOIDAL(), ControlType.LINEAR_CONTINUOUS),
        (OdeSolver.COLLOCATION(), ControlType.CONSTANT),
    ],
)
def test_state_continuity_is_a_single_internal_constraint(ode_solver, control_type):
    nlp = _pendulum_ocp(ode_solver, control_type).nlp[0]

    assert _names(nlp.g_internal) == ["STATE_CONTINUITY"]
    assert _names(nlp.J_internal) == []


def test_collocation_with_duplicated_starting_point_adds_the_helper_constraint():
    nlp = _pendulum_ocp(OdeSolver.COLLOCATION(duplicate_starting_point=True)).nlp[0]

    assert _names(nlp.g_internal) == ["STATE_CONTINUITY", "FIRST_COLLOCATION_HELPER_EQUALS_STATE"]


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4(), OdeSolver.COLLOCATION(duplicate_starting_point=True)])
def test_state_continuity_weight_declares_an_objective_instead(ode_solver):
    nlp = _pendulum_ocp(ode_solver, state_continuity_weight=1000.0).nlp[0]

    assert _names(nlp.g_internal) == []
    assert _names(nlp.J_internal) == ["STATE_CONTINUITY"]


def test_skip_continuity_declares_nothing():
    nlp = _pendulum_ocp(OdeSolver.RK4(), skip_continuity=True).nlp[0]

    assert _names(nlp.g_internal) == []
    assert _names(nlp.J_internal) == []
