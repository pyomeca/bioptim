import pytest

from bioptim import SocpType, PhaseDynamics
from bioptim.optimization.stochastic_optimal_control_program import (
    _check_multi_threading_and_problem_type,
    _check_has_no_phase_dynamics_shared_during_the_phase,
)
from bioptim.examples.stochastic_optimal_control.models.mass_point_model import MassPointDynamicsModel

bio_model = MassPointDynamicsModel(problem_type=SocpType.COLLOCATION)


# Tests for _check_multi_threading_and_problem_type()
def test_check_multi_threading_and_problem_type_no_n_thread():
    _check_multi_threading_and_problem_type(problem_type=SocpType.COLLOCATION, bio_model=bio_model)


def test_check_multi_threading_and_problem_type_n_thread_1():
    _check_multi_threading_and_problem_type(problem_type=SocpType.COLLOCATION, bio_model=bio_model, n_threads=1)


def test_check_multi_threading_and_problem_type_n_thread_not_1():
    with pytest.raises(ValueError, match="Multi-threading is not possible yet"):
        _check_multi_threading_and_problem_type(problem_type=SocpType.COLLOCATION, bio_model=bio_model, n_threads=2)


def test_check_multi_threading_and_problem_type_not_the_same():
    with pytest.raises(
        RuntimeError,
        match="The problem type should be the same in the StochasticModel as in the StochasticOptimalControlProblem.",
    ):
        _check_multi_threading_and_problem_type(problem_type=SocpType.TRAPEZOIDAL_IMPLICIT, bio_model=bio_model)


# Tests for _check_has_no_phase_dynamics_shared_during_the_phase()
def test_check_has_no_phase_dynamics_shared_during_the_phase_shared_dynamics():
    with pytest.raises(ValueError, match="The dynamics cannot be SHARED_DURING_THE_PHASE"):
        _check_has_no_phase_dynamics_shared_during_the_phase(
            "some_type", phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE
        )


def test_check_has_no_phase_dynamics_shared_during_the_phase_not_shared_dynamics():
    _check_has_no_phase_dynamics_shared_during_the_phase("some_type", phase_dynamics=PhaseDynamics.ONE_PER_NODE)


def test_check_has_no_phase_dynamics_shared_during_the_phase_collocation():
    _check_has_no_phase_dynamics_shared_during_the_phase(SocpType.COLLOCATION)
