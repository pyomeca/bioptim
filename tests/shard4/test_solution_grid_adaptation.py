import numpy as np
import pytest

from bioptim import InitialGuessList, InterpolationType, adapt_solution_to_initial_guesses


class _FakeSolution:
    def decision_states(self, **_):
        return {"q": np.array([[0.0, 1.0, 2.0]])}

    def decision_controls(self, **_):
        return {"tau": np.array([[0.0, 2.0, np.nan]])}

    def decision_parameters(self):
        return {"mass": np.array([3.0])}

    def decision_algebraic_states(self, **_):
        return {"contact": np.array([[1.0, 3.0, 5.0]])}


def test_adapt_solution_between_different_grids():
    x_init = InitialGuessList()
    x_init.add("q", np.zeros((1, 5)), interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuessList()
    u_init.add("tau", np.zeros((1, 4)), interpolation=InterpolationType.EACH_FRAME)
    p_init = InitialGuessList()
    p_init.add("mass", [0.0], interpolation=InterpolationType.CONSTANT)
    a_init = InitialGuessList()
    a_init.add("contact", np.zeros((1, 7)), interpolation=InterpolationType.ALL_POINTS)

    states, controls, parameters, algebraic_states = adapt_solution_to_initial_guesses(
        _FakeSolution(), x_init, u_init, p_init, a_init
    )

    np.testing.assert_allclose(states[0]["q"].init, [[0.0, 0.5, 1.0, 1.5, 2.0]])
    np.testing.assert_allclose(controls[0]["tau"].init, [[0.0, 2 / 3, 4 / 3, 2.0]])
    np.testing.assert_allclose(parameters[0]["mass"].init, [[3.0]])
    np.testing.assert_allclose(algebraic_states[0]["contact"].init, [[1, 5 / 3, 7 / 3, 3, 11 / 3, 13 / 3, 5]])


def test_adapt_solution_preserves_control_grid_semantics():
    x_init = InitialGuessList()
    x_init.add("q", np.zeros((1, 3)), interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuessList()
    u_init.add("tau", np.zeros((1, 2)), interpolation=InterpolationType.LINEAR)

    _, controls, _, _ = _FakeSolutionAdapter().to_initial_guesses(x_init, u_init)

    assert controls[0]["tau"].type == InterpolationType.LINEAR
    np.testing.assert_allclose(controls[0]["tau"].init, [[0.0, 2.0]])


@pytest.mark.parametrize(
    ("interpolation", "initial_guess", "extra_arguments"),
    (
        (InterpolationType.SPLINE, [[0.0, 0.0]], {"t": [0.0, 1.0]}),
        (InterpolationType.CUSTOM, lambda _index: np.array([0.0]), {}),
    ),
)
def test_adapt_solution_rejects_interpolations_that_cannot_be_inferred(interpolation, initial_guess, extra_arguments):
    x_init = InitialGuessList()
    x_init.add("q", initial_guess, interpolation=interpolation, **extra_arguments)
    u_init = InitialGuessList()
    u_init.add("tau", np.zeros((1, 2)), interpolation=InterpolationType.EACH_FRAME)

    with pytest.raises(NotImplementedError, match="cannot be linearly resampled"):
        adapt_solution_to_initial_guesses(_FakeSolution(), x_init, u_init)


class _FakeSolutionAdapter(_FakeSolution):
    to_initial_guesses = __import__("bioptim").Solution.to_initial_guesses
