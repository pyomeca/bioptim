from types import SimpleNamespace

import numpy as np

from bioptim import CyclicVariableShift, RecedingHorizonFailurePolicy
from bioptim.misc.enums import SolverType
from bioptim.optimization.optimal_control_program import OptimalControlProgram
from bioptim.optimization.receding_horizon_optimization import (
    CyclicRecedingHorizonOptimization,
    RecedingHorizonOptimization,
)


def test_window_hooks_can_stop_after_collecting_diagnostics(monkeypatch):
    solved = SimpleNamespace(status=1, vector=np.ones((1, 1)), real_time_to_optimize=0.1)
    monkeypatch.setattr(OptimalControlProgram, "solve", lambda *args, **kwargs: solved)
    events = []

    rhe = object.__new__(RecedingHorizonOptimization)
    rhe.nlp = [SimpleNamespace(x_bounds={})]
    solver = SimpleNamespace(type=SolverType.IPOPT, online_optim=False)
    returned = rhe.solve(
        update_function=lambda *args: True,
        solver=solver,
        failure_policy=RecedingHorizonFailurePolicy.CONTINUE_DIAGNOSTIC,
        before_window_solve=lambda _, index, __: events.append(("before", index)),
        after_window_solve=lambda _, index, result: events.append(("after", index, result.solver_succeeded)) or False,
    )

    assert returned is solved
    assert events == [("before", 0), ("after", 0, False)]


def test_cyclic_shift_uses_configured_period_and_state_index():
    cyclic = object.__new__(CyclicRecedingHorizonOptimization)
    cyclic.nlp = [SimpleNamespace(states={"q": [object(), object()], "qdot": [object(), object()]})]
    cyclic._initialize_cyclic_variable_shifts(
        {"variable_shifts": [CyclicVariableShift(key="q", indices=1, period=3.5, turns=-2)]}
    )

    q = cyclic._apply_cyclic_variable_shifts("q", np.array([[1.0, 2.0], [10.0, 20.0]]))
    qdot = cyclic._apply_cyclic_variable_shifts("qdot", np.array([[1.0], [2.0]]))

    np.testing.assert_allclose(q, [[1.0, 2.0], [3.0, 13.0]])
    np.testing.assert_allclose(qdot, [[1.0], [2.0]])
