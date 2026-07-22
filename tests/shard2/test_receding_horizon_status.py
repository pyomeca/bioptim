from types import SimpleNamespace

import numpy as np
import pytest

from bioptim import RecedingHorizonFailurePolicy
from bioptim.misc.enums import SolverType
from bioptim.optimization.optimal_control_program import OptimalControlProgram
from bioptim.optimization.receding_horizon_optimization import (
    RecedingHorizonOptimization,
)


def test_failed_window_is_reported_and_not_exported(monkeypatch):
    failed_solution = SimpleNamespace(
        status=3,
        vector=np.ones((1, 1)),
        real_time_to_optimize=0.1,
    )
    monkeypatch.setattr(OptimalControlProgram, "solve", lambda *args, **kwargs: failed_solution)

    rhe = object.__new__(RecedingHorizonOptimization)
    rhe.nlp = [SimpleNamespace()]
    solver = SimpleNamespace(type=SolverType.IPOPT, online_optim=False)
    solution = rhe.solve(
        update_function=lambda *args: True,
        solver=solver,
        failure_policy=RecedingHorizonFailurePolicy.STOP,
    )

    assert solution is failed_solution
    assert solution.status == 3
    assert len(solution.window_results) == 1
    assert not solution.window_results[0].solver_succeeded
    assert solution.window_results[0].trajectory_available
    assert not solution.window_results[0].physically_acceptable
    assert not solution.window_results[0].exported


class _FakeAcadosSolver:
    def get_stats(self, name):
        if name == "unavailable":
            raise RuntimeError
        return {
            "res_stat": 1e-8,
            "sqp_iter": 4,
            "time_tot": 0.02,
            "statistics": np.ones((2, 2)),
        }[name]

    def get(self, node, field):
        return np.array([node]) if field == "x" else np.array([-node])


def test_acados_diagnostics_and_iterates_are_public():
    pytest.importorskip("acados_template")
    from bioptim.interfaces.acados_interface import AcadosInterface

    interface = object.__new__(AcadosInterface)
    interface.status = 3
    interface.real_time_to_optimize = 0.03
    interface.ocp_solver = _FakeAcadosSolver()
    interface.acados_ocp = SimpleNamespace(dims=SimpleNamespace(N=2))

    diagnostics = interface.get_solver_diagnostics()
    iterates = interface.get_iterates()

    assert diagnostics["status"] == 3
    assert diagnostics["status_label"] == "qp_solver_failure"
    assert diagnostics["res_stat"] == 1e-8
    assert diagnostics["sqp_iter"] == 4
    assert len(iterates) == 3
    np.testing.assert_array_equal(iterates[1]["u"], [-1])
    assert iterates[-1]["u"] is None
