import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from casadi import MX, Function

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    BiMapping,
    Mapping,
    OdeSolver,
    BoundsList,
    InitialGuessList,
    Shooting,
    Solver,
    SolutionIntegrator,
    Solution,
    SolutionMerge,
    OptimizationVariableList,
)


class TestUtils:
    @staticmethod
    def bioptim_folder() -> str:
        return str(Path(__file__).parent / "../bioptim")

    @staticmethod
    def load_module(path: str) -> Any:
        module_name = path.split("/")[-1].split(".")[0]
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def deep_assert(first_elem, second_elem):
        if isinstance(first_elem, dict):
            for key in first_elem:
                TestUtils.deep_assert(first_elem[key], second_elem[key])
        elif isinstance(first_elem, (list, tuple)):
            for i in range(len(first_elem)):
                TestUtils.deep_assert(first_elem[i], second_elem[i])
        elif isinstance(
            first_elem, (OptimalControlProgram, BoundsList, InitialGuessList, BiMapping, Mapping, OdeSolver)
        ):
            for key in dir(first_elem):
                if type(first_elem) in (Mapping, BiMapping) and key in ("param_when_copying", "shape", "value"):
                    # These are designed to fail, so don't test them
                    continue
                TestUtils.deep_assert(getattr(first_elem, key), getattr(second_elem, key))
        else:
            if not callable(first_elem) and not isinstance(first_elem, (MX, BiorbdModel)):
                try:
                    elem_loaded = np.asarray(first_elem, dtype=float)
                    elem_original = np.array(second_elem, dtype=float)
                    npt.assert_almost_equal(elem_original, elem_loaded)
                except (ValueError, Exception):
                    pass

    @staticmethod
    def assert_warm_start(ocp, sol, state_decimal=2, control_decimal=2, param_decimal=2):
        ocp.set_warm_start(sol)

        solver = Solver.IPOPT()
        solver.set_maximum_iterations(0)
        solver.set_initialization_options(1e-10)

        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

        sol_warm_start = ocp.solve(solver)
        warm_start_states = sol_warm_start.decision_states(to_merge=SolutionMerge.NODES)
        warm_start_controls = sol_warm_start.decision_controls(to_merge=SolutionMerge.NODES)
        if ocp.n_phases > 1:
            for i in range(ocp.n_phases):
                for key in states[i]:
                    npt.assert_almost_equal(warm_start_states[i][key], states[i][key], decimal=state_decimal)
                for key in controls[i]:
                    npt.assert_almost_equal(warm_start_controls[i][key], controls[i][key], decimal=control_decimal)
        else:
            for key in states:
                npt.assert_almost_equal(warm_start_states[key], states[key], decimal=state_decimal)
            for key in controls:
                npt.assert_almost_equal(warm_start_controls[key], controls[key], decimal=control_decimal)

        for key in sol_warm_start.parameters.keys():
            npt.assert_almost_equal(sol_warm_start.parameters[key], sol.parameters[key], decimal=param_decimal)

    @staticmethod
    def simulate(sol: Solution, decimal_value=7):
        if sum([nlp.ode_solver.is_direct_collocation for nlp in sol.ocp.nlp]):
            with pytest.raises(
                ValueError,
                match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
            ):
                sol.integrate(
                    shooting_type=Shooting.SINGLE,
                    integrator=SolutionIntegrator.OCP,
                )
            return

        if sum([isinstance(nlp.ode_solver, OdeSolver.TRAPEZOIDAL) for nlp in sol.ocp.nlp]):
            with pytest.raises(
                ValueError,
                match="When the ode_solver of the Optimal Control Problem is OdeSolver.TRAPEZOIDAL, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
            ):
                sol.integrate(
                    shooting_type=Shooting.SINGLE,
                    integrator=SolutionIntegrator.OCP,
                )
            return

        sol_single = sol.integrate(
            shooting_type=Shooting.SINGLE,
            integrator=SolutionIntegrator.OCP,
            to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES],
        )

        # Evaluate the final error of the single shooting integration versus the final node
        sol_merged = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
        for key in sol_merged.keys():
            npt.assert_almost_equal(
                sol_merged[key][:, -1],
                sol_single[key][:, -1],
                decimal=decimal_value,
            )

    @staticmethod
    def mx_to_array(mx: MX, squeeze: bool = True, expand: bool = True) -> np.ndarray:
        """
        Convert a casadi MX to a numpy array if it is only numeric values
        """
        val = Function(
            "f",
            [],
            [mx],
            [],
            ["f"],
        )
        if expand:
            val = val.expand()
        val = val()["f"].toarray()

        return val.squeeze() if squeeze else val

    @staticmethod
    def to_array(value: MX | np.ndarray):
        if isinstance(value, MX):
            return TestUtils.mx_to_array(value)
        else:
            return value

    @staticmethod
    def mx_assert_equal(mx: MX, expected: Any, decimal: int = 6, squeeze: bool = True, expand: bool = True):
        """
        Assert that a casadi MX is equal to a numpy array if it is only numeric values
        """
        if isinstance(expected, MX):
            expected = TestUtils.mx_to_array(mx, squeeze=squeeze, expand=expand)

        npt.assert_almost_equal(TestUtils.mx_to_array(mx, squeeze=squeeze, expand=expand), expected, decimal=decimal)

    @staticmethod
    def assert_equal(
        value: MX | np.ndarray, expected: Any, decimal: int = 6, squeeze: bool = True, expand: bool = True
    ):
        """
        Assert that a casadi MX or numpy array is equal to a numpy array if it is only numeric values
        """
        if isinstance(value, MX):
            TestUtils.mx_assert_equal(value, expected, decimal=decimal, squeeze=squeeze, expand=expand)
        else:
            npt.assert_almost_equal(value, expected, decimal=decimal)

    @staticmethod
    def initialize_numerical_timeseries(nlp, dynamics):
        numerical_timeseries = OptimizationVariableList(nlp.cx, dynamics.phase_dynamics)
        if dynamics.numerical_data_timeseries is not None:
            for key in dynamics.numerical_data_timeseries.keys():
                variable_shape = dynamics.numerical_data_timeseries[key].shape
                for i_component in range(variable_shape[1] if len(variable_shape) > 1 else 1):
                    cx = nlp.cx.sym(
                        f"{key}_phase{nlp.phase_idx}_{i_component}_cx",
                        variable_shape[0],
                    )

                    numerical_timeseries.append(
                        name=f"{key}_{i_component}",
                        cx=[cx, cx, cx],
                        bimapping=BiMapping(
                            Mapping(list(range(variable_shape[0]))), Mapping(list(range(variable_shape[0])))
                        ),
                    )
        return numerical_timeseries
