import importlib.util
from pathlib import Path
import os
from typing import Any
import pickle

import numpy as np
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
    def save_and_load(sol, ocp, test_solve_of_loaded=False):
        file_path = "test"
        ocp.save(sol, f"{file_path}.bo")
        ocp_load, sol_load = OptimalControlProgram.load(f"{file_path}.bo")

        TestUtils.deep_assert(sol, sol_load)
        TestUtils.deep_assert(sol_load, sol)
        if test_solve_of_loaded:
            sol_from_load = ocp_load.solve()
            TestUtils.deep_assert(sol, sol_from_load)
            TestUtils.deep_assert(sol_from_load, sol)

        TestUtils.deep_assert(ocp_load, ocp)
        TestUtils.deep_assert(ocp, ocp_load)

        ocp.save(sol, f"{file_path}_sa.bo", stand_alone=True)
        with open(f"{file_path}_sa.bo", "rb") as file:
            states, controls, parameters = pickle.load(file)
        TestUtils.deep_assert(states, sol.states)
        TestUtils.deep_assert(controls, sol.controls)
        TestUtils.deep_assert(parameters, sol.parameters)

        os.remove(f"{file_path}.bo")
        os.remove(f"{file_path}_sa.bo")

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
                    np.testing.assert_almost_equal(elem_original, elem_loaded)
                except (ValueError, Exception):
                    pass

    @staticmethod
    def assert_warm_start(ocp, sol, state_decimal=2, control_decimal=2, param_decimal=2):
        ocp.set_warm_start(sol)

        solver = Solver.IPOPT()
        solver.set_maximum_iterations(0)
        solver.set_initialization_options(1e-10)

        sol_warm_start = ocp.solve(solver)
        if ocp.n_phases > 1:
            for i in range(ocp.n_phases):
                for key in sol.states[i]:
                    np.testing.assert_almost_equal(
                        sol_warm_start.states[i][key], sol.states[i][key], decimal=state_decimal
                    )
                for key in sol.controls[i]:
                    np.testing.assert_almost_equal(
                        sol_warm_start.controls[i][key], sol.controls[i][key], decimal=control_decimal
                    )
        else:
            for key in sol.states:
                np.testing.assert_almost_equal(sol_warm_start.states[key], sol.states[key], decimal=state_decimal)
            for key in sol.controls:
                np.testing.assert_almost_equal(sol_warm_start.controls[key], sol.controls[key], decimal=control_decimal)
        for key in sol_warm_start.parameters.keys():
            np.testing.assert_almost_equal(sol_warm_start.parameters[key], sol.parameters[key], decimal=param_decimal)

    @staticmethod
    def simulate(sol, decimal_value=7):
        sol_merged = sol.merge_phases()
        if sum([nlp.ode_solver.is_direct_collocation for nlp in sol.ocp.nlp]):
            with pytest.raises(
                ValueError,
                match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
            ):
                sol.integrate(
                    merge_phases=True,
                    shooting_type=Shooting.SINGLE,
                    keep_intermediate_points=True,
                    integrator=SolutionIntegrator.OCP,
                )
            return

        sol_single = sol.integrate(
            merge_phases=True,
            shooting_type=Shooting.SINGLE,
            keep_intermediate_points=True,
            integrator=SolutionIntegrator.OCP,
        )

        # Evaluate the final error of the single shooting integration versus the finale node
        for key in sol_merged.states.keys():
            np.testing.assert_almost_equal(
                sol_merged.states[key][:, -1],
                sol_single.states[key][:, -1],
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

        np.testing.assert_almost_equal(
            TestUtils.mx_to_array(mx, squeeze=squeeze, expand=expand), expected, decimal=decimal
        )

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
            np.testing.assert_almost_equal(value, expected, decimal=decimal)
