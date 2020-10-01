import numpy as np
import os
import pickle

from casadi import MX
import biorbd

from bioptim import (
    OptimalControlProgram,
    Data,
    Bounds,
    InitialGuess,
    BidirectionalMapping,
    Mapping,
    Simulate,
    OdeSolver,
)


class TestUtils:
    @staticmethod
    def save_and_load(sol, ocp, test_solve_of_loaded=False):
        file_path = "test.bo"
        ocp.save(sol, file_path)
        ocp_load, sol_load = OptimalControlProgram.load(file_path)

        TestUtils.deep_assert(sol, sol_load)
        TestUtils.deep_assert(sol_load, sol)
        if test_solve_of_loaded:
            sol_from_load = ocp_load.solve()
            TestUtils.deep_assert(sol, sol_from_load)
            TestUtils.deep_assert(sol_from_load, sol)

        TestUtils.deep_assert(ocp_load, ocp)
        TestUtils.deep_assert(ocp, ocp_load)
        os.remove(file_path)

        file_path_bob = "test.bob"
        ocp.save_get_data(sol, file_path_bob, interpolate_nb_frames=-1, concatenate=True)
        data = Data.get_data(ocp, sol, file_path_bob, interpolate_nb_frames=-1, concatenate=True)

        with open(file_path_bob, "rb") as file:
            data_load = pickle.load(file)["data"]

        TestUtils.deep_assert(data, data_load)
        TestUtils.deep_assert(data_load, data)
        os.remove(file_path_bob)

    @staticmethod
    def deep_assert(first_elem, second_elem):
        if isinstance(first_elem, dict):
            for key in first_elem:
                TestUtils.deep_assert(first_elem[key], second_elem[key])
        elif isinstance(first_elem, (list, tuple)):
            for i in range(len(first_elem)):
                TestUtils.deep_assert(first_elem[i], second_elem[i])
        elif isinstance(
            first_elem, (OptimalControlProgram, Bounds, InitialGuess, BidirectionalMapping, Mapping, OdeSolver)
        ):
            for key in dir(first_elem):
                TestUtils.deep_assert(getattr(first_elem, key), getattr(second_elem, key))
        else:
            if not callable(first_elem) and not isinstance(first_elem, (MX, biorbd.Model)):
                try:
                    elem_loaded = np.asarray(first_elem, dtype=float)
                    elem_original = np.array(second_elem, dtype=float)
                    np.testing.assert_almost_equal(elem_original, elem_loaded)
                except (ValueError, Exception):
                    pass

    @staticmethod
    def simulate(sol, ocp, decimal_value=None):
        sol_from_solver = np.array(sol["x"]).squeeze()
        sol_simulation_from_solve = Simulate.from_solve(ocp, sol)["x"]
        sol_simulation_from_data = Simulate.from_data(ocp, Data.get_data(ocp, sol))["x"]

        if decimal_value is None:
            np.testing.assert_almost_equal(sol_from_solver, sol_simulation_from_solve)
            np.testing.assert_almost_equal(sol_from_solver, sol_simulation_from_data)
            np.testing.assert_almost_equal(sol_simulation_from_solve, sol_simulation_from_data)
        else:
            np.testing.assert_almost_equal(sol_from_solver, sol_simulation_from_solve, decimal=decimal_value)
            np.testing.assert_almost_equal(sol_from_solver, sol_simulation_from_data, decimal=decimal_value)
            np.testing.assert_almost_equal(sol_simulation_from_solve, sol_simulation_from_data, decimal=decimal_value)
