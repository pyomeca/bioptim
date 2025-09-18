import numpy as np
import pytest

from bioptim.dynamics.configure_problem import (
    _check_numerical_timeseries_format,
)
from bioptim import ContactType, BiorbdModel
from ..utils import TestUtils


class MockData:
    pass


BIOPTIM_FOLDER = TestUtils.bioptim_folder()
MODEL_RIGID_CONTACT = BiorbdModel(bioptim_folder + "examples/models/2segments_4dof_2contacts_1muscle.bioMod")
MODEL_SOFT_CONTACT = BiorbdModel(bioptim_folder + "examples/models/2segments_4dof_2soft_contacts_1muscle.bioMod")
from bioptim.examples.getting_started import basic_ocp as ocp_module

BIOPTIM_FOLDER = TestUtils.module_folder(ocp_module)
MODEL_NO_CONTACT = BiorbdModel(bioptim_folder + "examples/models/pendulum.bioMod")


def test_check_external_forces_format_valid():
    _check_numerical_timeseries_format(np.ones((5, 5, 6)), 5, 0)


def test_check_external_forces_format_invalid():
    with pytest.raises(
        RuntimeError,
        match="Phase 0 has numerical_data_timeseries of type <class 'list'> but it should be of type np.ndarray",
    ):
        _check_numerical_timeseries_format([0, 0, 0, 0, 0, 0], 5, 0)


def test_check_external_forces_format_invalid():
    with pytest.raises(RuntimeError):
        _check_numerical_timeseries_format([MockData(), MockData()], 1, 0)


# More tests for _check_external_forces_format
def test_check_external_forces_format_none():
    with pytest.raises(
        RuntimeError,
        match="Phase 0 has numerical_data_timeseries of type <class 'NoneType'> but it should be of type np.ndarray",
    ):
        _check_numerical_timeseries_format(None, 5, 0)


def test_check_external_forces_format_wrong_length():
    with pytest.raises(RuntimeError):
        _check_numerical_timeseries_format([MockData(), MockData()], 3, 0)
