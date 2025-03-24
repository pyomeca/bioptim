import numpy as np
import pytest

from bioptim.dynamics.configure_problem import (
    _check_numerical_timeseries_format,
    _check_contacts_in_biorbd_model,
)
from bioptim import ContactType


class MockData:
    pass

# TODO
path
MODEL_RIGID_CONTACT = BiorbdModel("")


# TODO: Verify these tests


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


# Tests for _check_contacts_in_biorbd_model
def test_check_contacts_in_biorbd_model_with_contact():
    _check_contacts_in_biorbd_model(ContactType.RIGID_EXPLICIT, 1, 0)


def test_check_contacts_in_biorbd_model_no_contact_but_flag_true():
    with pytest.raises(ValueError):
        _check_contacts_in_biorbd_model(True, 0, 0)


def test_check_contacts_in_biorbd_model_no_contact_and_flag_false():
    _check_contacts_in_biorbd_model(False, 0, 0)
