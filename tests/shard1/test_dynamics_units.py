import numpy as np
import pytest

from bioptim.dynamics.configure_problem import (
    _check_numerical_timeseries_format,
    _check_contacts_in_biomodel,
)
from bioptim import ContactType, BiorbdModel
from ..utils import TestUtils


class MockData:
    pass


from bioptim.examples.muscle_driven_with_contact import contact_forces_inequality_constraint_muscle as ocp_module

BIOPTIM_FOLDER = TestUtils.module_folder(ocp_module)
MODEL_RIGID_CONTACT = BiorbdModel(BIOPTIM_FOLDER + "/models/2segments_4dof_2contacts_1muscle.bioMod")
MODEL_SOFT_CONTACT = BiorbdModel(BIOPTIM_FOLDER + "/models/2segments_4dof_2soft_contacts_1muscle.bioMod")
from bioptim.examples.getting_started import pendulum as ocp_module

BIOPTIM_FOLDER = TestUtils.module_folder(ocp_module)
MODEL_NO_CONTACT = BiorbdModel(BIOPTIM_FOLDER + "/models/pendulum.bioMod")


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


# Tests for _check_contacts_in_biomodel
def test_check_contacts_in_biomodel_with_rigid_contact_explicit():
    _check_contacts_in_biomodel([ContactType.RIGID_EXPLICIT], MODEL_RIGID_CONTACT, 0)


def test_check_contacts_in_biomodel_with_rigid_contact_implicit():
    _check_contacts_in_biomodel([ContactType.RIGID_IMPLICIT], MODEL_RIGID_CONTACT, 0)


def test_check_contacts_in_biomodel_with_soft_contact_explicit():
    _check_contacts_in_biomodel([ContactType.SOFT_EXPLICIT], MODEL_SOFT_CONTACT, 0)


def test_check_contacts_in_biomodel_with_soft_contact_implicit():
    _check_contacts_in_biomodel([ContactType.SOFT_IMPLICIT], MODEL_SOFT_CONTACT, 0)


def test_check_contacts_in_biomodel_no_contact_but_flag_true_rigid_explicit():
    with pytest.raises(ValueError):
        _check_contacts_in_biomodel([ContactType.RIGID_EXPLICIT], MODEL_NO_CONTACT, 0)


def test_check_contacts_in_biomodel_no_contact_but_flag_true_rigid_implicit():
    with pytest.raises(ValueError):
        _check_contacts_in_biomodel([ContactType.RIGID_IMPLICIT], MODEL_NO_CONTACT, 0)


def test_check_contacts_in_biomodel_no_contact_but_flag_true_soft_explicit():
    with pytest.raises(ValueError):
        _check_contacts_in_biomodel([ContactType.SOFT_EXPLICIT], MODEL_NO_CONTACT, 0)


def test_check_contacts_in_biomodel_no_contact_but_flag_true_soft_implicit():
    with pytest.raises(ValueError):
        _check_contacts_in_biomodel([ContactType.SOFT_IMPLICIT], MODEL_NO_CONTACT, 0)


def test_check_contacts_in_biomodel_no_contact_and_flag_false():
    _check_contacts_in_biomodel([], MODEL_NO_CONTACT, 0)


def test_check_contacts_in_biomodel_rigid_contact_and_flag_false():
    _check_contacts_in_biomodel([], MODEL_RIGID_CONTACT, 0)


def test_check_contacts_in_biomodel_soft_contact_and_flag_false():
    _check_contacts_in_biomodel([], MODEL_SOFT_CONTACT, 0)
