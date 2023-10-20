import pytest

from bioptim import PhaseDynamics, RigidBodyDynamics, SoftContactDynamics
from bioptim.dynamics.configure_problem import (
    _check_external_forces_format,
    _check_external_forces_and_phase_dynamics,
    _check_soft_contacts_dynamics,
    _check_contacts_in_biorbd_model,
)


class MockData:
    pass


def test_check_external_forces_format_valid():
    _check_external_forces_format([MockData()], 1, 0)


def test_check_external_forces_format_invalid():
    with pytest.raises(RuntimeError):
        _check_external_forces_format([MockData(), MockData()], 1, 0)


# Tests for _check_external_forces_and_phase_dynamics
def test_check_external_forces_and_phase_dynamics_valid():
    _check_external_forces_and_phase_dynamics([MockData()], PhaseDynamics.ONE_PER_NODE, 0)


def test_check_external_forces_and_phase_dynamics_invalid():
    with pytest.raises(RuntimeError):
        _check_external_forces_and_phase_dynamics([MockData()], "InvalidPhaseDynamics", 0)


# Tests for _check_soft_contacts_dynamics
def test_check_soft_contacts_dynamics_valid_ode():
    _check_soft_contacts_dynamics("NotDAEInverseDynamics", SoftContactDynamics.ODE, 1, 0)


def test_check_soft_contacts_dynamics_invalid_rigid():
    with pytest.raises(ValueError):
        _check_soft_contacts_dynamics(RigidBodyDynamics.DAE_INVERSE_DYNAMICS, SoftContactDynamics.ODE, 1, 0)


# More tests for _check_external_forces_format
def test_check_external_forces_format_none():
    _check_external_forces_format(None, 2, 0)


def test_check_external_forces_format_wrong_length():
    with pytest.raises(RuntimeError):
        _check_external_forces_format([MockData(), MockData()], 3, 0)


# More tests for _check_external_forces_and_phase_dynamics
def test_check_external_forces_and_phase_dynamics_no_external_forces():
    _check_external_forces_and_phase_dynamics(None, "InvalidPhaseDynamics", 0)


# Tests for _check_soft_contacts_dynamics
def test_check_soft_contacts_dynamics_valid_constraint():
    _check_soft_contacts_dynamics("SomeRigidBodyDynamics", SoftContactDynamics.CONSTRAINT, 1, 0)


def test_check_soft_contacts_dynamics_invalid_soft_contacts_dynamics():
    with pytest.raises(ValueError):
        _check_soft_contacts_dynamics("SomeRigidBodyDynamics", "InvalidSoftContactDynamics", 1, 0)


def test_check_soft_contacts_dynamics_no_soft_contacts():
    _check_soft_contacts_dynamics("SomeRigidBodyDynamics", SoftContactDynamics.ODE, 0, 0)


# Tests for _check_contacts_in_biorbd_model
def test_check_contacts_in_biorbd_model_with_contact():
    _check_contacts_in_biorbd_model(True, 1, 0)


def test_check_contacts_in_biorbd_model_no_contact_but_flag_true():
    with pytest.raises(ValueError):
        _check_contacts_in_biorbd_model(True, 0, 0)


def test_check_contacts_in_biorbd_model_no_contact_and_flag_false():
    _check_contacts_in_biorbd_model(False, 0, 0)
