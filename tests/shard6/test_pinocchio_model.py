"""
Test-first suite for the improvements to PinocchioModel / TorquePinocchioModel.

Part of the suite covers what the improvements brought in: the classical marker acceleration, a set_gravity that
reaches the dynamics, normalize_state_quaternions, a constructor taking an already converted CasADi model, and
copy/serialize keeping the concrete subclass. The rest locks the behavior that already worked and must not regress.
All reference values are computed directly with pinocchio inside the tests.
"""

import importlib.util
import re

import numpy as np
import numpy.testing as npt
import pytest

from bioptim import PinocchioModel, TorquePinocchioModel
from ..utils import TestUtils

HAS_PYORERUN = importlib.util.find_spec("pyorerun") is not None

import pinocchio as pin
import pinocchio.casadi as cpin

EMPTY = np.zeros((0, 1))


def test_pinocchio_model_import_is_lazy():
    expected_message = "The model should be of type 'str', 'pinocchio.Model' or 'pinocchio.casadi.Model'"

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        PinocchioModel(1)

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        PinocchioModel([])


def test_pinocchio_model_from_urdf_has_expected_basic_properties():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    assert model.nb_q > 0
    assert model.nb_qdot > 0
    assert model.nb_tau > 0
    assert model.nb_segments > 0
    assert model.nb_root == 0
    assert model.nb_quaternions >= 0

    segment_name = list(model._model.names)[1]
    assert model.segment_index(segment_name) == 1


def test_pinocchio_segment_index_raises_for_unknown_segment():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    with pytest.raises(ValueError, match="not_a_segment is not a segment name"):
        model.segment_index("not_a_segment")


def test_pinocchio_gravity_setter_updates_gravity_function_value():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    model.set_gravity(np.array([0.1, -0.2, -3.5]))
    updated_gravity = np.array(model._model.gravity.linear).squeeze()
    np.testing.assert_almost_equal(updated_gravity, np.array([0.1, -0.2, -3.5]))


def test_pinocchio_function_caching_for_center_of_mass_and_marker():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    assert len(model._cached_functions.keys()) == 0

    model.center_of_mass()(model.q, model.parameters)
    com_id = id(model._cached_functions[("center_of_mass", (), frozenset())])
    assert len(model._cached_functions.keys()) == 1

    model.center_of_mass()(model.q, model.parameters)
    assert com_id == id(model._cached_functions[("center_of_mass", (), frozenset())])
    assert len(model._cached_functions.keys()) == 1

    if model.nb_markers > 0:
        model.marker(index=0)(model.q, model.parameters)
        marker_id = id(model._cached_functions[("marker", (), frozenset({("index", 0)}))])
        assert len(model._cached_functions.keys()) == 2

        model.marker(index=0)(model.q, model.parameters)
        assert marker_id == id(model._cached_functions[("marker", (), frozenset({("index", 0)}))])
        assert len(model._cached_functions.keys()) == 2


def test_pinocchio_not_implemented_methods_raise():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    with pytest.raises(NotImplementedError, match="forward_dynamics with contact is not implemented"):
        model.forward_dynamics(with_contact=True)

    # Animation now goes through pyorerun; any other viewer is still refused.
    with pytest.raises(NotImplementedError, match="only 'pyorerun' is supported"):
        model.animate(ocp=None, solution=None, viewer="bioviz")


def test_pinocchio_copy_and_serialize_from_path_model():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    model_copy = model.copy()
    assert isinstance(model_copy, PinocchioModel)
    assert model_copy is not model
    assert model_copy.path == model_path

    serializer, data = model.serialize()
    assert serializer is PinocchioModel
    assert data["bio_model"] == model_path


def test_pinocchio_ranges_and_marker_index_error_paths():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    q_ranges = model.ranges_from_model("q")
    qdot_ranges = model.ranges_from_model("qdot")
    qddot_ranges = model.ranges_from_model("qddot")
    assert len(q_ranges) == model.nb_q
    assert len(qdot_ranges) == model.nb_qdot
    assert len(qddot_ranges) == model.nb_qddot

    with pytest.raises(RuntimeError, match="Wrong variable name"):
        model.ranges_from_model("invalid")

    with pytest.raises(ValueError, match="not_a_marker is not a marker name"):
        model.marker_index("not_a_marker")


def test_pinocchio_forward_and_inverse_dynamics_shapes():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    q = np.zeros((model.nb_q, 1))
    qdot = np.zeros((model.nb_qdot, 1))
    tau = np.zeros((model.nb_tau, 1))
    qddot = np.zeros((model.nb_qddot, 1))
    external_forces = np.zeros((0, 1))
    parameters = np.zeros((0, 1))

    fd = model.forward_dynamics()(q, qdot, tau, external_forces, parameters)
    id_tau = model.inverse_dynamics()(q, qdot, qddot, external_forces, parameters)

    np.testing.assert_equal(np.array(fd).shape, (model.nb_qddot, 1))
    np.testing.assert_equal(np.array(id_tau).shape, (model.nb_tau, 1))


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def _manipulator_pin_model() -> "pin.Model":
    """Deterministic 6-dof serial arm with three fixed-joint frames (these are the bioptim markers)."""
    model = pin.buildSampleModelManipulator()
    placements = [
        (3, np.array([0.1, 0.2, 0.3])),
        (6, np.array([-0.05, 0.15, 0.4])),
        (model.njoints - 1, np.array([0.0, 0.1, -0.2])),
    ]
    for i, (joint_id, translation) in enumerate(placements):
        model.addFrame(pin.Frame(f"marker_{i}", joint_id, pin.SE3(np.eye(3), translation), pin.FrameType.FIXED_JOINT))
    return model


def _spherical_pin_model() -> "pin.Model":
    """Spherical joint (nq=4, nv=3) followed by a revolute joint (nq=1, nv=1): nq != nv."""
    model = pin.Model()
    sph = model.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), "sph")
    model.appendBodyToJoint(
        sph, pin.Inertia(1.0, np.array([0.0, 0.0, -0.2]), np.diag([0.02, 0.02, 0.01])), pin.SE3.Identity()
    )
    rev = model.addJoint(sph, pin.JointModelRY(), pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.4])), "rev")
    model.appendBodyToJoint(
        rev, pin.Inertia(0.5, np.array([0.0, 0.0, -0.15]), np.diag([0.01, 0.01, 0.005])), pin.SE3.Identity()
    )
    model.addFrame(
        pin.Frame("marker_0", rev, pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.3])), pin.FrameType.FIXED_JOINT)
    )
    return model


def _freeflyer_pin_model() -> "pin.Model":
    """Free-flyer root (nq=7, nv=6) followed by a revolute joint."""
    model = pin.Model()
    root = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
    model.appendBodyToJoint(
        root, pin.Inertia(2.0, np.array([0.0, 0.0, 0.1]), np.diag([0.05, 0.06, 0.03])), pin.SE3.Identity()
    )
    rev = model.addJoint(root, pin.JointModelRX(), pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.3])), "rev")
    model.appendBodyToJoint(
        rev, pin.Inertia(1.0, np.array([0.0, 0.0, 0.2]), np.diag([0.02, 0.02, 0.01])), pin.SE3.Identity()
    )
    return model


def _random_state(model: "pin.Model", seed: int = 42):
    rng = np.random.default_rng(seed)
    q = (
        pin.randomConfiguration(model, -np.ones(model.nq) * 1.5, np.ones(model.nq) * 1.5)
        if model.nq == model.nv
        else None
    )
    if q is None:
        q = pin.neutral(model)
        q = pin.integrate(model, q, rng.uniform(-0.8, 0.8, model.nv))
    qdot = rng.uniform(-1.5, 1.5, model.nv)
    qddot = rng.uniform(-2.0, 2.0, model.nv)
    tau = rng.uniform(-3.0, 3.0, model.nv)
    return q, qdot, qddot, tau


def _col(x):
    return np.asarray(x, dtype=float).reshape(-1, 1)


def _marker_frame_ids(bio_model: PinocchioModel, pin_model: "pin.Model"):
    return [pin_model.getFrameId(name) for name in bio_model.marker_names]


@pytest.fixture
def manipulator():
    pin_model = _manipulator_pin_model()
    return PinocchioModel(pin_model), pin_model


# ----------------------------------------------------------------------------------------------------------------------
# 1. Marker acceleration convention (classical, biorbd-like)
# ----------------------------------------------------------------------------------------------------------------------
def _reference_marker_accelerations(pin_model, frame_ids, q, qdot, qddot):
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q, qdot, qddot)
    pin.updateFramePlacements(pin_model, data)
    classical = [
        pin.getFrameClassicalAcceleration(pin_model, data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear.copy()
        for f in frame_ids
    ]
    spatial = [
        pin.getFrameAcceleration(pin_model, data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear.copy()
        for f in frame_ids
    ]
    return classical, spatial


def test_pinocchio_marker_acceleration_is_classical_acceleration(manipulator):
    model, pin_model = manipulator
    q, qdot, qddot, _ = _random_state(pin_model)
    classical, spatial = _reference_marker_accelerations(pin_model, _marker_frame_ids(model, pin_model), q, qdot, qddot)

    # The test is only meaningful if the two conventions differ for this state (omega x v != 0)
    assert np.linalg.norm(classical[0] - spatial[0]) > 1e-3

    for i in range(model.nb_markers):
        value = model.marker_acceleration(i)(_col(q), _col(qdot), _col(qddot), EMPTY)
        npt.assert_almost_equal(np.array(value).squeeze(), classical[i], decimal=8)


def test_pinocchio_markers_accelerations_are_classical_accelerations(manipulator):
    model, pin_model = manipulator
    q, qdot, qddot, _ = _random_state(pin_model)
    classical, _ = _reference_marker_accelerations(pin_model, _marker_frame_ids(model, pin_model), q, qdot, qddot)

    value = np.array(model.markers_accelerations()(_col(q), _col(qdot), _col(qddot), EMPTY))
    assert value.shape == (3, model.nb_markers)
    npt.assert_almost_equal(value, np.array(classical).T, decimal=8)


def test_pinocchio_marker_acceleration_matches_classical_when_velocity_is_zero(manipulator):
    # With qdot = 0, omega x v = 0 so both conventions coincide: this must hold before and after porting.
    model, pin_model = manipulator
    q, _, qddot, _ = _random_state(pin_model)
    qdot = np.zeros(pin_model.nv)
    classical, _ = _reference_marker_accelerations(pin_model, _marker_frame_ids(model, pin_model), q, qdot, qddot)
    for i in range(model.nb_markers):
        value = model.marker_acceleration(i)(_col(q), _col(qdot), _col(qddot), EMPTY)
        npt.assert_almost_equal(np.array(value).squeeze(), classical[i], decimal=8)


# ----------------------------------------------------------------------------------------------------------------------
# 2. set_gravity must propagate to the casadi model
# ----------------------------------------------------------------------------------------------------------------------
def test_pinocchio_set_gravity_changes_forward_dynamics():
    pin_model = _manipulator_pin_model()
    model = PinocchioModel(pin_model)
    q, _, _, _ = _random_state(pin_model)
    qdot, tau = np.zeros(pin_model.nv), np.zeros(pin_model.nv)
    args = (_col(q), _col(qdot), _col(tau), EMPTY, EMPTY)

    qddot_before = np.array(model.forward_dynamics()(*args)).squeeze()
    assert np.linalg.norm(qddot_before) > 1e-3  # gravity does move the arm

    model.set_gravity(np.zeros(3))
    model._cached_functions.clear()
    qddot_after = np.array(model.forward_dynamics()(*args)).squeeze()

    npt.assert_almost_equal(qddot_after, np.zeros(pin_model.nv), decimal=8)
    assert np.linalg.norm(qddot_after - qddot_before) > 1e-3


def test_pinocchio_set_gravity_is_consistent_between_gravity_dynamics_and_nle():
    pin_model = _manipulator_pin_model()
    model = PinocchioModel(pin_model)
    q, qdot, _, tau = _random_state(pin_model)
    new_gravity = np.array([0.5, -1.0, -3.0])

    model.set_gravity(new_gravity)
    model._cached_functions.clear()

    npt.assert_almost_equal(np.array(model.gravity()(EMPTY)).squeeze(), new_gravity)

    reference_model = _manipulator_pin_model()
    reference_model.gravity.linear = new_gravity
    data = reference_model.createData()
    expected_qddot = pin.aba(reference_model, data, q, qdot, tau).copy()
    expected_nle = pin.nonLinearEffects(reference_model, data, q, qdot).copy()

    qddot = model.forward_dynamics()(_col(q), _col(qdot), _col(tau), EMPTY, EMPTY)
    nle = model.non_linear_effects()(_col(q), _col(qdot), EMPTY)
    npt.assert_almost_equal(np.array(qddot).squeeze(), expected_qddot, decimal=8)
    npt.assert_almost_equal(np.array(nle).squeeze(), expected_nle, decimal=8)


# ----------------------------------------------------------------------------------------------------------------------
# 3. normalize_state_quaternions
# ----------------------------------------------------------------------------------------------------------------------
def test_pinocchio_normalize_state_quaternions_exists():
    assert hasattr(PinocchioModel, "normalize_state_quaternions")


def test_pinocchio_normalize_state_quaternions_spherical_joint():
    pin_model = _spherical_pin_model()
    model = PinocchioModel(pin_model)
    assert (model.nb_q, model.nb_qdot) == (5, 4)

    q = np.array([0.3, -0.4, 0.5, 1.7, 0.6])  # un-normalized quaternion (x, y, z, w) + revolute angle
    value = np.array(model.normalize_state_quaternions()(_col(q))).squeeze()

    expected = q.copy()
    expected[:4] = q[:4] / np.linalg.norm(q[:4])
    npt.assert_almost_equal(value, expected, decimal=10)
    npt.assert_almost_equal(np.linalg.norm(value[:4]), 1.0)
    npt.assert_almost_equal(value[4], q[4])


def test_pinocchio_normalize_state_quaternions_freeflyer():
    pin_model = _freeflyer_pin_model()
    model = PinocchioModel(pin_model)
    assert (model.nb_q, model.nb_qdot) == (8, 7)

    q = np.array([0.1, -0.2, 0.3, 0.2, -0.5, 0.1, 2.0, 0.4])
    value = np.array(model.normalize_state_quaternions()(_col(q))).squeeze()

    expected = q.copy()
    expected[3:7] = q[3:7] / np.linalg.norm(q[3:7])
    npt.assert_almost_equal(value, expected, decimal=10)
    npt.assert_almost_equal(np.linalg.norm(value[3:7]), 1.0)


def test_pinocchio_normalize_state_quaternions_is_idempotent_on_unit_quaternion():
    pin_model = _freeflyer_pin_model()
    model = PinocchioModel(pin_model)
    q = pin.integrate(pin_model, pin.neutral(pin_model), np.array([0.1, 0.2, 0.3, 0.2, -0.1, 0.4, 0.5]))
    value = np.array(model.normalize_state_quaternions()(_col(q))).squeeze()
    npt.assert_almost_equal(value, q, decimal=10)


def test_pinocchio_normalize_state_quaternions_is_passthrough_without_quaternion(manipulator):
    model, pin_model = manipulator
    assert model.nb_q == model.nb_qdot
    q, _, _, _ = _random_state(pin_model)
    value = np.array(model.normalize_state_quaternions()(_col(q))).squeeze()
    npt.assert_almost_equal(value, q, decimal=12)


# ----------------------------------------------------------------------------------------------------------------------
# 4. Constructor accepts an already converted pinocchio.casadi Model
# ----------------------------------------------------------------------------------------------------------------------
def test_pinocchio_constructor_accepts_casadi_model():
    pin_model = _manipulator_pin_model()
    reference = PinocchioModel(pin_model)
    from_casadi = PinocchioModel(cpin.Model(pin_model))

    assert from_casadi.nb_q == reference.nb_q
    assert from_casadi.nb_qdot == reference.nb_qdot

    q, qdot, _, tau = _random_state(pin_model)
    args = (_col(q), _col(qdot), _col(tau), EMPTY, EMPTY)
    npt.assert_almost_equal(
        np.array(from_casadi.forward_dynamics()(*args)),
        np.array(reference.forward_dynamics()(*args)),
        decimal=10,
    )


def test_pinocchio_constructor_still_rejects_invalid_types():
    with pytest.raises(ValueError):
        PinocchioModel(1)
    with pytest.raises(ValueError):
        PinocchioModel(None)


# ----------------------------------------------------------------------------------------------------------------------
# 5. copy() / serialize() must preserve the subclass
# ----------------------------------------------------------------------------------------------------------------------
def test_pinocchio_torque_model_serialize_keeps_subclass():
    model = TorquePinocchioModel(_manipulator_pin_model())
    serializer, data = model.serialize()
    assert serializer is TorquePinocchioModel
    assert "bio_model" in data

    rebuilt = serializer(**data)
    assert isinstance(rebuilt, TorquePinocchioModel)
    assert rebuilt.nb_q == model.nb_q


def test_pinocchio_torque_model_copy_keeps_subclass_from_pin_model():
    pin_model = _manipulator_pin_model()
    model = TorquePinocchioModel(pin_model)
    model_copy = model.copy()

    assert type(model_copy) is TorquePinocchioModel
    assert model_copy is not model
    assert model_copy.nb_q == model.nb_q
    assert model_copy.nb_markers == model.nb_markers
    assert model_copy.marker_names == model.marker_names


def test_pinocchio_torque_model_copy_and_serialize_keep_subclass_from_path():
    from ..utils import TestUtils

    model_path = TestUtils.bioptim_folder() + "/examples/models/pendulum.urdf"
    model = TorquePinocchioModel(model_path)

    model_copy = model.copy()
    assert type(model_copy) is TorquePinocchioModel
    assert model_copy.path == model_path

    serializer, data = model.serialize()
    assert serializer is TorquePinocchioModel
    assert data["bio_model"] == model_path


def test_pinocchio_base_model_copy_and_serialize_keep_base_class():
    # Non-regression: the base class must keep being reported as PinocchioModel.
    model = PinocchioModel(_manipulator_pin_model())
    assert type(model.copy()) is PinocchioModel
    assert model.serialize()[0] is PinocchioModel


# ----------------------------------------------------------------------------------------------------------------------
# 6. Non-regression: behaviors identical to the reference implementation, locked against raw pinocchio
# ----------------------------------------------------------------------------------------------------------------------
def test_pinocchio_forward_dynamics_matches_aba(manipulator):
    model, pin_model = manipulator
    q, qdot, _, tau = _random_state(pin_model)
    expected = pin.aba(pin_model, pin_model.createData(), q, qdot, tau)
    value = model.forward_dynamics()(_col(q), _col(qdot), _col(tau), EMPTY, EMPTY)
    npt.assert_almost_equal(np.array(value).squeeze(), expected, decimal=8)


def test_pinocchio_inverse_dynamics_matches_rnea(manipulator):
    model, pin_model = manipulator
    q, qdot, qddot, _ = _random_state(pin_model)
    expected = pin.rnea(pin_model, pin_model.createData(), q, qdot, qddot)
    value = model.inverse_dynamics()(_col(q), _col(qdot), _col(qddot), EMPTY, EMPTY)
    npt.assert_almost_equal(np.array(value).squeeze(), expected, decimal=8)


def test_pinocchio_forward_and_inverse_dynamics_are_consistent(manipulator):
    model, pin_model = manipulator
    q, qdot, _, tau = _random_state(pin_model)
    qddot = model.forward_dynamics()(_col(q), _col(qdot), _col(tau), EMPTY, EMPTY)
    tau_back = model.inverse_dynamics()(_col(q), _col(qdot), qddot, EMPTY, EMPTY)
    npt.assert_almost_equal(np.array(tau_back).squeeze(), tau, decimal=8)


def test_pinocchio_mass_matrix_matches_crba(manipulator):
    model, pin_model = manipulator
    q, _, _, _ = _random_state(pin_model)
    expected = pin.crba(pin_model, pin_model.createData(), q)
    expected = np.triu(expected) + np.triu(expected, 1).T  # crba only fills the upper triangle
    value = np.array(model.mass_matrix()(_col(q), EMPTY))
    npt.assert_almost_equal(value, expected, decimal=8)


def test_pinocchio_non_linear_effects_matches_pinocchio(manipulator):
    model, pin_model = manipulator
    q, qdot, _, _ = _random_state(pin_model)
    expected = pin.nonLinearEffects(pin_model, pin_model.createData(), q, qdot)
    value = model.non_linear_effects()(_col(q), _col(qdot), EMPTY)
    npt.assert_almost_equal(np.array(value).squeeze(), expected, decimal=8)


def test_pinocchio_center_of_mass_and_derivatives(manipulator):
    model, pin_model = manipulator
    q, qdot, qddot, _ = _random_state(pin_model)
    data = pin_model.createData()

    com = pin.centerOfMass(pin_model, data, q).copy()
    npt.assert_almost_equal(np.array(model.center_of_mass()(_col(q), EMPTY)).squeeze(), com, decimal=8)

    pin.centerOfMass(pin_model, data, q, qdot)
    com_velocity = data.vcom[0].copy()
    npt.assert_almost_equal(
        np.array(model.center_of_mass_velocity()(_col(q), _col(qdot), EMPTY)).squeeze(), com_velocity, decimal=8
    )

    pin.centerOfMass(pin_model, data, q, qdot, qddot)
    com_acceleration = data.acom[0].copy()
    npt.assert_almost_equal(
        np.array(model.center_of_mass_acceleration()(_col(q), _col(qdot), _col(qddot), EMPTY)).squeeze(),
        com_acceleration,
        decimal=8,
    )


def test_pinocchio_markers_positions_match_frame_placements(manipulator):
    model, pin_model = manipulator
    assert model.nb_markers == 3
    q, _, _, _ = _random_state(pin_model)
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q)
    pin.updateFramePlacements(pin_model, data)
    expected = np.array([data.oMf[f].translation for f in _marker_frame_ids(model, pin_model)]).T

    markers = np.array(model.markers()(_col(q), EMPTY))
    npt.assert_almost_equal(markers, expected, decimal=8)
    for i in range(model.nb_markers):
        npt.assert_almost_equal(np.array(model.marker(i)(_col(q), EMPTY)).squeeze(), expected[:, i], decimal=8)


def test_pinocchio_marker_in_reference_segment_frame(manipulator):
    model, pin_model = manipulator
    q, _, _, _ = _random_state(pin_model)
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q)
    pin.updateFramePlacements(pin_model, data)
    frame_ids = _marker_frame_ids(model, pin_model)
    reference_frame = pin_model.getFrameId("upperarm_body")
    expected = data.oMf[reference_frame].actInv(pin.SE3(np.eye(3), data.oMf[frame_ids[2]].translation)).translation

    value = model.marker(2, reference_segment_index=reference_frame)(_col(q), EMPTY)
    npt.assert_almost_equal(np.array(value).squeeze(), expected, decimal=8)


def test_pinocchio_marker_velocity_matches_frame_velocity(manipulator):
    model, pin_model = manipulator
    q, qdot, _, _ = _random_state(pin_model)
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q, qdot)
    pin.updateFramePlacements(pin_model, data)
    expected = [
        pin.getFrameVelocity(pin_model, data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear.copy()
        for f in _marker_frame_ids(model, pin_model)
    ]
    for i in range(model.nb_markers):
        value = model.marker_velocity(i)(_col(q), _col(qdot), EMPTY)
        npt.assert_almost_equal(np.array(value).squeeze(), expected[i], decimal=8)

    values = np.array(model.markers_velocities()(_col(q), _col(qdot), EMPTY))
    npt.assert_almost_equal(values, np.array(expected).T, decimal=8)


def test_pinocchio_angular_momentum_matches_centroidal_momentum(manipulator):
    model, pin_model = manipulator
    q, qdot, _, _ = _random_state(pin_model)
    data = pin_model.createData()
    pin.computeCentroidalMomentum(pin_model, data, q, qdot)
    expected = data.hg.angular.copy()
    value = model.angular_momentum()(_col(q), _col(qdot), EMPTY)
    npt.assert_almost_equal(np.array(value).squeeze(), expected, decimal=8)


def test_pinocchio_bounds_from_ranges_q_match_position_limits(manipulator):
    model, pin_model = manipulator
    bounds = model.bounds_from_ranges("q")
    # Bounds are (nb_q, 3): one column per node phase (first, intermediate, last)
    assert np.array(bounds.min).shape == (pin_model.nq, 3)
    for column in range(3):
        npt.assert_almost_equal(np.array(bounds.min)[:, column], pin_model.lowerPositionLimit)
        npt.assert_almost_equal(np.array(bounds.max)[:, column], pin_model.upperPositionLimit)


# ----------------------------------------------------------------------------------------------------------------------
# A. Markers on OP_FRAME frames (OpenSim conversions emit markers as operational points)
# ----------------------------------------------------------------------------------------------------------------------
def _manipulator_with_op_frame_only() -> "pin.Model":
    model = pin.buildSampleModelManipulator()
    model.addFrame(pin.Frame("mk", 1, 0, pin.SE3(np.eye(3), np.array([0.1, -0.2, 0.3])), pin.FrameType.OP_FRAME))
    return model


def _manipulator_with_mixed_markers() -> "pin.Model":
    """FIXED_JOINT, OP_FRAME, FIXED_JOINT, in that frame order."""
    model = pin.buildSampleModelManipulator()
    model.addFrame(pin.Frame("fixed_a", 2, pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.2])), pin.FrameType.FIXED_JOINT))
    model.addFrame(pin.Frame("op_b", 3, 0, pin.SE3(np.eye(3), np.array([0.0, 0.3, 0.1])), pin.FrameType.OP_FRAME))
    model.addFrame(pin.Frame("fixed_c", 5, pin.SE3(np.eye(3), np.array([-0.1, 0.1, 0.0])), pin.FrameType.FIXED_JOINT))
    return model


def _frame_translations(pin_model, q):
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q)
    pin.updateFramePlacements(pin_model, data)
    return {f.name: data.oMf[i].translation.copy() for i, f in enumerate(pin_model.frames)}


def test_pinocchio_op_frame_is_exposed_as_marker():
    pin_model = _manipulator_with_op_frame_only()
    model = PinocchioModel(pin_model)
    assert model.marker_names == ("mk",)
    assert model.nb_markers == 1

    q, _, _, _ = _random_state(pin_model)
    expected = _frame_translations(pin_model, q)["mk"]
    npt.assert_almost_equal(np.array(model.marker(0)(_col(q), EMPTY)).squeeze(), expected, decimal=8)
    npt.assert_almost_equal(np.array(model.markers()(_col(q), EMPTY)), expected.reshape(3, 1), decimal=8)


def test_pinocchio_mixed_fixed_joint_and_op_frame_markers_keep_frame_order():
    pin_model = _manipulator_with_mixed_markers()
    model = PinocchioModel(pin_model)
    assert model.marker_names == ("fixed_a", "op_b", "fixed_c")
    assert model.nb_markers == 3

    q, _, _, _ = _random_state(pin_model)
    translations = _frame_translations(pin_model, q)
    expected = np.array([translations[name] for name in model.marker_names]).T
    npt.assert_almost_equal(np.array(model.markers()(_col(q), EMPTY)), expected, decimal=8)
    for i in range(3):
        npt.assert_almost_equal(np.array(model.marker(i)(_col(q), EMPTY)).squeeze(), expected[:, i], decimal=8)


def test_pinocchio_model_without_op_frame_has_same_markers_as_before():
    # The OP_FRAME support is additive: only FIXED_JOINT frames are markers when there is no OP_FRAME.
    pin_model = _manipulator_pin_model()
    model = PinocchioModel(pin_model)
    expected = tuple(f.name for f in pin_model.frames[1:] if f.type == pin.FrameType.FIXED_JOINT)
    assert expected == ("marker_0", "marker_1", "marker_2")
    assert model.marker_names == expected
    assert not any(f.type == pin.FrameType.OP_FRAME for f in pin_model.frames)

    # JOINT and BODY frames never become markers
    base = PinocchioModel(pin.buildSampleModelManipulator())
    assert base.nb_markers == 0


def test_pinocchio_marker_index_and_marker_frame_index_are_consistent_for_op_frame():
    pin_model = _manipulator_with_mixed_markers()
    model = PinocchioModel(pin_model)
    index = model.marker_index("op_b")
    assert index == 1
    frame_id = model._marker_frame_index(index)
    assert frame_id == pin_model.getFrameId("op_b")
    assert pin_model.frames[frame_id].type == pin.FrameType.OP_FRAME
    for i, name in enumerate(model.marker_names):
        assert model.marker_index(name) == i
        assert pin_model.frames[model._marker_frame_index(i)].name == name


def test_pinocchio_op_frame_marker_velocity_matches_frame_velocity():
    pin_model = _manipulator_with_op_frame_only()
    model = PinocchioModel(pin_model)
    q, qdot, _, _ = _random_state(pin_model)
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q, qdot)
    pin.updateFramePlacements(pin_model, data)
    expected = pin.getFrameVelocity(
        pin_model, data, pin_model.getFrameId("mk"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    ).linear
    value = model.marker_velocity(0)(_col(q), _col(qdot), EMPTY)
    npt.assert_almost_equal(np.array(value).squeeze(), expected, decimal=8)


# ----------------------------------------------------------------------------------------------------------------------
# B. pyorerun integration (never opens a viewer here)
# ----------------------------------------------------------------------------------------------------------------------
def test_pinocchio_pyorerun_marker_names_is_list_of_marker_names():
    model = PinocchioModel(_manipulator_with_mixed_markers())
    names = model.pyorerun_marker_names
    assert isinstance(names, list)
    assert names == list(model.marker_names)
    assert names == ["fixed_a", "op_b", "fixed_c"]


def test_pinocchio_pyorerun_marker_names_empty_without_markers():
    assert PinocchioModel(pin.buildSampleModelManipulator()).pyorerun_marker_names == []


@pytest.mark.skipif(not HAS_PYORERUN, reason="pyorerun is not installed")
def test_pinocchio_to_pyorerun_model_from_pin_model():
    import pyorerun

    model = PinocchioModel(_manipulator_pin_model())
    pyorerun_model = model.to_pyorerun_model()
    assert isinstance(pyorerun_model, pyorerun.PinocchioModel)


@pytest.mark.skipif(not HAS_PYORERUN, reason="pyorerun is not installed")
def test_pinocchio_to_pyorerun_model_from_casadi_model():
    # This path goes through pin.Model(cpin_model) internally, which is the fragile one.
    import pyorerun

    pin_model = _manipulator_pin_model()
    model = PinocchioModel(cpin.Model(pin_model))
    pyorerun_model = model.to_pyorerun_model()
    assert isinstance(pyorerun_model, pyorerun.PinocchioModel)


def test_pinocchio_visual_and_collision_models_are_stored_as_is():
    visual, collision = object(), object()
    model = PinocchioModel(_manipulator_pin_model(), visual_model=visual, collision_model=collision)
    assert model.visual_model is visual
    assert model.collision_model is collision

    default = PinocchioModel(_manipulator_pin_model())
    assert default.visual_model is None
    assert default.collision_model is None


@pytest.mark.parametrize("model_class", [PinocchioModel, TorquePinocchioModel])
def test_pinocchio_visual_and_collision_models_survive_serialize_and_copy(model_class):
    visual, collision = object(), object()
    model = model_class(_manipulator_pin_model(), visual_model=visual, collision_model=collision)

    serializer, data = model.serialize()
    assert serializer is model_class
    assert data["visual_model"] is visual
    assert data["collision_model"] is collision
    rebuilt = serializer(**data)
    assert rebuilt.visual_model is visual and rebuilt.collision_model is collision

    model_copy = model.copy()
    assert type(model_copy) is model_class
    assert model_copy.visual_model is visual
    assert model_copy.collision_model is collision


def test_pinocchio_animate_refuses_other_viewers_and_mentions_pyorerun():
    with pytest.raises(NotImplementedError, match="pyorerun"):
        PinocchioModel.animate(None, None, viewer="bioviz")

    # Also reachable from an instance, since animate is a staticmethod
    model = PinocchioModel(_manipulator_pin_model())
    with pytest.raises(NotImplementedError, match="pyorerun"):
        model.animate(None, None, show_now=False, viewer="bioviz")


def test_pinocchio_animate_signature():
    import inspect

    parameters = inspect.signature(PinocchioModel.animate).parameters
    assert list(parameters) == ["ocp", "solution", "show_now", "show_tracked_markers", "viewer", "n_frames", "kwargs"]
    assert parameters["show_now"].default is True
    assert parameters["show_tracked_markers"].default is False
    assert parameters["viewer"].default == "pyorerun"
    assert parameters["n_frames"].default == 0
    assert isinstance(inspect.getattr_static(PinocchioModel, "animate"), staticmethod)


# ----------------------------------------------------------------------------------------------------------------------
# Geometries declared in the URDF
# ----------------------------------------------------------------------------------------------------------------------
def _pendulum_urdf_path() -> str:
    return TestUtils.bioptim_folder() + "/examples/models/pendulum.urdf"


def test_pinocchio_geometries_are_read_from_the_urdf():
    """buildModelFromUrdf reads the kinematics only, so without this the animation had nothing to display."""
    model = PinocchioModel(_pendulum_urdf_path())

    assert model.visual_model is not None
    assert model.collision_model is not None
    assert model.visual_model.ngeoms == 1
    assert model.collision_model.ngeoms == 1
    # The mesh is resolved against the folder holding the URDF, not against the working directory.
    assert model.visual_model.geometryObjects[0].meshPath.endswith("mesh/pendulum.STL")


def test_pinocchio_explicit_geometries_win_over_the_urdf_ones():
    sentinel_visual, sentinel_collision = object(), object()

    model = PinocchioModel(_pendulum_urdf_path(), visual_model=sentinel_visual, collision_model=sentinel_collision)

    assert model.visual_model is sentinel_visual
    assert model.collision_model is sentinel_collision


def test_pinocchio_model_built_from_an_object_has_no_geometry():
    """Only a URDF carries geometries; a model handed over as an object has none to read."""
    model = PinocchioModel(_manipulator_pin_model())

    assert model.visual_model is None
    assert model.collision_model is None
