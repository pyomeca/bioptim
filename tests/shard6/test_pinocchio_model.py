import importlib.util

import numpy as np
import pytest

from bioptim import PinocchioModel
from ..utils import TestUtils

HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None


def test_pinocchio_model_import_is_lazy():
    with pytest.raises(ValueError, match="The model should be of type 'str' or 'pinocchio.Model'"):
        PinocchioModel(1)

    with pytest.raises(ValueError, match="The model should be of type 'str' or 'pinocchio.Model'"):
        PinocchioModel([])


def test_pinocchio_missing_dependency_message():
    if HAS_PINOCCHIO:
        pytest.skip("Pinocchio is installed in this environment")

    with pytest.raises(ModuleNotFoundError, match="requires the optional dependency 'pinocchio'"):
        PinocchioModel("model.urdf")


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
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


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
def test_pinocchio_segment_index_raises_for_unknown_segment():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    with pytest.raises(ValueError, match="not_a_segment is not a segment name"):
        model.segment_index("not_a_segment")


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
def test_pinocchio_gravity_setter_updates_gravity_function_value():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    model.set_gravity(np.array([0.1, -0.2, -3.5]))
    updated_gravity = np.array(model._model.gravity.linear).squeeze()
    np.testing.assert_almost_equal(updated_gravity, np.array([0.1, -0.2, -3.5]))


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
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


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
def test_pinocchio_not_implemented_methods_raise():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.urdf"
    model = PinocchioModel(model_path)

    with pytest.raises(NotImplementedError, match="forward_dynamics with contact is not implemented"):
        model.forward_dynamics(with_contact=True)

    with pytest.raises(NotImplementedError, match="Animation is not implemented"):
        model.animate()


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
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


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
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


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")
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
