import os
import re

import numpy as np
import pytest

from bioptim import ExternalForceSetTimeSeries, BiorbdModel


# Fixture for creating a standard ExternalForceSetTimeSeries instance
@pytest.fixture
def external_forces():
    return ExternalForceSetTimeSeries(nb_frames=10)


# Fixture for creating a numpy array of forces
@pytest.fixture
def force_array():
    return np.random.rand(6, 10)


# Fixture for creating a numpy array of torques
@pytest.fixture
def torque_array():
    return np.random.rand(3, 10)


def test_initialization(external_forces):
    """Test the basic initialization of ExternalForceSetTimeSeries."""
    assert external_forces.nb_frames == 10
    assert external_forces._can_be_modified is True
    assert external_forces.nb_external_forces == 0
    assert external_forces.nb_external_forces_components == 0


def test_add_global_force(external_forces, force_array):
    """Test adding global forces to a segment."""
    segment_name = "segment1"
    external_forces.add(segment_name, force_array)

    assert len(external_forces.in_global[segment_name]) == 1
    assert np.array_equal(external_forces.in_global[segment_name][0]["values"], force_array)


def test_add_global_force_invalid_shape(external_forces):
    """Test adding global forces with incorrect shape raises an error."""
    segment_name = "segment1"
    invalid_array = np.random.rand(5, 10)  # Wrong number of rows

    with pytest.raises(ValueError, match="External forces must have 6 rows"):
        external_forces.add(segment_name, invalid_array)


def test_add_global_force_wrong_frame_count(external_forces):
    """Test adding global forces with incorrect frame count raises an error."""
    segment_name = "segment1"
    wrong_frame_array = np.random.rand(6, 5)  # Wrong number of frames

    with pytest.raises(ValueError, match="External forces must have the same number of columns"):
        external_forces.add(segment_name, wrong_frame_array)


def test_add_torque(external_forces, torque_array):
    """Test adding global torques to a segment."""
    segment_name = "segment1"
    external_forces.add_torque(segment_name, torque_array)

    assert len(external_forces.torque_in_global[segment_name]) == 1
    assert np.array_equal(external_forces.torque_in_global[segment_name][0]["values"], torque_array)


def test_add_torque_invalid_shape(external_forces):
    """Test adding torques with incorrect shape raises an error."""
    segment_name = "segment1"
    invalid_array = np.random.rand(4, 10)  # Wrong number of rows

    with pytest.raises(ValueError, match="External torques must have 3 rows"):
        external_forces.add_torque(segment_name, invalid_array)


def test_point_of_application(external_forces, force_array):
    """Test adding forces with custom point of application."""
    segment_name = "segment1"
    point_of_application = np.random.rand(3, 10)
    external_forces.add(segment_name, force_array, point_of_application)

    added_force = external_forces.in_global[segment_name][0]
    assert np.array_equal(added_force["point_of_application"], point_of_application)


def test_bind_prevents_modification(external_forces, force_array):
    """Test that binding prevents further modifications."""
    segment_name = "segment1"
    external_forces.add(segment_name, force_array)
    external_forces._bind()

    with pytest.raises(RuntimeError, match="External forces have been binded"):
        external_forces.add(segment_name, force_array)


def test_external_forces_components_calculation(external_forces):
    """Test the calculation of external forces components."""
    segment1, segment2 = "segment1", "segment2"

    # Add various types of forces
    external_forces.add(segment1, np.random.rand(6, 10))
    external_forces.add_torque(segment2, np.random.rand(3, 10))
    external_forces.add_translational_force(segment1, np.random.rand(3, 10))

    # The actual calculation depends on implementation details
    assert external_forces.nb_external_forces_components == 9 + 6 + 6
    assert external_forces.nb_external_forces == 3


def test_to_numerical_time_series(external_forces, force_array):
    """Test conversion to numerical time series."""
    segment_name = "segment1"
    external_forces.add(segment_name, force_array)

    numerical_series = external_forces.to_numerical_time_series()

    assert numerical_series.shape == (9, 1, 11)  # Depends on implementation
    assert np.array_equal(numerical_series[0:6, 0, :-1], force_array)


def test_multiple_force_types(external_forces):
    """Test adding multiple types of forces to the same segment."""
    segment_name = "segment1"

    external_forces.add(segment_name, np.random.rand(6, 10))
    external_forces.add_torque(segment_name, np.random.rand(3, 10))
    external_forces.add_translational_force(segment_name, np.random.rand(3, 10))
    external_forces.add_in_segment_frame(segment_name, np.random.rand(6, 10))
    external_forces.add_torque_in_segment_frame(segment_name, np.random.rand(3, 10))

    assert external_forces.nb_external_forces == 5


def test_fail_within_biomod(external_forces):
    """Test inserting the external forces in a model."""
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    invalid_segment_name = "segment1"
    force_array = np.random.rand(6, 10)
    torque_array = np.random.rand(3, 10)
    point_of_application = np.random.rand(3, 10)

    external_forces.add(invalid_segment_name, force_array, point_of_application)
    external_forces.add_torque(invalid_segment_name, torque_array)

    # Define a model with valid segment names
    valid_segment_names = ("Seg1", "ground", "Test")

    # Check that the ValueError is raised with the correct message
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Segments ['{invalid_segment_name}', '{invalid_segment_name}'] "
            f"specified in the external forces are not in the model."
            f" Available segments are {valid_segment_names}."
        ),
    ):
        BiorbdModel(
            f"{bioptim_folder}/models/cube_with_forces.bioMod",
            external_force_set=external_forces,
        )


def test_success_within_biomod(external_forces):
    """Test inserting the external forces in a model."""
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    force_array = np.random.rand(6, 10)
    torque_array = np.random.rand(3, 10)
    point_of_application = np.random.rand(3, 10)

    external_forces.add("Seg1", force_array, point_of_application)
    external_forces.add_torque("Test", torque_array)

    model = BiorbdModel(
        f"{bioptim_folder}/models/cube_with_forces.bioMod",
        external_force_set=external_forces,
    )

    assert model.external_forces.shape == (15, 1)
    assert model.biorbd_external_forces_set is not None

    with pytest.raises(RuntimeError, match="External forces have been binded and cannot be modified anymore."):
        external_forces.add("Seg1", force_array, point_of_application)
