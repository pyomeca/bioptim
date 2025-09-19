import re

from bioptim import ExternalForceSetTimeSeries, ExternalForceSetVariables, BiorbdModel
import numpy as np
import pytest

from ..utils import TestUtils


# Fixture for creating a standard ExternalForceSetTimeSeries instance
@pytest.fixture
def external_forces_timeseries():
    return ExternalForceSetTimeSeries(nb_frames=10)

# Fixture for creating a standard ExternalForceSetVariable instance
@pytest.fixture
def external_forces_variables():
    return ExternalForceSetVariables()

# Fixture for creating a numpy array of forces
@pytest.fixture
def force_array():
    return np.random.rand(6, 10)


# Fixture for creating a numpy array of torques
@pytest.fixture
def torque_array():
    return np.random.rand(3, 10)


def test_initialization_timeseries(external_forces_timeseries):
    """Test the basic initialization of ExternalForceSetTimeSeries."""
    assert external_forces_timeseries.nb_frames == 10
    assert external_forces_timeseries._can_be_modified is True
    assert external_forces_timeseries.nb_external_forces_components == 0

def test_initialization_variables(external_forces_variables):
    """Test the basic initialization of ExternalForceSetTimeSeries."""
    assert external_forces_variables._can_be_modified is True
    assert external_forces_variables.nb_external_forces_components == 0


def test_add_global_force_timeseries(external_forces_timeseries, force_array):
    """Test adding global forces to a segment."""
    force_name = "force0"
    segment_name = "segment1"
    external_forces_timeseries.add(force_name, segment_name, force_array)
    assert np.array_equal(external_forces_timeseries.in_global[force_name]["values"], force_array)

def test_add_global_force_variables(external_forces_variables):
    """Test adding global forces to a segment."""
    force_name = "force0"
    segment_name = "segment1"
    external_forces_variables.add(force_name, segment_name)
    assert external_forces_variables.in_global[force_name]["force"].shape == (6, 1)


def test_add_global_force_invalid_shape_timeseries(external_forces_timeseries):
    """Test adding global forces with incorrect shape raises an error."""
    force_name = "force0"
    segment_name = "segment1"
    invalid_array = np.random.rand(5, 10)  # Wrong number of rows

    with pytest.raises(ValueError, match="External forces must have 6 rows"):
        external_forces_timeseries.add(force_name, segment_name, invalid_array)


def test_add_global_force_wrong_frame_count_timeseries(external_forces_timeseries):
    """Test adding global forces with incorrect frame count raises an error."""
    force_name = "force0"
    segment_name = "segment1"
    wrong_frame_array = np.random.rand(6, 5)  # Wrong number of frames

    with pytest.raises(ValueError, match="External forces must have the same number of columns"):
        external_forces_timeseries.add(force_name, segment_name, wrong_frame_array)


def test_add_torque_timeseries(external_forces_timeseries, torque_array):
    """Test adding global torques to a segment."""
    segment_name = "segment1"
    force_name = "torque"
    external_forces_timeseries.add_torque(force_name, segment_name, torque_array)
    assert np.array_equal(external_forces_timeseries.torque_in_global[force_name]["values"], torque_array)

def test_add_torque_variables(external_forces_variables):
    """Test adding global torques to a segment."""
    segment_name = "segment1"
    force_name = "torque"
    external_forces_variables.add_torque(force_name, segment_name)
    assert external_forces_variables.torque_in_global[force_name]["force"].shape == (3, 1)


def test_add_torque_invalid_shape_timeseries(external_forces_timeseries):
    """Test adding torques with incorrect shape raises an error."""
    force_name = "torque"
    segment_name = "segment1"
    invalid_array = np.random.rand(4, 10)  # Wrong number of rows

    with pytest.raises(ValueError, match="External torques must have 3 rows"):
        external_forces_timeseries.add_torque(force_name, segment_name, invalid_array)


def test_point_of_application_timeseries(external_forces_timeseries, force_array):
    """Test adding forces with custom point of application."""
    force_name = "force0"
    segment_name = "segment1"
    point_of_application = np.random.rand(3, 10)
    external_forces_timeseries.add(force_name, segment_name, force_array, point_of_application)

    added_force = external_forces_timeseries.in_global[force_name]
    assert np.array_equal(added_force["point_of_application"], point_of_application)

def test_point_of_application_variables(external_forces_variables):
    """Test adding forces with custom point of application."""
    force_name = "force0"
    segment_name = "segment1"
    external_forces_variables.add(force_name, segment_name, use_point_of_application=True)
    assert external_forces_variables.in_global[force_name]["point_of_application"].shape == (3, 1)


def test_bind_prevents_modification_timeseries(external_forces_timeseries, force_array):
    """Test that binding prevents further modifications."""
    force_name = "force0"
    segment_name = "segment1"
    external_forces_timeseries.add(force_name, segment_name, force_array)
    external_forces_timeseries.bind()

    with pytest.raises(RuntimeError, match="External forces have been binded"):
        external_forces_timeseries.add(force_name, segment_name, force_array)

def test_bind_prevents_modification_variables(external_forces_variables, force_array):
    """Test that binding prevents further modifications."""
    force_name = "force0"
    segment_name = "segment1"
    external_forces_variables.add(force_name, segment_name)
    external_forces_variables.bind()

    with pytest.raises(RuntimeError, match="External forces have been binded"):
        external_forces_variables.add(force_name, segment_name)


def test_external_forces_timeseries_components_calculation_timeseries(external_forces_timeseries):
    """Test the calculation of external forces components."""
    segment1, segment2 = "segment1", "segment2"

    # Add various types of forces
    external_forces_timeseries.add("force0", segment1, np.random.rand(6, 10))
    external_forces_timeseries.add_torque("force1", segment2, np.random.rand(3, 10))
    external_forces_timeseries.add_translational_force("force2", segment1, np.random.rand(3, 10))

    # The actual calculation depends on implementation details
    assert external_forces_timeseries.nb_external_forces_components == 9 + 3 + 6

def test_external_forces_timeseries_components_calculation_variables(external_forces_variables):
    """Test the calculation of external forces components."""
    segment1, segment2 = "segment1", "segment2"

    # Add various types of forces
    external_forces_variables.add("force0", segment1)
    external_forces_variables.add_torque("force1", segment2)
    external_forces_variables.add_translational_force("force2", segment1)

    # The actual calculation depends on implementation details
    assert external_forces_variables.nb_external_forces_components == 9 + 3 + 6


def test_to_numerical_time_series_timeseries(external_forces_timeseries, force_array):
    """Test conversion to numerical time series."""
    force_name = "force0"
    segment_name = "segment1"
    external_forces_timeseries.add(force_name, segment_name, force_array)

    numerical_series = external_forces_timeseries.to_numerical_time_series()

    assert numerical_series.shape == (9, 1, 11)  # Depends on implementation
    assert np.array_equal(numerical_series[0:6, 0, :-1], force_array)


def test_multiple_force_types_timeseries(external_forces_timeseries):
    """Test adding multiple types of forces to the same segment."""
    segment_name = "segment1"

    external_forces_timeseries.add("force0", segment_name, np.random.rand(6, 10))
    external_forces_timeseries.add_torque("force1", segment_name, np.random.rand(3, 10))
    external_forces_timeseries.add_translational_force("force2", segment_name, np.random.rand(3, 10))
    external_forces_timeseries.add_in_segment_frame("force3", segment_name, np.random.rand(6, 10))
    external_forces_timeseries.add_torque_in_segment_frame("force4", segment_name, np.random.rand(3, 10))

    assert external_forces_timeseries.nb_external_forces_components == 30

def test_multiple_force_types_variables(external_forces_variables):
    """Test adding multiple types of forces to the same segment."""
    segment_name = "segment1"

    external_forces_variables.add("force0", segment_name)
    external_forces_variables.add_torque("force1", segment_name)
    external_forces_variables.add_translational_force("force2", segment_name)
    external_forces_variables.add_in_segment_frame("force3", segment_name)
    external_forces_variables.add_torque_in_segment_frame("force4", segment_name)

    assert external_forces_variables.nb_external_forces_components == 30


def test_fail_within_biomod_timeseries(external_forces_timeseries):
    """Test inserting the external forces in a model."""
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    invalid_segment_name = "segment1"
    force_array = np.random.rand(6, 10)
    torque_array = np.random.rand(3, 10)
    point_of_application = np.random.rand(3, 10)

    external_forces_timeseries.add("force", invalid_segment_name, force_array, point_of_application)
    external_forces_timeseries.add_torque("torque", invalid_segment_name, torque_array)

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
            f"{bioptim_folder}/examples/models/cube_with_forces.bioMod",
            external_force_set=external_forces_timeseries,
        )

def test_fail_within_biomod_variables(external_forces_variables):
    """Test inserting the external forces in a model."""
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    invalid_segment_name = "segment1"

    external_forces_variables.add("force", invalid_segment_name)
    external_forces_variables.add_torque("torque", invalid_segment_name)

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
            f"{bioptim_folder}/examples/models/cube_with_forces.bioMod",
            external_force_set=external_forces_variables,
        )


def test_success_within_biomod_timeseries(external_forces_timeseries):
    """Test inserting the external forces in a model."""
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    force_array = np.random.rand(6, 10)
    torque_array = np.random.rand(3, 10)
    point_of_application = np.random.rand(3, 10)

    external_forces_timeseries.add("force0", "Seg1", force_array, point_of_application)
    external_forces_timeseries.add_torque("force1", "Test", torque_array)

    model = BiorbdModel(
        f"{bioptim_folder}/examples/models/cube_with_forces.bioMod",
        external_force_set=external_forces_timeseries,
    )

    assert model.external_forces.shape == (9 + 3, 1)
    assert model.biorbd_external_forces_set is not None

    with pytest.raises(RuntimeError, match="External forces have been binded and cannot be modified anymore."):
        external_forces_timeseries.add("force2", "Seg1", force_array, point_of_application)

def test_success_within_biomod_variables(external_forces_variables):
    """Test inserting the external forces in a model."""
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    external_forces_variables.add("force0", "Seg1")
    external_forces_variables.add_torque("force1", "Test")

    model = BiorbdModel(
        f"{bioptim_folder}/examples/models/cube_with_forces.bioMod",
        external_force_set=external_forces_variables,
    )

    assert model.external_forces.shape == (9 + 3, 1)
    assert model.biorbd_external_forces_set is not None

    with pytest.raises(RuntimeError, match="External forces have been binded and cannot be modified anymore."):
        external_forces_variables.add("force2", "Seg1")
