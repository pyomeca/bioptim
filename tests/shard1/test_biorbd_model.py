"""
Test for file IO.
It tests that a model path with another type than string or biorbdmodel return an error
"""

import biorbd_casadi as biorbd
from bioptim import BiorbdModel
import pytest
import numpy as np
import numpy.testing as npt

from ..utils import TestUtils


def _build_biorbd_model(relative_model_path: str) -> BiorbdModel:
    return BiorbdModel(TestUtils.bioptim_folder() + relative_model_path)


def test_biorbd_model_import():

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = "/examples/models/pendulum.bioMod"
    BiorbdModel(bioptim_folder + model_path)

    BiorbdModel(biorbd.Model(bioptim_folder + model_path))

    with pytest.raises(ValueError, match="The model should be of type 'str' or 'biorbd.Model'"):
        BiorbdModel(1)

    with pytest.raises(ValueError, match="The model should be of type 'str' or 'biorbd.Model'"):
        BiorbdModel([])


@pytest.mark.parametrize(
    "my_keys",
    [
        ["q"],
        ["qdot"],
        ["qddot"],
        ["q", "qdot"],
        ["qdot", "q"],
        ["q", "qdot", "qddot"],
        ["qddot", "q", "qdot"],
        ["qdot", "qddot", "q"],
        ["q", "qddot", "qdot"],
        ["qddot", "qdot", "q"],
        ["qdot", "q", "qddot"],
    ],
)
def test_bounds_from_ranges(my_keys):
    x_min_q = [[-1.0, -1.0, -1.0], [-6.28318531, -6.28318531, -6.28318531]]
    x_min_qdot = [[-31.41592654, -31.41592654, -31.41592654], [-31.41592654, -31.41592654, -31.41592654]]
    x_min_qddot = [[-314.15926536, -314.15926536, -314.15926536], [-314.15926536, -314.15926536, -314.15926536]]
    x_max_q = [[5.0, 5.0, 5.0], [6.28318531, 6.28318531, 6.28318531]]
    x_max_qdot = [[31.41592654, 31.41592654, 31.41592654], [31.41592654, 31.41592654, 31.41592654]]
    x_max_qddot = [[314.15926536, 314.15926536, 314.15926536], [314.15926536, 314.15926536, 314.15926536]]

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = "/examples/models/pendulum.bioMod"
    bio_model = BiorbdModel(bioptim_folder + model_path)

    for key in my_keys:
        x_bounds = bio_model.bounds_from_ranges(key)

        if key == "q":
            x_min = np.array(x_min_q)
            x_max = np.array(x_max_q)

        elif key == "qdot":
            x_min = np.array(x_min_qdot)
            x_max = np.array(x_max_qdot)

        elif key == "qddot":
            x_min = np.array(x_min_qddot)
            x_max = np.array(x_max_qddot)

        else:
            raise ValueError("Wrong key")

        # Check min and max have the right value
        npt.assert_almost_equal(x_bounds.min, x_min)
        npt.assert_almost_equal(x_bounds.max, x_max)


def test_function_cached():

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = "/examples/models/pendulum.bioMod"
    bio_model = BiorbdModel(bioptim_folder + model_path)

    # No cached function
    assert len(bio_model._cached_functions.keys()) == 0

    # CoM (no argument in the first parenthesis)
    # First call create the function and cache it
    com = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    com_id = id(bio_model._cached_functions[("center_of_mass", (), frozenset())])
    assert len(bio_model._cached_functions.keys()) == 1

    # Second call use the cached function
    com = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    assert com_id == id(bio_model._cached_functions[("center_of_mass", (), frozenset())])
    assert len(bio_model._cached_functions.keys()) == 1

    # marker (with an argument in the first parenthesis)
    # First call create the function and cache it
    marker = bio_model.marker(index=0)(bio_model.q, bio_model.parameters)
    marker_id = id(bio_model._cached_functions[("marker", (), frozenset({("index", 0)}))])
    assert len(bio_model._cached_functions.keys()) == 2

    # Second call use the cached function
    marker = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    assert marker_id == id(bio_model._cached_functions[("marker", (), frozenset({("index", 0)}))])
    assert len(bio_model._cached_functions.keys()) == 2

    # First call with other argument creates the function again and cache it
    marker2 = bio_model.marker(index=1)(bio_model.q, bio_model.parameters)
    marker_id2 = id(bio_model._cached_functions[("marker", (), frozenset({("index", 1)}))])
    assert len(bio_model._cached_functions.keys()) == 3
    assert marker_id != marker_id2

    # Second call use the cached function
    marker2 = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    assert marker_id2 == id(bio_model._cached_functions[("marker", (), frozenset({("index", 1)}))])
    assert len(bio_model._cached_functions.keys()) == 3


def test_biorbd_model_metadata_copy_and_serialize():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.bioMod"
    bio_model = BiorbdModel(model_path)

    assert bio_model.path.replace("\\", "/") == model_path.replace("\\", "/")
    assert bio_model.name == "pendulum.bioMod"

    model_copy = bio_model.copy()
    assert isinstance(model_copy, BiorbdModel)
    assert model_copy is not bio_model
    assert model_copy.path == bio_model.path

    serializer, data = bio_model.serialize()
    assert serializer is BiorbdModel
    assert data["bio_model"].replace("\\", "/") == model_path.replace("\\", "/")
    assert data["external_force_set"] is None


def test_biorbd_gravity_setter_updates_value():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.bioMod"
    bio_model = BiorbdModel(model_path)

    new_gravity = np.array([0.1, -0.2, -3.5])
    bio_model.set_gravity(new_gravity)

    updated_gravity = np.array(bio_model.gravity()(np.zeros((0, 1)))).squeeze()
    npt.assert_almost_equal(updated_gravity, new_gravity)


def test_biorbd_set_friction_coefficients_validation_and_shape():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/pendulum.bioMod"
    bio_model = BiorbdModel(model_path)

    assert bio_model.friction_coefficients is None

    valid_coefficients = np.ones((bio_model.nb_tau, bio_model.nb_qdot))
    bio_model.set_friction_coefficients(valid_coefficients)
    npt.assert_almost_equal(bio_model.friction_coefficients, valid_coefficients)

    with pytest.raises(ValueError, match="Friction coefficients must be positive"):
        bio_model.set_friction_coefficients(-valid_coefficients)


def test_biorbd_tau_max_output_shape_and_signs():
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/models/cube_with_actuators.bioMod"
    bio_model = BiorbdModel(model_path)

    q = np.zeros((bio_model.nb_q, 1))
    qdot = np.zeros((bio_model.nb_qdot, 1))
    parameters = np.zeros((0, 1))

    tau_max, tau_min = bio_model.tau_max()(q, qdot, parameters)
    tau_max = np.array(tau_max)
    tau_min = np.array(tau_min)

    assert tau_max.shape == (bio_model.nb_tau, 1)
    assert tau_min.shape == (bio_model.nb_tau, 1)
    assert np.all(tau_max >= tau_min)


def test_biorbd_basic_properties_and_name_consistency():
    bio_model = _build_biorbd_model("/examples/models/pendulum.bioMod")

    assert bio_model.nb_q == bio_model.nb_qdot == bio_model.nb_qddot == bio_model.nb_tau == bio_model.nb_dof
    assert bio_model.nb_segments == len(bio_model.segments)
    assert len(bio_model.name_dofs) == bio_model.nb_dof
    assert len(bio_model.marker_names) == bio_model.nb_markers
    assert bio_model.nb_muscles == len(bio_model.muscle_names)
    assert bio_model.nb_soft_contacts == len(bio_model.soft_contact_names)

    if bio_model.nb_markers > 0:
        first_marker_name = bio_model.marker_names[0]
        marker_idx = bio_model.marker_index(first_marker_name)
        assert 0 <= marker_idx < bio_model.nb_markers


def test_biorbd_kinematics_and_dynamics_functions_shapes():
    bio_model = _build_biorbd_model("/examples/models/pendulum.bioMod")

    q = np.zeros((bio_model.nb_q, 1))
    qdot = np.zeros((bio_model.nb_qdot, 1))
    qddot = np.zeros((bio_model.nb_qddot, 1))
    tau = np.zeros((bio_model.nb_tau, 1))
    external_forces = np.zeros((0, 1))
    parameters = np.zeros((0, 1))

    mass_value = np.array(bio_model.mass()(parameters)).squeeze()
    assert np.isfinite(mass_value)
    assert float(mass_value) > 0

    com = np.array(bio_model.center_of_mass()(q, parameters))
    com_velocity = np.array(bio_model.center_of_mass_velocity()(q, qdot, parameters))
    com_acceleration = np.array(bio_model.center_of_mass_acceleration()(q, qdot, qddot, parameters))
    assert com.shape == (3, 1)
    assert com_velocity.shape == (3, 1)
    assert com_acceleration.shape == (3, 1)

    mass_matrix = np.array(bio_model.mass_matrix()(q, parameters))
    non_linear_effects = np.array(bio_model.non_linear_effects()(q, qdot, parameters))
    angular_momentum = np.array(bio_model.angular_momentum()(q, qdot, parameters))
    reshaped_qdot = np.array(bio_model.reshape_qdot()(q, qdot, parameters))
    lagrangian = np.array(bio_model.lagrangian()(q, qdot)).squeeze()

    assert mass_matrix.shape == (bio_model.nb_q, bio_model.nb_q)
    assert non_linear_effects.shape == (bio_model.nb_q, 1)
    assert angular_momentum.shape == (3, 1)
    assert reshaped_qdot.shape == (bio_model.nb_qdot, 1)
    assert np.isfinite(float(lagrangian))

    forward_dynamics = np.array(bio_model.forward_dynamics()(q, qdot, tau, external_forces, parameters))
    inverse_dynamics = np.array(bio_model.inverse_dynamics()(q, qdot, qddot, external_forces, parameters))
    assert forward_dynamics.shape == (bio_model.nb_qddot, 1)
    assert inverse_dynamics.shape == (bio_model.nb_tau, 1)


def test_biorbd_rotation_and_homogeneous_matrix_utilities():
    bio_model = _build_biorbd_model("/examples/models/pendulum.bioMod")

    q = np.zeros((bio_model.nb_q, 1))
    parameters = np.zeros((0, 1))

    euler_angles = np.array(bio_model.rotation_matrix_to_euler_angles("xyz")(np.eye(3))).squeeze()
    npt.assert_allclose(euler_angles, np.zeros((3,)), atol=1e-12)

    homogeneous = np.array(bio_model.homogeneous_matrices_in_global(0)(q, parameters))
    homogeneous_inverse = np.array(bio_model.homogeneous_matrices_in_global(0, inverse=True)(q, parameters))
    local_homogeneous = np.array(bio_model.homogeneous_matrices_in_child(0)(parameters))
    assert homogeneous.shape == (4, 4)
    assert homogeneous_inverse.shape == (4, 4)
    assert local_homogeneous.shape == (4, 4)
    npt.assert_allclose(homogeneous @ homogeneous_inverse, np.eye(4), atol=1e-8)


def test_biorbd_segment_orientation_currently_raises_on_casadi_types():
    bio_model = _build_biorbd_model("/examples/models/pendulum.bioMod")
    q = np.zeros((bio_model.nb_q, 1))
    parameters = np.zeros((0, 1))

    with pytest.raises(NotImplementedError):
        bio_model.segment_orientation(0)(q, parameters)


def test_biorbd_marker_related_outputs_are_consistent():
    bio_model = _build_biorbd_model("/examples/models/pendulum.bioMod")
    if bio_model.nb_markers == 0:
        pytest.skip("This model has no markers")

    q = np.zeros((bio_model.nb_q, 1))
    qdot = np.zeros((bio_model.nb_qdot, 1))
    qddot = np.zeros((bio_model.nb_qddot, 1))
    parameters = np.zeros((0, 1))

    markers = np.array(bio_model.markers()(q, parameters))
    marker0 = np.array(bio_model.marker(0)(q, parameters))
    marker0_velocity = np.array(bio_model.marker_velocity(0)(q, qdot, parameters))
    marker0_acceleration = np.array(bio_model.marker_acceleration(0)(q, qdot, qddot, parameters))
    markers_velocities = np.array(bio_model.markers_velocities()(q, qdot, parameters))
    markers_accelerations = np.array(bio_model.markers_accelerations()(q, qdot, qddot, parameters))

    assert markers.shape == (3, bio_model.nb_markers)
    assert marker0.shape == (3, 1)
    assert marker0_velocity.shape == (3, 1)
    assert marker0_acceleration.shape == (3, 1)
    assert markers_velocities.shape == (3, bio_model.nb_markers)
    assert markers_accelerations.shape == (3, bio_model.nb_markers)
    npt.assert_allclose(marker0.squeeze(), markers[:, 0], atol=1e-12)

    with pytest.raises(RuntimeError, match="Mismatching number of output names"):
        bio_model.markers_jacobian()(q, parameters)


def test_biorbd_rigid_contact_methods_shapes():
    bio_model = _build_biorbd_model("/examples/models/2segments_2dof_2contacts.bioMod")
    assert bio_model.nb_rigid_contacts > 0
    assert bio_model.nb_contacts > 0

    q = np.zeros((bio_model.nb_q, 1))
    qdot = np.zeros((bio_model.nb_qdot, 1))
    qddot = np.zeros((bio_model.nb_qddot, 1))
    tau = np.zeros((bio_model.nb_tau, 1))
    external_forces = np.zeros((0, 1))
    parameters = np.zeros((0, 1))

    rigid_contact_names = bio_model.rigid_contact_names
    if len(rigid_contact_names) > 0:
        assert 0 <= bio_model.contact_index(rigid_contact_names[0]) < bio_model.nb_contacts

    for contact_idx in range(bio_model.nb_rigid_contacts):
        axes = bio_model.rigid_contact_axes_index(contact_idx)
        assert len(axes) > 0
        assert isinstance(bio_model.rigid_contact_segment(contact_idx), str)

        velocity = np.array(bio_model.rigid_contact_velocity(contact_idx)(q, qdot, parameters))
        velocity_on_available_axes = np.array(
            bio_model.rigid_contact_velocity(contact_idx, contact_axis=tuple(axes))(q, qdot, parameters)
        )
        acceleration_on_first_axis = np.array(
            bio_model.rigid_contact_acceleration(contact_idx, (axes[0],))(q, qdot, qddot, parameters)
        )

        assert velocity.shape == (3, 1)
        assert velocity_on_available_axes.shape == (len(axes), 1)
        assert acceleration_on_first_axis.shape == (1, 1)


def test_biorbd_constrained_contact_dynamics_currently_raises_bad_allocation():
    bio_model = _build_biorbd_model("/examples/models/2segments_2dof_2contacts.bioMod")

    q = np.zeros((bio_model.nb_q, 1))
    qdot = np.zeros((bio_model.nb_qdot, 1))
    tau = np.zeros((bio_model.nb_tau, 1))
    external_forces = np.zeros((0, 1))
    parameters = np.zeros((0, 1))

    with pytest.raises(RuntimeError, match="bad allocation"):
        bio_model.forward_dynamics(with_contact=True)(q, qdot, tau, external_forces, parameters)


def test_biorbd_inverse_dynamics_with_contact_not_implemented():
    bio_model = _build_biorbd_model("/examples/models/pendulum.bioMod")
    with pytest.raises(NotImplementedError, match="Inverse dynamics with contact is not implemented yet"):
        bio_model.inverse_dynamics(with_contact=True)


def test_biorbd_reorder_qddot_root_joints_static_method():
    qddot_root = np.array([[1.0], [2.0]])
    qddot_joints = np.array([[3.0]])
    reordered = BiorbdModel.reorder_qddot_root_joints(qddot_root, qddot_joints)
    npt.assert_allclose(np.array(reordered), np.array([[1.0], [2.0], [3.0]]))
