import pytest
from bioptim import BiorbdModel
from casadi import MX
from tests.utils import TestUtils

bioptim_folder = TestUtils.bioptim_folder()


@pytest.fixture
def model():
    return


def generate_q_vectors(model):
    q_valid = MX([0.1] * model.nb_q)
    q_too_large = MX([0.1] * (model.nb_q + 1))
    return q_valid, q_too_large


def generate_q_and_qdot_vectors(model):
    q_valid = MX([0.1] * model.nb_q)
    qdot_valid = MX([0.1] * model.nb_qdot)
    q_too_large = MX([0.1] * (model.nb_q + 1))
    qdot_too_large = MX([0.1] * (model.nb_qdot + 1))

    return q_valid, qdot_valid, q_too_large, qdot_too_large


def generate_q_qdot_qddot_vectors(model, root_dynamics=False):
    q_valid = MX([0.1] * model.nb_q)
    qdot_valid = MX([0.1] * model.nb_qdot)
    nb_qddot = model.nb_qddot - model.nb_root if root_dynamics else model.nb_qddot
    qddot_valid = MX([0.1] * nb_qddot)

    q_too_large = MX([0.1] * (model.nb_q + 1))
    qdot_too_large = MX([0.1] * (model.nb_qdot + 1))
    qddot_too_large = MX([0.1] * (model.nb_qddot + 1))

    return (
        q_valid,
        qdot_valid,
        qddot_valid,
        q_too_large,
        qdot_too_large,
        qddot_too_large,
    )


def generate_tau_activations_vectors(model):
    tau_activations_valid = MX([0.1] * model.nb_tau)
    tau_activations_too_large = MX([0.1] * (model.nb_tau + 1))
    return tau_activations_valid, tau_activations_too_large


def generate_muscle_vectors(model):
    muscle_valid = MX([0.1] * model.nb_muscles)
    muscle_too_large = MX([0.1] * (model.nb_muscles + 1))
    return muscle_valid, muscle_too_large


def test_center_of_mass_valid_and_too_large_q_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, q_too_large = generate_q_vectors(model)

    # q valid
    model.center_of_mass(q_valid)
    # q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.center_of_mass(q_too_large)


def test_center_of_mass_velocity_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.center_of_mass_velocity(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.center_of_mass_velocity(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.center_of_mass_velocity(q_valid, qdot_too_large)


def test_center_of_mass_acceleration_valid_and_too_large_q_or_qdot_or_qddot_input(
    model,
):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    (
        q_valid,
        qdot_valid,
        qddot_valid,
        q_too_large,
        qdot_too_large,
        qddot_too_large,
    ) = generate_q_qdot_qddot_vectors(model)

    # q, qdot and qddot valid
    model.center_of_mass_acceleration(q_valid, qdot_valid, qddot_valid)
    # qdot and qddot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.center_of_mass_acceleration(q_too_large, qdot_valid, qddot_valid)
    # q and qddot valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.center_of_mass_acceleration(q_valid, qdot_too_large, qddot_valid)
    # q and qdot valid but qddot not valid
    with pytest.raises(ValueError, match="Length of qddot size should be: 4, but got: 5"):
        model.center_of_mass_acceleration(q_valid, qdot_valid, qddot_too_large)


def test_body_rotation_rate_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.body_rotation_rate(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.body_rotation_rate(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.body_rotation_rate(q_valid, qdot_too_large)


def test_mass_matrix_valid_and_too_large_q_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, q_too_large = generate_q_vectors(model)

    # q valid
    model.mass_matrix(q_valid)
    # q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.mass_matrix(q_too_large)


def test_non_linear_effects_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.non_linear_effects(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.non_linear_effects(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.non_linear_effects(q_valid, qdot_too_large)


def test_angular_momentum_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.angular_momentum(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.angular_momentum(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.angular_momentum(q_valid, qdot_too_large)


def test_reshape_qdot_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.reshape_qdot(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.reshape_qdot(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.reshape_qdot(q_valid, qdot_too_large)


def test_segment_angular_velocity_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)
    idx = 1
    # q and qdot valid
    model.segment_angular_velocity(q_valid, qdot_valid, idx)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.segment_angular_velocity(q_too_large, qdot_valid, idx)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.segment_angular_velocity(q_valid, qdot_too_large, idx)


def test_forward_dynamics_free_floating_base_valid_and_too_large_q_or_qdot_or_qddot_joints_input(
    model,
):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    (
        q_valid,
        qdot_valid,
        qddot_joints_valid,
        q_too_large,
        qdot_too_large,
        qddot_joints_too_large,
    ) = generate_q_qdot_qddot_vectors(model, root_dynamics=True)

    # q, qdot and qddot_joints valid
    model.forward_dynamics_free_floating_base(q_valid, qdot_valid, qddot_joints_valid)
    # qdot and qddot_joints valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.forward_dynamics_free_floating_base(q_too_large, qdot_valid, qddot_joints_valid)
    # q and qddot_joints valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.forward_dynamics_free_floating_base(q_valid, qdot_too_large, qddot_joints_valid)
    # q and qdot valid but qddot_joints not valid
    with pytest.raises(ValueError, match="Length of qddot_joints size should be: 1, but got: 5"):
        model.forward_dynamics_free_floating_base(q_valid, qdot_valid, qddot_joints_too_large)


def test_forward_dynamics_valid_and_too_large_q_or_qdot_or_tau_activations_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)
    tau_valid, tau_too_large = generate_tau_activations_vectors(model)

    # q, qdot and tau valid
    model.forward_dynamics(q_valid, qdot_valid, tau_valid)
    # qdot and tau valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.forward_dynamics(q_too_large, qdot_valid, tau_valid)
    # q and tau valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.forward_dynamics(q_valid, qdot_too_large, tau_valid)
    # q and qdot valid but tau not valid
    with pytest.raises(ValueError, match="Length of tau size should be: 4, but got: 5"):
        model.forward_dynamics(q_valid, qdot_valid, tau_too_large)


def test_constrained_forward_dynamics_valid_and_too_large_q_or_qdot_or_tau_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)
    tau_valid, tau_too_large = generate_tau_activations_vectors(model)

    # q, qdot and tau valid
    model.forward_dynamics(with_contact=True)(q_valid, qdot_valid, tau_valid)
    # qdot and tau valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.forward_dynamics(with_contact=True)(q_too_large, qdot_valid, tau_valid)
    # q and tau valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.forward_dynamics(with_contact=True)(q_valid, qdot_too_large, tau_valid)
    # q and qdot valid but tau not valid
    with pytest.raises(ValueError, match="Length of tau size should be: 4, but got: 5"):
        model.forward_dynamics(with_contact=True)(q_valid, qdot_valid, tau_too_large)


def test_inverse_dynamics_valid_and_too_large_q_or_qdot_or_qddot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    (
        q_valid,
        qdot_valid,
        qddot_valid,
        q_too_large,
        qdot_too_large,
        qddot_too_large,
    ) = generate_q_qdot_qddot_vectors(model)

    # q, qdot and qddot valid
    model.inverse_dynamics(q_valid, qdot_valid, qddot_valid)
    # qdot and qddot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.inverse_dynamics(q_too_large, qdot_valid, qddot_valid)
    # q and qddot valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.inverse_dynamics(q_valid, qdot_too_large, qddot_valid)
    # q and qdot valid but qddot not valid
    with pytest.raises(ValueError, match="Length of qddot size should be: 4, but got: 5"):
        model.inverse_dynamics(q_valid, qdot_valid, qddot_too_large)


def test_contact_forces_from_constrained_forward_dynamics_valid_and_too_large_q_or_qdot_or_tau_input(
    model,
):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)
    tau_valid, tau_too_large = generate_tau_activations_vectors(model)

    # q, qdot and tau valid
    model.contact_forces_from_constrained_forward_dynamics(q_valid, qdot_valid, tau_valid)
    # qdot and tau valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.contact_forces_from_constrained_forward_dynamics(q_too_large, qdot_valid, tau_valid)
    # q and tau valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.contact_forces_from_constrained_forward_dynamics(q_valid, qdot_too_large, tau_valid)
    # q and qdot valid but tau not valid
    with pytest.raises(ValueError, match="Length of tau size should be: 4, but got: 5"):
        model.contact_forces_from_constrained_forward_dynamics(q_valid, qdot_valid, tau_too_large)


def test_qdot_from_impact_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.qdot_from_impact(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.qdot_from_impact(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.qdot_from_impact(q_valid, qdot_too_large)


def test_muscle_activation_dot_valid_and_too_large_q_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    muscle_valid, muscle_too_large = generate_muscle_vectors(model)

    # muscle valid
    model.muscle_activation_dot(muscle_valid)
    # muscle not valid
    with pytest.raises(ValueError, match="Length of muscle size should be: 1, but got: 2"):
        model.muscle_activation_dot(muscle_too_large)


def test_muscle_length_jacobian_valid_and_too_large_q_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, q_too_large = generate_q_vectors(model)

    # q valid
    model.muscle_length_jacobian(q_valid)
    # q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.muscle_length_jacobian(q_too_large)


def test_muscle_velocity_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.muscle_velocity(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.muscle_velocity(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.muscle_velocity(q_valid, qdot_too_large)


def test_muscle_joint_torque_valid_and_too_large_q_or_qdot_or_qddot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    (
        q_valid,
        qdot_valid,
        qddot_valid,
        q_too_large,
        qdot_too_large,
        qddot_too_large,
    ) = generate_q_qdot_qddot_vectors(model)
    muscle_valid, muscle_too_large = generate_muscle_vectors(model)

    # q, qdot and qddot valid
    model.muscle_joint_torque(muscle_valid, q_valid, qdot_valid)
    # qdot and qddot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.muscle_joint_torque(muscle_valid, q_too_large, qdot_valid)
    # q and qddot valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.muscle_joint_torque(muscle_valid, q_valid, qdot_too_large)
    # q and qdot valid but qddot not valid
    with pytest.raises(ValueError, match="Length of muscle size should be: 1, but got: 2"):
        model.muscle_joint_torque(muscle_too_large, q_valid, qdot_valid)


def test_markers_valid_and_too_large_q_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, q_too_large = generate_q_vectors(model)

    # q valid
    model.markers(q_valid)
    # q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.markers(q_too_large)


def test_marker_valid_and_too_large_q_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, q_too_large = generate_q_vectors(model)

    # q valid
    model.marker(q_valid, 1)
    # q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.marker(q_too_large, 1)


def test_marker_velocities_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.marker_velocities(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.marker_velocities(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.marker_velocities(q_valid, qdot_too_large)


def test_marker_accelerations_valid_and_too_large_q_or_qdot_or_qddot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    (
        q_valid,
        qdot_valid,
        qddot_valid,
        q_too_large,
        qdot_too_large,
        qddot_too_large,
    ) = generate_q_qdot_qddot_vectors(model)

    # q, qdot and qddot valid
    model.marker_accelerations(q_valid, qdot_valid, qddot_valid)
    # qdot and qddot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.marker_accelerations(q_too_large, qdot_valid, qddot_valid)
    # q and qddot valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.marker_accelerations(q_valid, qdot_too_large, qddot_valid)
    # q and qdot valid but qddot not valid
    with pytest.raises(ValueError, match="Length of qddot size should be: 4, but got: 5"):
        model.marker_accelerations(q_valid, qdot_valid, qddot_too_large)


def test_tau_max_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = bioptim_folder + "/examples/optimal_time_ocp/models/cube.bioMod"
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.tau_max(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 3, but got: 4"):
        model.tau_max(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 3, but got: 4"):
        model.tau_max(q_valid, qdot_too_large)


def test_rigid_contact_acceleration_valid_and_too_large_q_or_qdot_or_qddot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    (
        q_valid,
        qdot_valid,
        qddot_valid,
        q_too_large,
        qdot_too_large,
        qddot_too_large,
    ) = generate_q_qdot_qddot_vectors(model)

    # q, qdot and qddot valid
    model.rigid_contact_acceleration(q_valid, qdot_valid, qddot_valid, 0, 0)
    # qdot and qddot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.rigid_contact_acceleration(q_too_large, qdot_valid, qddot_valid, 0, 0)
    # q and qddot valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.rigid_contact_acceleration(q_valid, qdot_too_large, qddot_valid, 0, 0)
    # q and qdot valid but qddot not valid
    with pytest.raises(ValueError, match="Length of qddot size should be: 4, but got: 5"):
        model.rigid_contact_acceleration(q_valid, qdot_valid, qddot_too_large, 0, 0)


def test_markers_jacobian_valid_and_too_large_q_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, q_too_large = generate_q_vectors(model)

    # q valid
    model.markers_jacobian(q_valid)
    # q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.markers_jacobian(q_too_large)


def test_soft_contact_forces_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.soft_contact_forces(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.soft_contact_forces(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.soft_contact_forces(q_valid, qdot_too_large)


def test_contact_forces_valid_and_too_large_q_or_qdot_or_tau_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)
    tau_valid, tau_too_large = generate_tau_activations_vectors(model)

    # q, qdot and tau valid
    model.contact_forces(q_valid, qdot_valid, tau_valid)
    # qdot and tau valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.contact_forces(q_too_large, qdot_valid, tau_valid)
    # q and tau valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.contact_forces(q_valid, qdot_too_large, tau_valid)
    # q and qdot valid but tau not valid
    with pytest.raises(ValueError, match="Length of tau size should be: 4, but got: 5"):
        model.contact_forces(q_valid, qdot_valid, tau_too_large)


def test_passive_joint_torque_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.passive_joint_torque(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.passive_joint_torque(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.passive_joint_torque(q_valid, qdot_too_large)


def test_ligament_joint_torque_valid_and_too_large_q_or_qdot_input(model):
    biorbd_model_path = (
        bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
    )
    model = BiorbdModel(biorbd_model_path)
    q_valid, qdot_valid, q_too_large, qdot_too_large = generate_q_and_qdot_vectors(model)

    # q and qdot valid
    model.ligament_joint_torque(q_valid, qdot_valid)
    # qdot valid but q not valid
    with pytest.raises(ValueError, match="Length of q size should be: 4, but got: 5"):
        model.ligament_joint_torque(q_too_large, qdot_valid)
    # q valid but qdot not valid
    with pytest.raises(ValueError, match="Length of qdot size should be: 4, but got: 5"):
        model.ligament_joint_torque(q_valid, qdot_too_large)
