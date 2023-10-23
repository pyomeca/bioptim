import os
import pytest
import numpy as np
from casadi import MX, DM, vertcat, Function
import biorbd_casadi as biorbd
from bioptim import (
    MultiBiorbdModel,
    BiMappingList,
    BoundsList,
)
from ..utils import TestUtils


def test_biorbd_model_import():
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = "/models/triple_pendulum.bioMod"
    biorbd_model_path_modified_inertia = "/models/triple_pendulum_modified_inertia.bioMod"
    MultiBiorbdModel((bioptim_folder + biorbd_model_path, bioptim_folder + biorbd_model_path_modified_inertia))

    MultiBiorbdModel(
        (
            biorbd.Model(bioptim_folder + biorbd_model_path),
            biorbd.Model(bioptim_folder + biorbd_model_path_modified_inertia),
        )
    )

    with pytest.raises(
        ValueError, match="The models must be a 'str', 'biorbd.Model', 'bioptim.BiorbdModel'" " or a tuple of those"
    ):
        MultiBiorbdModel([1])


# TODO: test all cases with models containing at least on element (muscles, contacts, ...)
def test_biorbd_model():
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = "/models/triple_pendulum.bioMod"
    biorbd_model_path_modified_inertia = "/models/triple_pendulum_modified_inertia.bioMod"
    models = MultiBiorbdModel(
        bio_model=(bioptim_folder + biorbd_model_path, bioptim_folder + biorbd_model_path_modified_inertia),
        extra_bio_models=(
            bioptim_folder + biorbd_model_path_modified_inertia,
            bioptim_folder + biorbd_model_path_modified_inertia,
            bioptim_folder + biorbd_model_path_modified_inertia,
        ),
    )

    assert models.nb_q == 6
    assert models.nb_qdot == 6
    assert models.nb_qddot == 6
    assert models.nb_root == 2
    assert models.nb_tau == 6
    assert models.nb_quaternions == 0
    assert models.nb_segments == 6
    assert models.nb_muscles == 0
    assert models.nb_soft_contacts == 0
    assert models.nb_markers == 12
    assert models.nb_rigid_contacts == 0
    assert models.nb_contacts == 0
    assert models.nb_dof == 6
    assert models.nb_models == 2
    assert models.nb_extra_models == 3

    assert models.name_dof == ("Seg1_RotX", "Seg2_RotX", "Seg3_RotX", "Seg1_RotX", "Seg2_RotX", "Seg3_RotX")
    assert models.contact_names == ()
    assert models.soft_contact_names == ()
    assert models.marker_names == (
        "marker_1",
        "marker_2",
        "marker_3",
        "marker_4",
        "marker_5",
        "marker_6",
        "marker_1",
        "marker_2",
        "marker_3",
        "marker_4",
        "marker_5",
        "marker_6",
    )
    assert models.muscle_names == ()

    variable_mappings = BiMappingList()
    variable_mappings.add("q", to_second=[0, 1, 2, 3, 4, 5], to_first=[0, 1, 2, 3, 4, 5])
    variable_mappings.add("qdot", to_second=[0, 1, 2, 3, 4, 5], to_first=[0, 1, 2, 3, 4, 5])
    variable_mappings.add("qddot", to_second=[0, 1, 2, 3, 4, 5], to_first=[0, 1, 2, 3, 4, 5])
    variable_mappings.add("tau", to_second=[None, 0, 1, None, 0, 2], to_first=[1, 2, 5])

    # model_deep_copied = models.deep_copy() # TODO: Fix deep copy
    models.copy()
    models.serialize()

    models.set_gravity(np.array([0, 0, -3]))
    TestUtils.assert_equal(models.gravity, np.array([0, 0, -3, 0, 0, -3]))
    models.set_gravity(np.array([0, 0, -9.81]))

    with pytest.raises(NotImplementedError, match="segment_index is not implemented for MultiBiorbdModel"):
        segment_index = models.segment_index("Seg1")

    assert len(models.segments) == 6
    assert isinstance(models.segments, tuple)
    assert isinstance(models.segments[0], biorbd.biorbd.Segment)

    TestUtils.assert_equal(
        # one of the last ouput of BiorbdModel which is not a MX but a biorbd object
        models.homogeneous_matrices_in_global(np.array([1, 2, 3]), 0, 0).to_mx(),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.540302, -0.841471, 0.0],
                [0.0, 0.841471, 0.540302, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    TestUtils.assert_equal(
        models.homogeneous_matrices_in_child(4),
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]]),
    )

    TestUtils.assert_equal(models.mass, np.array([3, 3]))
    TestUtils.assert_equal(
        models.center_of_mass(np.array([1, 2, 3, 4, 5, 6])),
        np.array([-5.000000e-04, 8.433844e-01, -1.764446e-01, -5.000000e-04, -3.232674e-01, 1.485815e00]),
    )
    TestUtils.assert_equal(
        models.center_of_mass_velocity(np.array([1, 2.1, 3, 4.1, 5, 6.1]), np.array([1, 2.1, 3, 4.1, 5, 6])),
        np.array([0.0, 0.425434, 0.638069, 0.0, -12.293126, 0.369492]),
    )
    TestUtils.assert_equal(
        models.center_of_mass_acceleration(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ),
        np.array([0.0, 0.481652, 6.105858, 0.0, -33.566971, -126.795179]),
    )

    mass_matrices = models.mass_matrix(np.array([1, 2.1, 3, 4.1, 5, 6.1]))
    assert len(mass_matrices) == 2
    TestUtils.assert_equal(
        mass_matrices[0],
        np.array(
            [
                [2.711080e00, 3.783457e-01, 4.243336e-01],
                [3.783457e-01, 9.999424e-01, -2.881681e-05],
                [4.243336e-01, -2.881681e-05, 9.543311e-01],
            ]
        ),
    )
    TestUtils.assert_equal(
        mass_matrices[1],
        np.array([[9.313616, 5.580191, 2.063886], [5.580191, 4.791997, 1.895999], [2.063886, 1.895999, 0.945231]]),
    )

    nonlinear_effects = models.non_linear_effects(np.array([1, 2.1, 3, 4.1, 5, 6.1]), np.array([1, 2.1, 3, 4.1, 5, 6]))
    assert len(nonlinear_effects) == 2
    TestUtils.assert_equal(
        nonlinear_effects[0],
        np.array([38.817401, -1.960653, -1.259441]),
    )
    TestUtils.assert_equal(
        nonlinear_effects[1],
        np.array([322.060726, -22.147881, -20.660836]),
    )

    TestUtils.assert_equal(
        models.angular_momentum(np.array([1, 2.1, 3, 4.1, 5, 6.1]), np.array([1, 2.1, 3, 4.1, 5, 6])),
        np.array([3.001448e00, 0.000000e00, -2.168404e-19, 2.514126e01, 3.252607e-19, 0.000000e00]),
        decimal=5,
    )

    TestUtils.assert_equal(
        models.reshape_qdot(np.array([1, 2.1, 3, 4.1, 5, 6.1]), np.array([1, 2.1, 3, 4.1, 5, 6]), 1),
        np.array([1.0, 2.1, 3.0, 4.1, 5.0, 6.0]),
    )

    TestUtils.assert_equal(
        models.segment_angular_velocity(np.array([1, 2.1, 3, 4.1, 5, 6.1]), np.array([1, 2.1, 3, 4.1, 5, 6]), 1),
        np.array([3.1, 0.0, 0.0, 9.1, 0.0, 0.0]),
    )

    assert models.soft_contact(0, 0) == []  # TODO: Fix soft contact (biorbd call error)

    with pytest.raises(RuntimeError, match="Close the actuator model before calling torqueMax"):
        models.torque(
            np.array([3.1, 0.0, 0.0, 9.1, 0.0, 0.0]),
            np.array([3.1, 0.0, 0.0, 9.1, 0.0, 0.0]),
            np.array([3.1, 0.0, 0.0, 9.1, 0.0, 0.0]),
        )  # TODO: Fix torque (Close the actuator model before calling torqueMax)

    TestUtils.assert_equal(
        models.forward_dynamics_free_floating_base(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 0.0, 0.0, 9.1, 0.0, 0.0]),
        ),
        np.array([-14.750679, -36.596107]),
    )

    TestUtils.assert_equal(
        models.forward_dynamics(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
        ),
        np.array([-16.092093, 9.049853, 10.570878, -121.783105, 154.820759, -20.664452]),
    )

    TestUtils.assert_equal(
        models.constrained_forward_dynamics(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
        ),
        np.array([-16.092093, 9.049853, 10.570878, -121.783105, 154.820759, -20.664452]),
    )

    with pytest.raises(NotImplementedError, match="External forces are not implemented yet for MultiBiorbdModel."):
        models.constrained_forward_dynamics(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
        )

    TestUtils.assert_equal(
        models.inverse_dynamics(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
        ),
        np.array([4.844876e01, 2.121037e-01, 1.964626e00, 4.165226e02, 3.721585e01, 1.906986e00]),
        decimal=5,
    )

    TestUtils.assert_equal(
        models.contact_forces_from_constrained_forward_dynamics(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
            None,
        ),
        np.array([0.0, 0.0]),
    )

    with pytest.raises(NotImplementedError, match="External forces are not implemented yet for MultiBiorbdModel."):
        models.contact_forces_from_constrained_forward_dynamics(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
        )

    TestUtils.assert_equal(
        models.qdot_from_impact(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
        ),
        np.array([1.0, 2.1, 3.0, 4.1, 5.0, 6.0]),
    )

    TestUtils.assert_equal(
        models.muscle_activation_dot(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
        ),
        np.array([], dtype=np.float64),
    )

    TestUtils.assert_equal(
        models.muscle_joint_torque(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
        ),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    TestUtils.assert_equal(
        models.markers(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
        )[0],
        np.array([0.0, 0.0, 0.0]),
    )

    TestUtils.assert_equal(
        models.marker(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            index=1,
        ),
        np.array([0.0, 0.841471, -0.540302]),
    )

    assert models.marker_index("marker_3") == 2

    markers_velocities = models.marker_velocities(
        np.array([1, 2.1, 3, 4.1, 5, 6.1]),
        np.array([1, 2.1, 3, 4.1, 5, 6]),
    )
    assert isinstance(markers_velocities, list)

    TestUtils.assert_equal(
        markers_velocities[0],
        np.array([0.0, 0.0, 0.0]),
    )
    TestUtils.assert_equal(
        markers_velocities[1],
        np.array([0.0, 0.540302, 0.841471]),
    )

    with pytest.raises(RuntimeError, match="All dof must have their actuators set"):
        models.tau_max(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
        )  # TODO: add an actuator model (AnaisFarr will do it when her PR will be merged)

    # TODO: add a model with a contact to test this function
    # rigid_contact_acceleration = models.rigid_contact_acceleration(q, qdot, qddot, 0, 0)

    TestUtils.assert_equal(
        models.soft_contact_forces(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
        ),
        np.array([], dtype=np.float64),
    )

    with pytest.raises(
        NotImplementedError, match="reshape_fext_to_fcontact is not implemented yet for MultiBiorbdModel"
    ):
        models.reshape_fext_to_fcontact(np.array([1, 2.1, 3, 4.1, 5, 6.1]))

    # this test doesn't properly test the function, but it's the best we can do for now
    # we should add a quaternion to the model to test it
    # anyway it's tested elsewhere.
    TestUtils.assert_equal(
        models.normalize_state_quaternions(
            np.array([1, 2.1, 3, 4.1, 5, 6.1, 1, 2.1, 3, 4.1, 5, 6.6]),
        ),
        np.array([1.0, 2.1, 3.0, 4.1, 5.0, 6.1, 1.0, 2.1, 3.0, 4.1, 5.0, 6.6]),
    )

    TestUtils.assert_equal(
        models.contact_forces(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
            None,
        ),
        np.array([0.0, 0.0]),
    )

    with pytest.raises(
        NotImplementedError, match="contact_forces is not implemented yet with external_forces for MultiBiorbdModel"
    ):
        models.contact_forces(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
            np.array([3.1, 1, 2, 9.1, 1, 2]),
        )

    TestUtils.assert_equal(
        models.passive_joint_torque(
            np.array([1, 2.1, 3, 4.1, 5, 6.1]),
            np.array([1, 2.1, 3, 4.1, 5, 6]),
        ),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    q_mapping = models._q_mapping(variable_mappings)
    qdot_mapping = models._q_mapping(variable_mappings)
    qddot_mapping = models._q_mapping(variable_mappings)
    tau_mapping = models._q_mapping(variable_mappings)

    np.testing.assert_equal(q_mapping["q"].to_first.map_idx, [0, 1, 2, 3, 4, 5])
    np.testing.assert_equal(qdot_mapping["qdot"].to_first.map_idx, [0, 1, 2, 3, 4, 5])
    np.testing.assert_equal(qddot_mapping["qddot"].to_first.map_idx, [0, 1, 2, 3, 4, 5])
    np.testing.assert_equal(tau_mapping["tau"].to_first.map_idx, [1, 2, 5])

    bounds_from_ranges = BoundsList()
    bounds_from_ranges["q"] = models.bounds_from_ranges("q", variable_mappings)
    bounds_from_ranges["qdot"] = models.bounds_from_ranges("qdot", variable_mappings)

    for key in bounds_from_ranges.keys():
        if key == "q":
            expected = [
                [-31.41592654, -31.41592654, -31.41592654],
                [-9.42477796, -9.42477796, -9.42477796],
                [-9.42477796, -9.42477796, -9.42477796],
                [-31.41592654, -31.41592654, -31.41592654],
                [-9.42477796, -9.42477796, -9.42477796],
                [-9.42477796, -9.42477796, -9.42477796],
            ]
        elif key == "qdot":
            expected = [
                [-31.41592654, -31.41592654, -31.41592654],
                [-31.41592654, -31.41592654, -31.41592654],
                [-31.41592654, -31.41592654, -31.41592654],
                [-31.41592654, -31.41592654, -31.41592654],
                [-31.41592654, -31.41592654, -31.41592654],
                [-31.41592654, -31.41592654, -31.41592654],
            ]
        else:
            raise NotImplementedError("Wrong value")

        np.testing.assert_almost_equal(bounds_from_ranges[key].min, DM(np.array(expected)), decimal=5)

        assert models.variable_index("q", 0) == range(0, 3)
        assert models.variable_index("qdot", 1) == range(3, 6)
        assert models.variable_index("tau", 0) == range(0, 3)
        assert models.variable_index("qddot", 1) == range(3, 6)
        assert models.variable_index("qddot_joints", 0) == range(0, 2)
        assert models.variable_index("qddot_root", 1) == range(1, 2)
        assert models.variable_index("contact", 0) == range(0, 0)
        assert models.variable_index("markers", 0) == range(0, 6)

        variable_name = "wrong"
        with pytest.raises(
            ValueError,
            match=f"The variable must be 'q', 'qdot', 'qddot', 'tau', 'contact' or 'markers'"
            f" and {variable_name} was sent.",
        ):
            models.variable_index(variable_name, 0)

        assert models.global_variable_id("q", 0, 1) == 1
        assert models.global_variable_id("qdot", 1, 0) == 3
        assert models.global_variable_id("tau", 0, 1) == 1
        assert models.global_variable_id("qddot", 1, 0) == 3
        assert models.global_variable_id("qddot_joints", 0, 1) == 1
        assert models.global_variable_id("qddot_root", 1, 0) == 1

        with pytest.raises(IndexError, match="range object index out of range"):
            models.global_variable_id("contact", 0, 1)

        assert models.global_variable_id("markers", 0, 1) == 1

        local_id, model_id = models.local_variable_id("q", 2)
        assert local_id == 2
        assert model_id == 0

        local_id, model_id = models.local_variable_id("qdot", 5)
        assert local_id == 2
        assert model_id == 1

        local_id, model_id = models.local_variable_id("tau", 2)
        assert local_id == 2
        assert model_id == 0

        local_id, model_id = models.local_variable_id("qddot", 5)
        assert local_id == 2
        assert model_id == 1

        local_id, model_id = models.local_variable_id("qddot_joints", 2)
        assert local_id == 0
        assert model_id == 1

        local_id, model_id = models.local_variable_id("qddot_root", 1)
        assert local_id == 0
        assert model_id == 1

        local_id, model_id = models.local_variable_id("markers", 2)
        assert local_id == 2
        assert model_id == 0

        local_id, model_id = models.local_variable_id("segment", 2)
        assert local_id == 2
        assert model_id == 0

        variable_name = "wrong"
        with pytest.raises(
            ValueError,
            match=f"The variable must be 'q', 'qdot', 'qddot', 'tau', 'contact' or 'markers'"
            f" and {variable_name} was sent.",
        ):
            models.local_variable_id(variable_name, 0)
