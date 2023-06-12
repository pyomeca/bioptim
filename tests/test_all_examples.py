import os
import pytest
import numpy as np
from bioptim import InterpolationType


def test__acados__cube():
    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(biorbd_model_path=bioptim_folder + "/models/cube.bioMod", n_shooting=10, tf=2)


def test__acados__pendulum():
    from bioptim.examples.acados import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod", n_shooting=41, final_time=3)


def test__acados__static_arm():
    from bioptim.examples.acados import static_arm as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/arm26.bioMod",
        final_time=2,
        x_warm=None,
        n_shooting=51,
        use_sx=False,
        n_threads=6,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__custom_bounds(assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_bounds as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__custom_constraints(assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_constraint as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod", assume_phase_dynamics=assume_phase_dynamics
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__custom_dynamics(assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_dynamics as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod", assume_phase_dynamics=assume_phase_dynamics
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("interpolation", [*InterpolationType])
@pytest.mark.parametrize("random", [True, False])
def test__getting_started__custom_initial_guess(interpolation, random, assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_initial_guess as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        random_init=random,
        initial_guess=interpolation,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__custom_objectives(assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_objectives as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__custom_parameters(assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_parameters as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        optim_gravity=True,
        optim_mass=True,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=np.array([0, 0, -9.81]),
        target_m=20,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__custom_phase_transitions(assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_phase_transitions as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod", assume_phase_dynamics=assume_phase_dynamics
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__custom_plotting(assume_phase_dynamics):
    from bioptim.examples.getting_started import custom_plotting as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__example_cyclic_movement(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_cyclic_movement as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        loop_from_constraint=True,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def test__getting_started__example_external_forces():
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod")


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__example_inequality_constraint(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_inequality_constraint as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../torque_driven_ocp/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
        max_bound=np.inf,
        mu=0.2,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def test__getting_started__example_mapping():
    from bioptim.examples.getting_started import example_mapping as ocp_module


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__example_multiphase(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=True,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])  # Shouldn't it be only False ?
def test__getting_started__example_multinode_constraints(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_multinode_constraints as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        assume_phase_dynamics=assume_phase_dynamics,
        n_shootings=(8, 8, 8),
    )


def test__getting_started__example_multinode_objective():
    from bioptim.examples.getting_started import example_multinode_objective as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=10,
    )

    with pytest.raises(RuntimeError, match="multinode_objectives cannot be used with multi-threading, set n_threads=1"):
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
            final_time=1,
            n_shooting=10,
            n_threads=3,
        )

    with pytest.raises(
        RuntimeError, match="multinode_objectives cannot be used with assume_phase_dynamics=True, set it to false"
    ):
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
            final_time=1,
            n_shooting=10,
            assume_phase_dynamics=True,
        )


def test__getting_started__example_optimal_time():
    from bioptim.examples.getting_started import example_optimal_time as ocp_module


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__example_save_and_load(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_save_and_load as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        n_threads=4,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def test__getting_started__example_simulation():
    from bioptim.examples.getting_started import example_optimal_time as ocp_module


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__pendulum(assume_phase_dynamics):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def test__moving_horizon_estimation__mhe():
    from bioptim.examples.moving_horizon_estimation import mhe as ocp_module


def test__muscle_driven_ocp__muscle_activations_tracker():
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module


def test__muscle_driven_ocp__muscle_excitations_tracker():
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__muscle_driven_ocp__static_arm(assume_phase_dynamics):
    from bioptim.examples.muscle_driven_ocp import static_arm as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/arm26.bioMod",
        final_time=3,
        n_shooting=50,
        weight=1000,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def test__muscle_driven_with_contact__muscle_activations_contacts_tracker():
    from bioptim.examples.muscle_driven_with_contact import muscle_activations_contacts_tracker as ocp_module


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__optimal_time_ocp__multiphase_time_constraint(assume_phase_dynamics):
    from bioptim.examples.optimal_time_ocp import multiphase_time_constraint as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    final_time = (2, 5, 4)
    time_min = (1, 3, 0.1)
    time_max = (2, 4, 0.8)
    ns = (20, 30, 20)
    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        final_time=final_time,
        time_min=time_min,
        time_max=time_max,
        n_shooting=ns,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__optimal_time_ocp__pendulum_min_time_Lagrange(assume_phase_dynamics):
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Lagrange as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__optimal_time_ocp__pendulum_min_time_Mayer(assume_phase_dynamics):
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__optimal_time_ocp__time_constraint(assume_phase_dynamics):
    from bioptim.examples.optimal_time_ocp import time_constraint as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        time_min=0.6,
        time_max=1,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__symmetrical_torque_driven_ocp__symmetry_by_constraint(assume_phase_dynamics):
    from bioptim.examples.symmetrical_torque_driven_ocp import symmetry_by_constraint as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cubeSym.bioMod",
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__symmetrical_torque_driven_ocp__symmetry_by_mapping(assume_phase_dynamics):
    from bioptim.examples.symmetrical_torque_driven_ocp import symmetry_by_mapping as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cubeSym.bioMod", assume_phase_dynamics=assume_phase_dynamics
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__maximize_predicted_height_CoM(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import maximize_predicted_height_CoM as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=20,
        use_actuators=False,
        objective_name="MINIMIZE_COM_VELOCITY",
        com_constraints=True,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__multi_biorbd_model(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/triple_pendulum.bioMod",
        biorbd_model_path_modified_inertia=bioptim_folder + "/models/triple_pendulum_modified_inertia.bioMod",
        n_shooting=40,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__phase_transition_uneven_variable_number_by_mapping(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import phase_transition_uneven_variable_number_by_mapping as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/double_pendulum.bioMod",
        biorbd_model_path_with_translations=bioptim_folder + "/models/double_pendulum_with_translations.bioMod",
        n_shooting=(5, 5),
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__phase_transition_uneven_variable_number_by_bounds(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import phase_transition_uneven_variable_number_by_bounds as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path_with_translations=bioptim_folder + "/models/double_pendulum_with_translations.bioMod",
        n_shooting=(5, 5),
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__spring_load(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import spring_load as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(biorbd_model_path=bioptim_folder + "/models/mass_point.bioMod")


def test__torque_driven_ocp__track_markers_2D_pendulum():
    from bioptim.examples.torque_driven_ocp import track_markers_2D_pendulum as ocp_module


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__track_markers_with_torque_actuators(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=1,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__trampo_quaternions(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import trampo_quaternions as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/TruncAnd2Arm_Quaternion.bioMod",
        n_shooting=5,
        final_time=0.25,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__minimize_segment_velocity(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import example_minimize_segment_velocity as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/triple_pendulum.bioMod",
        n_shooting=5,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__track__track_marker_on_segment(assume_phase_dynamics):
    from bioptim.examples.track import track_marker_on_segment as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__track__track_segment_on_rt(assume_phase_dynamics):
    from bioptim.examples.track import track_segment_on_rt as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=1,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__track__track_vector_orientation(assume_phase_dynamics):
    from bioptim.examples.track import track_vector_orientation as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=1,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__getting_started__example_variable_scaling(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_variable_scaling as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1 / 10,
        n_shooting=30,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__torque_driven_ocp__torque_activation_driven(assume_phase_dynamics):
    from bioptim.examples.torque_driven_ocp import torque_activation_driven as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_2dof_2contacts.bioMod",
        final_time=2,
        n_shooting=30,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test__inverse_optimal_control__double_pendulum_torque_driven_IOCP(assume_phase_dynamics):
    from bioptim.examples.inverse_optimal_control import double_pendulum_torque_driven_IOCP as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_module.prepare_ocp(
        weights=[0.4, 0.3, 0.3],
        coefficients=[1, 1, 1],
        biorbd_model_path=bioptim_folder + "/models/double_pendulum.bioMod",
    )
