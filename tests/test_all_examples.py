import numpy as np

from .utils import TestUtils


def test__acados__cube():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")
    module.prepare_ocp(biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod", n_shooting=10, tf=2)


def test__acados__pendulum():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/acados/pendulum.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/pendulum.bioMod", n_shooting=41, final_time=3
    )


def test__acados__static_arm():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/acados/static_arm.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/arm26.bioMod",
        final_time=2,
        x_warm=None,
        n_shooting=51,
        use_sx=False,
        n_threads=6,
    )


def test__getting_started__custom_bounds():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_bounds.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", n_shooting=30, final_time=2
    )


def test__getting_started__custom_constraints():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_constraint.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )


def test__getting_started__custom_dynamics():
    bioptim_folder = TestUtils.bioptim_folder()
    dynamics = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_dynamics.py")
    dynamics.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )


def test__getting_started__custom_initial_guess():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_initial_guess.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", n_shooting=30, final_time=2
    )


def test__getting_started__custom_objectives():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_objectives.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )


def test__getting_started__custom_parameters():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_parameters.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=np.array([0, 0, -9.81]),
        target_m=20,
    )


def test__getting_started__custom_phase_transitions():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_phase_transitions.py")
    module.prepare_ocp(biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod")


def test__getting_started__custom_plotting():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_plotting.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod", final_time=2, n_shooting=50
    )


def test__getting_started__example_cyclic_movement():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_cyclic_movement.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
        n_shooting=30,
        final_time=2,
        loop_from_constraint=True,
    )


def test__getting_started__example_external_forces():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_external_forces.py")
    module.prepare_ocp(biorbd_model_path=bioptim_folder + "/examples/getting_started/cube_with_forces.bioMod")


def test__getting_started__example_inequality_constraint():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_inequality_constraint.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
        max_bound=np.inf,
        mu=0.2,
    )


def test__getting_started__example_mapping():
    bioptim_folder = TestUtils.bioptim_folder()
    TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_mapping.py")


def test__getting_started__example_multiphase():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")
    module.prepare_ocp(biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", long_optim=True)


def test__getting_started__example_optimal_time():
    bioptim_folder = TestUtils.bioptim_folder()
    TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_optimal_time.py")


def test__getting_started__example_save_and_load():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_save_and_load.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        n_threads=4,
    )


def test__getting_started__example_simulation():
    bioptim_folder = TestUtils.bioptim_folder()
    TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_optimal_time.py")


def test__getting_started__pendulum():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/getting_started/pendulum.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod", final_time=3, n_shooting=100
    )


def test__moving_horizon_estimation__mhe():
    bioptim_folder = TestUtils.bioptim_folder()
    TestUtils.load_module(bioptim_folder + "/examples/moving_horizon_estimation/mhe.py")
    # Todo: Complete when the example is more clear


def test__muscle_driven_ocp__muscle_activations_tracker():
    bioptim_folder = TestUtils.bioptim_folder()
    TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/muscle_activations_tracker.py")


def test__muscle_driven_ocp__muscle_excitations_tracker():
    bioptim_folder = TestUtils.bioptim_folder()
    TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py")


def test__muscle_driven_ocp__static_arm():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/static_arm.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod",
        final_time=3,
        n_shooting=50,
        weight=1000,
    )


def test__muscle_driven_ocp__static_arm_with_contact():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/static_arm_with_contact.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/muscle_driven_ocp/arm26_with_contact.bioMod",
        final_time=3,
        n_shooting=50,
        weight=1000,
    )


def test__muscle_driven_with_contact__contact_forces_inequality_constraint_muscle():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(
        bioptim_folder + "/examples/muscle_driven_with_contact/contact_forces_inequality_constraint_muscle.py"
    )
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder
        + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
        max_bound=np.inf,
    )


def test__muscle_driven_with_contact__contact_forces_inequality_constraint_muscle_excitations():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(
        bioptim_folder
        + "/examples/muscle_driven_with_contact/contact_forces_inequality_constraint_muscle_excitations.py"
    )
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder
        + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
    )


def test__muscle_driven_with_contact__muscle_activations_contacts_tracker():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(
        bioptim_folder + "/examples/muscle_driven_with_contact/muscle_activations_contacts_tracker.py"
    )


def test__optimal_time_ocp__multiphase_time_constraint():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/optimal_time_ocp/multiphase_time_constraint.py")
    final_time = [2, 5, 4]
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    ns = [20, 30, 20]
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/optimal_time_ocp/cube.bioMod",
        final_time=final_time,
        time_min=time_min,
        time_max=time_max,
        n_shooting=ns,
    )


def test__optimal_time_ocp__pendulum_min_time_Lagrange():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/optimal_time_ocp/pendulum_min_time_Lagrange.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/optimal_time_ocp/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
    )


def test__optimal_time_ocp__pendulum_min_time_Mayer():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/optimal_time_ocp/pendulum_min_time_Mayer.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/optimal_time_ocp/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
    )


def test__optimal_time_ocp__time_constraint():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/optimal_time_ocp/time_constraint.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/optimal_time_ocp/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        time_min=0.6,
        time_max=1,
    )


def test__symmetrical_torque_driven_ocp__symmetry_by_constraint():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/symmetrical_torque_driven_ocp/symmetry_by_constraint.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
    )


def test__symmetrical_torque_driven_ocp__symmetry_by_mapping():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/symmetrical_torque_driven_ocp/symmetry_by_mapping.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
    )


def test__torque_driven_ocp__maximize_predicted_height_CoM():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/maximize_predicted_height_CoM.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=20,
        use_actuators=False,
        objective_name="MINIMIZE_COM_VELOCITY",
        com_constraints=True,
    )


def test__torque_driven_ocp__spring_load():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/spring_load.py")
    module.prepare_ocp(biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/mass_point.bioMod")


def test__torque_driven_ocp__track_markers_2D_pendulum():
    bioptim_folder = TestUtils.bioptim_folder()
    TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/track_markers_2D_pendulum.py")


def test__torque_driven_ocp__track_markers_with_torque_actuators():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(
        bioptim_folder + "/examples/torque_driven_ocp/track_markers_with_torque_actuators.py"
    )
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/cube.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=1,
    )


def test__torque_driven_ocp__trampo_quaternions():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/trampo_quaternions.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/TruncAnd2Arm_Quaternion.bioMod",
        n_shooting=5,
        final_time=0.25,
    )


def test__track__track_marker_on_segment():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/track/track_marker_on_segment.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/track/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
    )


def test__track__track_segment_on_rt():
    bioptim_folder = TestUtils.bioptim_folder()
    module = TestUtils.load_module(bioptim_folder + "/examples/track/track_segment_on_rt.py")
    module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/track/cube_and_line.bioMod",
        n_shooting=30,
        final_time=1,
    )
