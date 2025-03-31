import numpy as np
import pytest

from bioptim import InterpolationType, PhaseDynamics, OdeSolver, DefectType, ContactType
from ..utils import TestUtils


## examples/acados
def test__acados__cube():
    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )


def test__acados__pendulum():
    from bioptim.examples.acados import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=41,
        final_time=3,
        expand_dynamics=True,
    )


def test__acados__static_arm():
    from bioptim.examples.acados import static_arm as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/arm26.bioMod",
        final_time=2,
        x_warm=None,
        n_shooting=51,
        use_sx=False,
        n_threads=6,
        expand_dynamics=False,
    )


## examples/getting_started
def test__getting_started__custom_bounds():
    from bioptim.examples.getting_started import custom_bounds as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
    )


def test__getting_started__custom_constraints():
    from bioptim.examples.getting_started import custom_constraint as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__custom_dynamics():
    from bioptim.examples.getting_started import custom_dynamics as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


@pytest.mark.parametrize("interpolation", [*InterpolationType])
@pytest.mark.parametrize("random", [True, False])
def test__getting_started__custom_initial_guess(interpolation, random):
    from bioptim.examples.getting_started import custom_initial_guess as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        random_init=random,
        initial_guess=interpolation,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
    )


def test__getting_started__custom_objectives():
    from bioptim.examples.getting_started import custom_objectives as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__custom_parameters():
    from bioptim.examples.getting_started import custom_parameters as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    target_g = np.zeros((3, 1))
    target_g[2] = -9.81
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
        target_g=target_g,
        target_m=20,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__custom_phase_transitions():
    from bioptim.examples.getting_started import custom_phase_transitions as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__custom_plotting():
    from bioptim.examples.getting_started import custom_plotting as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__example_continuity_as_objective():
    from bioptim.examples.getting_started import example_continuity_as_objective as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp_first_pass(
        biorbd_model_path=bioptim_folder + "/models/pendulum_maze.bioMod",
        final_time=1,
        n_shooting=100,
        state_continuity_weight=1_000_000,
    )


def test__getting_started__example_cyclic_movement():
    from bioptim.examples.getting_started import example_cyclic_movement as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        loop_from_constraint=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__example_external_forces():
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
        expand_dynamics=False,
    )


def test__getting_started__example_inequality_constraint():
    from bioptim.examples.getting_started import (
        example_inequality_constraint as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../torque_driven_ocp/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
        max_bound=np.inf,
        mu=0.2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


# todo: Add example_joint_acceleration_driven.py?


def test__getting_started__example_mapping():
    pass


def test__getting_started__example_multinode_constraints():
    from bioptim.examples.getting_started import (
        example_multinode_constraints as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        n_shootings=(8, 8, 8),
        expand_dynamics=False,
    )


def test__getting_started__example_multinode_objective():
    from bioptim.examples.getting_started import (
        example_multinode_objective as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=10,
        expand_dynamics=False,
    )

    with pytest.raises(
        ValueError,
        match="Valid values for setting the cx is 0, 1 or 2. If you reach this error message, you probably tried to "
        "add more penalties than available in a multinode constraint. You can try to split the constraints "
        "into more penalties or use phase_dynamics=PhaseDynamics.ONE_PER_NODE",
    ):
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
            final_time=1,
            n_shooting=10,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            expand_dynamics=False,
        )


def test__getting_started__example_multiphase():
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__example_multiphase_different_ode_solvers():
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=[OdeSolver.RK1(), OdeSolver.RK4(), OdeSolver.COLLOCATION()],
    )

    with pytest.raises(
        RuntimeError,
        match="ode_solver should be built an instance of OdeSolver",
    ):
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
            long_optim=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            expand_dynamics=False,
            ode_solver=["hello", "world", "there"],
        )


# todo: Add example_multistart.py?


def test__getting_started__example_optimal_time():
    pass


def test__getting_started__example_simulation():
    pass


# todo: Add example_variable_scaling.py?


def test__getting_started__pendulum():
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__pendulum_constrained_states_controls():
    from bioptim.examples.getting_started import (
        pendulum_constrained_states_controls as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


## examples/moving_horizon_estimation
def test__moving_horizon_estimation__mhe():
    pass


def test__muscle_driven_ocp__muscle_activations_tracker():
    pass


def test__muscle_driven_ocp__muscle_excitations_tracker():
    pass


def test__muscle_driven_ocp__static_arm():
    from bioptim.examples.muscle_driven_ocp import static_arm as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/arm26.bioMod",
        final_time=3,
        n_shooting=50,
        weight=1000,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__muscle_driven_with_contact__muscle_activations_contacts_tracker():
    pass


def test__optimal_time_ocp__multiphase_time_constraint():
    from bioptim.examples.optimal_time_ocp import (
        multiphase_time_constraint as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

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
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__optimal_time_ocp__pendulum_min_time_Mayer():
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__optimal_time_ocp__time_constraint():
    from bioptim.examples.optimal_time_ocp import time_constraint as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        time_min=0.6,
        time_max=1,
        expand_dynamics=False,
    )


## torque_driven_ocp_folder
def test__symmetrical_torque_driven_ocp__symmetry_by_constraint():
    from bioptim.examples.symmetrical_torque_driven_ocp import (
        symmetry_by_constraint as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cubeSym.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__symmetrical_torque_driven_ocp__symmetry_by_mapping():
    from bioptim.examples.symmetrical_torque_driven_ocp import symmetry_by_mapping as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cubeSym.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__maximize_predicted_height_CoM():
    from bioptim.examples.torque_driven_ocp import (
        maximize_predicted_height_CoM as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=20,
        use_actuators=False,
        objective_name="MINIMIZE_COM_VELOCITY",
        com_constraints=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__multi_biorbd_model():
    from bioptim.examples.torque_driven_ocp import (
        example_multi_biorbd_model as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/triple_pendulum.bioMod",
        biorbd_model_path_modified_inertia=bioptim_folder + "/models/triple_pendulum_modified_inertia.bioMod",
        n_shooting=40,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__phase_transition_uneven_variable_number_by_mapping():
    from bioptim.examples.torque_driven_ocp import (
        phase_transition_uneven_variable_number_by_mapping as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/double_pendulum.bioMod",
        biorbd_model_path_with_translations=bioptim_folder + "/models/double_pendulum_with_translations.bioMod",
        n_shooting=(5, 5),
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__phase_transition_uneven_variable_number_by_bounds():
    from bioptim.examples.torque_driven_ocp import (
        phase_transition_uneven_variable_number_by_bounds as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path_with_translations=bioptim_folder + "/models/double_pendulum_with_translations.bioMod",
        n_shooting=(5, 5),
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__spring_load():
    from bioptim.examples.torque_driven_ocp import spring_load as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    for scenario in range(8):
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/mass_point.bioMod",
            expand_dynamics=False,
            scenario=scenario,
        )


def test__track__optimal_estimation():
    from bioptim.examples.track import optimal_estimation as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    markers_ref = np.array(
        [
            [
                [
                    0.0,
                    0.00407935,
                    0.0081587,
                    0.01223806,
                    0.01631741,
                    0.02039676,
                    0.02447611,
                    0.02855547,
                    0.03263482,
                    0.03671417,
                    0.04079352,
                    0.04487288,
                    0.04895223,
                    0.05303158,
                    0.05711093,
                    0.06119029,
                    0.06526964,
                    0.06934899,
                    0.07342834,
                    0.07750769,
                    0.08158705,
                    0.0856664,
                    0.08974575,
                    0.0938251,
                    0.09790446,
                    0.10198381,
                    0.10606316,
                    0.11014251,
                    0.11422187,
                    0.11830122,
                    0.12238057,
                ],
                [
                    0.0,
                    0.03465495,
                    0.06908073,
                    0.10323732,
                    0.13708508,
                    0.17058474,
                    0.20369751,
                    0.23638512,
                    0.26860985,
                    0.30033457,
                    0.33152283,
                    0.3621389,
                    0.39214777,
                    0.42151526,
                    0.45020801,
                    0.47819357,
                    0.50544041,
                    0.53191798,
                    0.55759673,
                    0.58244817,
                    0.60644491,
                    0.62956067,
                    0.65177034,
                    0.67305,
                    0.69337695,
                    0.71272974,
                    0.73108822,
                    0.74843354,
                    0.76474816,
                    0.78001592,
                    0.79422202,
                ],
                [
                    0.0,
                    0.01803043,
                    0.03646401,
                    0.05528183,
                    0.07446445,
                    0.09399195,
                    0.11384396,
                    0.13399968,
                    0.15443793,
                    0.17513713,
                    0.19607536,
                    0.21723041,
                    0.23857975,
                    0.26010061,
                    0.28177001,
                    0.30356476,
                    0.32546149,
                    0.34743672,
                    0.36946686,
                    0.39152826,
                    0.41359719,
                    0.43564997,
                    0.45766289,
                    0.47961233,
                    0.50147474,
                    0.52322669,
                    0.54484489,
                    0.56630626,
                    0.58758791,
                    0.6086672,
                    0.62952184,
                ],
                [
                    0.0,
                    0.04860603,
                    0.09738604,
                    0.1462811,
                    0.19523211,
                    0.24417992,
                    0.29306536,
                    0.34182934,
                    0.39041296,
                    0.43875752,
                    0.48680467,
                    0.53449643,
                    0.58177529,
                    0.62858429,
                    0.6748671,
                    0.72056805,
                    0.76563226,
                    0.81000571,
                    0.85363525,
                    0.89646873,
                    0.93845506,
                    0.97954424,
                    1.01968748,
                    1.05883723,
                    1.09694723,
                    1.13397262,
                    1.16986996,
                    1.20459728,
                    1.2381142,
                    1.2703819,
                    1.30136328,
                ],
                [
                    1.0,
                    1.00351444,
                    1.00589981,
                    1.00715834,
                    1.00729374,
                    1.00631122,
                    1.00421745,
                    1.00102056,
                    0.99673014,
                    0.99135723,
                    0.98491429,
                    0.97741518,
                    0.96887517,
                    0.95931089,
                    0.94874033,
                    0.93718279,
                    0.92465889,
                    0.91119051,
                    0.89680078,
                    0.88151404,
                    0.86535582,
                    0.84835278,
                    0.83053272,
                    0.8119245,
                    0.792558,
                    0.77246412,
                    0.7516747,
                    0.73022251,
                    0.70814117,
                    0.68546509,
                    0.66222945,
                ],
                [
                    1.0,
                    1.03409004,
                    1.06682183,
                    1.0981576,
                    1.12806141,
                    1.1564992,
                    1.18343885,
                    1.20885021,
                    1.23270517,
                    1.25497763,
                    1.2756436,
                    1.29468121,
                    1.31207072,
                    1.32779457,
                    1.34183741,
                    1.35418608,
                    1.36482967,
                    1.3737595,
                    1.38096916,
                    1.38645452,
                    1.39021368,
                    1.39224706,
                    1.39255731,
                    1.39114939,
                    1.38803049,
                    1.38321005,
                    1.37669977,
                    1.36851354,
                    1.35866746,
                    1.34717979,
                    1.33407089,
                ],
                [
                    1.0,
                    1.01746552,
                    1.03420512,
                    1.05020211,
                    1.06544078,
                    1.0799064,
                    1.09358529,
                    1.10646477,
                    1.11853325,
                    1.12978019,
                    1.14019613,
                    1.14977271,
                    1.15850269,
                    1.16637993,
                    1.17339941,
                    1.17955726,
                    1.18485074,
                    1.18927824,
                    1.1928393,
                    1.1955346,
                    1.19736596,
                    1.19833635,
                    1.19844986,
                    1.19771172,
                    1.19612828,
                    1.19370699,
                    1.19045644,
                    1.18638626,
                    1.18150721,
                    1.17583108,
                    1.16937071,
                ],
                [
                    0.0,
                    0.04860603,
                    0.09738604,
                    0.1462811,
                    0.19523211,
                    0.24417992,
                    0.29306536,
                    0.34182934,
                    0.39041296,
                    0.43875752,
                    0.48680467,
                    0.53449643,
                    0.58177529,
                    0.62858429,
                    0.6748671,
                    0.72056805,
                    0.76563226,
                    0.81000571,
                    0.85363525,
                    0.89646873,
                    0.93845506,
                    0.97954424,
                    1.01968748,
                    1.05883723,
                    1.09694723,
                    1.13397262,
                    1.16986996,
                    1.20459728,
                    1.2381142,
                    1.2703819,
                    1.30136328,
                ],
            ],
            [
                [
                    0.0,
                    0.00407935,
                    0.0081587,
                    0.01223806,
                    0.01631741,
                    0.02039676,
                    0.02447611,
                    0.02855547,
                    0.03263482,
                    0.03671417,
                    0.04079352,
                    0.04487288,
                    0.04895223,
                    0.05303158,
                    0.05711093,
                    0.06119029,
                    0.06526964,
                    0.06934899,
                    0.07342834,
                    0.07750769,
                    0.08158705,
                    0.0856664,
                    0.08974575,
                    0.0938251,
                    0.09790446,
                    0.10198381,
                    0.10606316,
                    0.11014251,
                    0.11422187,
                    0.11830122,
                    0.12238057,
                ],
                [
                    0.0,
                    0.01803043,
                    0.03646401,
                    0.05528183,
                    0.07446445,
                    0.09399195,
                    0.11384396,
                    0.13399968,
                    0.15443793,
                    0.17513713,
                    0.19607536,
                    0.21723041,
                    0.23857975,
                    0.26010061,
                    0.28177001,
                    0.30356476,
                    0.32546149,
                    0.34743672,
                    0.36946686,
                    0.39152826,
                    0.41359719,
                    0.43564997,
                    0.45766289,
                    0.47961233,
                    0.50147474,
                    0.52322669,
                    0.54484489,
                    0.56630626,
                    0.58758791,
                    0.6086672,
                    0.62952184,
                ],
                [
                    -1.0,
                    -0.99573179,
                    -0.99108611,
                    -0.98606371,
                    -0.98066584,
                    -0.97489422,
                    -0.9687511,
                    -0.96223919,
                    -0.9553617,
                    -0.94812232,
                    -0.94052522,
                    -0.93257502,
                    -0.92427684,
                    -0.91563623,
                    -0.90665921,
                    -0.89735224,
                    -0.88772219,
                    -0.87777638,
                    -0.86752256,
                    -0.85696884,
                    -0.84612376,
                    -0.83499624,
                    -0.82359555,
                    -0.81193133,
                    -0.80001357,
                    -0.7878526,
                    -0.77545903,
                    -0.76284381,
                    -0.75001816,
                    -0.73699356,
                    -0.72378172,
                ],
                [
                    -1.0,
                    -0.98178071,
                    -0.9627808,
                    -0.94301994,
                    -0.9225188,
                    -0.90129904,
                    -0.87938326,
                    -0.85679498,
                    -0.83355859,
                    -0.80969937,
                    -0.78524338,
                    -0.76021749,
                    -0.73464932,
                    -0.7085672,
                    -0.68200013,
                    -0.65497776,
                    -0.62753034,
                    -0.59968865,
                    -0.57148404,
                    -0.54294828,
                    -0.51411362,
                    -0.48501267,
                    -0.45567841,
                    -0.4261441,
                    -0.39644329,
                    -0.36660972,
                    -0.3366773,
                    -0.30668006,
                    -0.27665212,
                    -0.24662757,
                    -0.21664045,
                ],
                [
                    0.0,
                    0.01760887,
                    0.03477835,
                    0.05149118,
                    0.06773069,
                    0.08348085,
                    0.09872627,
                    0.11345222,
                    0.12764468,
                    0.14129029,
                    0.15437645,
                    0.16689127,
                    0.17882364,
                    0.1901632,
                    0.20090037,
                    0.21102637,
                    0.22053323,
                    0.22941378,
                    0.2376617,
                    0.24527149,
                    0.25223848,
                    0.25855888,
                    0.2642297,
                    0.26924887,
                    0.27361512,
                    0.27732807,
                    0.2803882,
                    0.28279683,
                    0.28455616,
                    0.28566922,
                    0.2861399,
                ],
                [
                    0.0,
                    0.03155995,
                    0.06308366,
                    0.09453495,
                    0.12587773,
                    0.15707603,
                    0.18809411,
                    0.21889644,
                    0.24944779,
                    0.27971325,
                    0.30965829,
                    0.3392488,
                    0.36845116,
                    0.39723223,
                    0.42555945,
                    0.45340084,
                    0.48072508,
                    0.50750151,
                    0.53370022,
                    0.55929205,
                    0.58424863,
                    0.60854244,
                    0.63214684,
                    0.65503609,
                    0.6771854,
                    0.69857095,
                    0.71916993,
                    0.73896058,
                    0.7579222,
                    0.77603521,
                    0.79328117,
                ],
                [
                    -1.0,
                    -0.98220227,
                    -0.96446646,
                    -0.94681059,
                    -0.92925256,
                    -0.91181014,
                    -0.89450095,
                    -0.87734244,
                    -0.86035185,
                    -0.84354621,
                    -0.82694229,
                    -0.81055662,
                    -0.79440543,
                    -0.77850462,
                    -0.76286978,
                    -0.74751615,
                    -0.7324586,
                    -0.71771159,
                    -0.7032892,
                    -0.68920505,
                    -0.67547233,
                    -0.66210376,
                    -0.64911159,
                    -0.63650757,
                    -0.62430291,
                    -0.61250834,
                    -0.60113399,
                    -0.59018949,
                    -0.57968387,
                    -0.56962555,
                    -0.56002238,
                ],
                [
                    -1.0,
                    -0.98178071,
                    -0.9627808,
                    -0.94301994,
                    -0.9225188,
                    -0.90129904,
                    -0.87938326,
                    -0.85679498,
                    -0.83355859,
                    -0.80969937,
                    -0.78524338,
                    -0.76021749,
                    -0.73464932,
                    -0.7085672,
                    -0.68200013,
                    -0.65497776,
                    -0.62753034,
                    -0.59968865,
                    -0.57148404,
                    -0.54294828,
                    -0.51411362,
                    -0.48501267,
                    -0.45567841,
                    -0.4261441,
                    -0.39644329,
                    -0.36660972,
                    -0.3366773,
                    -0.30668006,
                    -0.27665212,
                    -0.24662757,
                    -0.21664045,
                ],
            ],
            [
                [
                    0.0,
                    0.00418837,
                    0.0083593,
                    0.01251279,
                    0.01664884,
                    0.02076745,
                    0.02486863,
                    0.02895236,
                    0.03301866,
                    0.03706752,
                    0.04109894,
                    0.04511292,
                    0.04910947,
                    0.05308857,
                    0.05705024,
                    0.06099447,
                    0.06492126,
                    0.06883061,
                    0.07272253,
                    0.076597,
                    0.08045404,
                    0.08429363,
                    0.08811579,
                    0.09192052,
                    0.0957078,
                    0.09947764,
                    0.10323005,
                    0.10696501,
                    0.11068254,
                    0.11438263,
                    0.11806528,
                ],
                [
                    -1.0,
                    -0.99524672,
                    -0.98938181,
                    -0.9824075,
                    -0.97432749,
                    -0.96514701,
                    -0.95487271,
                    -0.94351273,
                    -0.93107666,
                    -0.91757554,
                    -0.90302182,
                    -0.88742938,
                    -0.87081347,
                    -0.85319074,
                    -0.83457916,
                    -0.81499804,
                    -0.79446799,
                    -0.77301091,
                    -0.75064991,
                    -0.72740934,
                    -0.70331473,
                    -0.67839275,
                    -0.65267118,
                    -0.62617888,
                    -0.59894574,
                    -0.57100267,
                    -0.54238149,
                    -0.51311499,
                    -0.48323676,
                    -0.45278124,
                    -0.42178359,
                ],
                [
                    0.0,
                    -0.00934115,
                    -0.01826035,
                    -0.02674033,
                    -0.03476444,
                    -0.04231663,
                    -0.04938153,
                    -0.05594439,
                    -0.0619912,
                    -0.0675086,
                    -0.07248398,
                    -0.07690547,
                    -0.08076195,
                    -0.08404304,
                    -0.08673919,
                    -0.08884161,
                    -0.09034233,
                    -0.09123418,
                    -0.09151083,
                    -0.09116679,
                    -0.0901974,
                    -0.08859884,
                    -0.08636816,
                    -0.08350325,
                    -0.08000286,
                    -0.07586662,
                    -0.07109499,
                    -0.0656893,
                    -0.05965175,
                    -0.05298537,
                    -0.04569405,
                ],
                [
                    -1.0,
                    -1.00877624,
                    -1.01600146,
                    -1.02166062,
                    -1.02574077,
                    -1.02823109,
                    -1.02912286,
                    -1.02840948,
                    -1.02608652,
                    -1.02215166,
                    -1.01660475,
                    -1.00944778,
                    -1.00068489,
                    -0.99032236,
                    -0.97836859,
                    -0.96483412,
                    -0.94973158,
                    -0.9330757,
                    -0.91488327,
                    -0.89517314,
                    -0.87396617,
                    -0.85128523,
                    -0.82715513,
                    -0.80160264,
                    -0.7746564,
                    -0.74634693,
                    -0.71670653,
                    -0.6857693,
                    -0.65357105,
                    -0.62014924,
                    -0.58554293,
                ],
                [
                    0.0,
                    0.03495282,
                    0.0700365,
                    0.10521028,
                    0.14043326,
                    0.17566444,
                    0.21086281,
                    0.24598736,
                    0.28099717,
                    0.31585142,
                    0.35050951,
                    0.38493105,
                    0.41907595,
                    0.45290444,
                    0.48637718,
                    0.51945524,
                    0.55210021,
                    0.58427422,
                    0.61594001,
                    0.64706094,
                    0.67760109,
                    0.70752527,
                    0.73679909,
                    0.76538898,
                    0.79326226,
                    0.82038717,
                    0.84673292,
                    0.87226971,
                    0.89696881,
                    0.92080256,
                    0.94374444,
                ],
                [
                    -1.0,
                    -0.96448227,
                    -0.92770461,
                    -0.88971,
                    -0.85054308,
                    -0.81025002,
                    -0.76887852,
                    -0.72647773,
                    -0.68309816,
                    -0.63879164,
                    -0.59361125,
                    -0.54761125,
                    -0.500847,
                    -0.45337487,
                    -0.40525222,
                    -0.35653727,
                    -0.30728904,
                    -0.2575673,
                    -0.20743243,
                    -0.15694541,
                    -0.10616768,
                    -0.05516112,
                    -0.00398789,
                    0.04728958,
                    0.09860872,
                    0.14990686,
                    0.20112138,
                    0.25218971,
                    0.30304951,
                    0.35363868,
                    0.40389557,
                ],
                [
                    0.0,
                    0.0214233,
                    0.04341685,
                    0.06595716,
                    0.08901998,
                    0.11258036,
                    0.13661266,
                    0.1610906,
                    0.18598731,
                    0.2112753,
                    0.23692659,
                    0.26291266,
                    0.28920453,
                    0.31577282,
                    0.34258774,
                    0.36961916,
                    0.39683662,
                    0.42420943,
                    0.45170665,
                    0.47929715,
                    0.50694965,
                    0.53463279,
                    0.56231513,
                    0.58996521,
                    0.6175516,
                    0.64504291,
                    0.67240788,
                    0.6996154,
                    0.72663452,
                    0.75343455,
                    0.77998511,
                ],
                [
                    -1.0,
                    -1.00877624,
                    -1.01600146,
                    -1.02166062,
                    -1.02574077,
                    -1.02823109,
                    -1.02912286,
                    -1.02840948,
                    -1.02608652,
                    -1.02215166,
                    -1.01660475,
                    -1.00944778,
                    -1.00068489,
                    -0.99032236,
                    -0.97836859,
                    -0.96483412,
                    -0.94973158,
                    -0.9330757,
                    -0.91488327,
                    -0.89517314,
                    -0.87396617,
                    -0.85128523,
                    -0.82715513,
                    -0.80160264,
                    -0.7746564,
                    -0.74634693,
                    -0.71670653,
                    -0.6857693,
                    -0.65357105,
                    -0.62014924,
                    -0.58554293,
                ],
            ],
        ]
    )

    ocp_module.prepare_optimal_estimation(
        biorbd_model_path=bioptim_folder + "/models/cube_6dofs.bioMod",
        time_ref=0.039998243355170666,
        n_shooting=30,
        markers_ref=markers_ref,
    )


def test__torque_driven_ocp__track_markers_2D_pendulum():
    pass


def test__torque_driven_ocp__track_markers_with_torque_actuators():
    from bioptim.examples.torque_driven_ocp import (
        track_markers_with_torque_actuators as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=1,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__example_quaternions():
    from bioptim.examples.torque_driven_ocp import example_quaternions as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/trunk_and_2arm_quaternion.bioMod",
        n_shooting=5,
        final_time=0.25,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__minimize_segment_velocity():
    from bioptim.examples.torque_driven_ocp import (
        example_minimize_segment_velocity as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/triple_pendulum.bioMod",
        n_shooting=5,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__track__track_marker_on_segment():
    from bioptim.examples.track import track_marker_on_segment as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__track__track_segment_on_rt():
    from bioptim.examples.track import track_segment_on_rt as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=1,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__getting_started__example_variable_scaling():
    from bioptim.examples.getting_started import example_variable_scaling as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1 / 10,
        n_shooting=30,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__torque_driven_ocp__torque_activation_driven():
    from bioptim.examples.torque_driven_ocp import (
        torque_activation_driven as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_2dof_2contacts.bioMod",
        final_time=2,
        n_shooting=30,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def test__inverse_optimal_control__double_pendulum_torque_driven_IOCP():
    from bioptim.examples.inverse_optimal_control import (
        double_pendulum_torque_driven_IOCP as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        weights=[0.4, 0.3, 0.3],
        coefficients=[1, 1, 1],
        biorbd_model_path=bioptim_folder + "/models/double_pendulum.bioMod",
        expand_dynamics=False,
    )


def test__contact_and_muscle_forces_example():
    from bioptim.examples.muscle_driven_with_contact import (
        contact_forces_inequality_constraint_muscle as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
        max_bound=np.inf,
        expand_dynamics=False,
    )


def test__contact_and_muscle_forces_example_excitation():
    from bioptim.examples.muscle_driven_with_contact import (
        contact_forces_inequality_constraint_muscle_excitations as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
        expand_dynamics=False,
    )


def test_min_max_example():
    from bioptim.examples.torque_driven_ocp import (
        minimize_maximum_torque_by_extra_parameter as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        bio_model_path=bioptim_folder + "/models/double_pendulum.bioMod",
    )


def test_custom_model():
    from bioptim.examples.custom_model.main import main as ocp_module

    ocp_module()


@pytest.mark.parametrize("defect_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS])
@pytest.mark.parametrize("contact_type", [[ContactType.RIGID_EXPLICIT], [ContactType.RIGID_IMPLICIT]])
def test_contact_forces_inverse_dynamics_constraint_muscle(defect_type, contact_type):
    from bioptim.examples.muscle_driven_with_contact import (
        contact_forces_inverse_dynamics_constraint_muscle as ocp_module,
    )
    bioptim_folder = TestUtils.module_folder(ocp_module)

    if defect_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS and ContactType.RIGID_EXPLICIT in contact_type:
        with pytest.raises(NotImplementedError, match="Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet"):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts_1muscle.bioMod",
                phase_time=0.3,
                n_shooting=10,
                defect_type=defect_type,
                contact_type=contact_type,
            )
        return

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        defect_type=defect_type,
        contact_type=contact_type,
    )


def test_contact_forces_inverse_dynamics_constraint_muscle_fdot():
    from bioptim.examples.muscle_driven_with_contact import (
        contact_forces_inverse_dynamics_constraint_muscle_fdot as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
    )


def test_contact_forces_inverse_dynamics_soft_contacts_muscle():
    from bioptim.examples.muscle_driven_with_contact import (
        contact_forces_inverse_dynamics_soft_contacts_muscle as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2soft_contacts_1muscle.bioMod",
        phase_time=1,
        n_shooting=100,
    )