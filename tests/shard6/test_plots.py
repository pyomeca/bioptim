import pytest
import platform

import numpy as np

from bioptim import InterpolationType, PhaseDynamics, Solver, ControlType, OdeSolver, TorqueBiorbdModel
from ..utils import TestUtils


if platform.system() != "Linux":
    pytest.skip("Skipping tests on non-Linux platforms", allow_module_level=True)

np.random.seed(0)


# Test 1: custom plots
@pytest.fixture(scope="module")
def plots_generator_for_custom_plotting():
    from bioptim.examples.getting_started import custom_plotting as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )
    sol = ocp.solve(Solver.IPOPT())
    return sol.graphs(show_now=False, automatically_organize=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_custom_plotting_0(plots_generator_for_custom_plotting):
    return plots_generator_for_custom_plotting[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_custom_plotting_1(plots_generator_for_custom_plotting):
    return plots_generator_for_custom_plotting[1]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_custom_plotting_2(plots_generator_for_custom_plotting):
    return plots_generator_for_custom_plotting[2]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_custom_plotting_3(plots_generator_for_custom_plotting):
    return plots_generator_for_custom_plotting[3]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_custom_plotting_4(plots_generator_for_custom_plotting):
    return plots_generator_for_custom_plotting[4]


# Test 2 : bounds
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__custom_constraints__constant():
    from bioptim.examples.toy_examples.feature_examples import custom_bounds as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../../models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        interpolation_type=InterpolationType.CONSTANT,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__custom_constraints__constant_with_first_and_last():
    from bioptim.examples.toy_examples.feature_examples import custom_bounds as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../../models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        interpolation_type=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__custom_constraints__linear():
    from bioptim.examples.toy_examples.feature_examples import custom_bounds as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../../models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        interpolation_type=InterpolationType.LINEAR,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__custom_constraints__each_frame():
    from bioptim.examples.toy_examples.feature_examples import custom_bounds as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../../models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        interpolation_type=InterpolationType.EACH_FRAME,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__custom_constraints__spline():
    from bioptim.examples.toy_examples.feature_examples import custom_bounds as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../../models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        interpolation_type=InterpolationType.SPLINE,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__custom_constraints__custom():
    from bioptim.examples.toy_examples.feature_examples import custom_bounds as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        interpolation_type=InterpolationType.CUSTOM,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


# Test 3: variable mapping
@pytest.fixture(scope="module")
def plots_generator_for_mapping():
    from bioptim.examples.symmetrical_torque_driven_ocp import symmetry_by_mapping as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/cubeSym.bioMod",
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_mapping_0(plots_generator_for_mapping):
    return plots_generator_for_mapping[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_mapping_1(plots_generator_for_mapping):
    return plots_generator_for_mapping[1]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_mapping_2(plots_generator_for_mapping):
    return plots_generator_for_mapping[2]


# Test 4: multi-phase
@pytest.fixture(scope="module")
def plots_generator_for_multi_phase():
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/cube.bioMod",
        long_optim=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_multi_phase_0(plots_generator_for_multi_phase):
    return plots_generator_for_multi_phase[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_multi_phase_1(plots_generator_for_multi_phase):
    return plots_generator_for_multi_phase[1]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_multi_phase_2(plots_generator_for_multi_phase):
    return plots_generator_for_multi_phase[2]


# Test 5: optimized time
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__optimal_time_ocp__min_time_mayer__constant():
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=30,
        max_time=0.5,
        weight=-1,
        control_type=ControlType.CONSTANT,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__optimal_time_ocp__min_time_mayer__constant_with_last():
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=30,
        max_time=0.5,
        weight=-1,
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__optimal_time_ocp__min_time_mayer__linear_continuous():
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=30,
        max_time=0.5,
        weight=-1,
        control_type=ControlType.LINEAR_CONTINUOUS,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


# TODO: Add variable scaling when the plots are fixed for this case


# Test 6: solvers
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__pendulum__rk1():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=OdeSolver.RK1(),
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__pendulum__rk2():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=OdeSolver.RK2(n_integration_steps=3),
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__pendulum__rk4():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=OdeSolver.RK4(),
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__pendulum__trapezoidal():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=OdeSolver.TRAPEZOIDAL(),
        control_type=ControlType.LINEAR_CONTINUOUS,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__pendulum__irk():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=OdeSolver.IRK(),
        use_sx=False,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__pendulum__collocation_legendre():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=OdeSolver.COLLOCATION(method="legendre"),
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__getting_started__pendulum__collocation_radau():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
        ode_solver=OdeSolver.COLLOCATION(method="radau"),
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


# TODO: add multi-biorbd model when the plots are fixed for this case


# Test 7: phase transition with different number of variables
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__torque_driven_ocp__phase_transition_uneven_variable_number_by_mapping():
    from bioptim.examples.torque_driven_ocp import (
        phase_transition_uneven_variable_number_by_mapping as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/double_pendulum.bioMod",
        biorbd_model_path_with_translations=bioptim_folder + "examples/models/double_pendulum_with_translations.bioMod",
        n_shooting=(5, 5),
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


# Test 8: track markers
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__torque_driven_ocp__track_markers_2D_pendulum():
    from bioptim.examples.torque_driven_ocp import track_markers_2D_pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)
    model_path = bioptim_folder + "examples/models/pendulum.bioMod"

    np.random.seed(42)
    n_shooting = 30
    ocp = ocp_module.prepare_ocp(
        bio_model=TorqueBiorbdModel(model_path),
        final_time=2,
        n_shooting=n_shooting,
        markers_ref=np.random.rand(3, 2, n_shooting + 1),
        tau_ref=np.random.rand(2, n_shooting),
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


# Test 9: quaternion
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test__torque_driven_ocp__example_quaternions():
    from bioptim.examples.torque_driven_ocp import example_quaternions as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/trunk_and_2arm_quaternion.bioMod",
        n_shooting=5,
        final_time=0.25,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)[0]


# Test 10: muscle-driven with contact
@pytest.fixture(scope="module")
def plots_generator_for_muscle_driven_with_contact():
    from bioptim.examples.muscle_driven_with_contact import (
        contact_forces_inequality_constraint_muscle as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=50,
        max_bound=np.inf,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)
    return sol.graphs(show_now=False, show_bounds=True, automatically_organize=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_muscle_with_contact_0(plots_generator_for_muscle_driven_with_contact):
    return plots_generator_for_muscle_driven_with_contact[0]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_muscle_with_contact_1(plots_generator_for_muscle_driven_with_contact):
    return plots_generator_for_muscle_driven_with_contact[1]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_muscle_with_contact_2(plots_generator_for_muscle_driven_with_contact):
    return plots_generator_for_muscle_driven_with_contact[2]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_muscle_with_contact_3(plots_generator_for_muscle_driven_with_contact):
    return plots_generator_for_muscle_driven_with_contact[3]


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_muscle_with_contact_4(plots_generator_for_muscle_driven_with_contact):
    return plots_generator_for_muscle_driven_with_contact[4]
