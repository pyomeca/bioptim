import pytest
import numpy as np
import numpy.testing as npt
from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    BoundsList,
    ParameterList,
    InterpolationType,
    InitialGuessList,
    OdeSolver,
    MagnitudeType,
    PhaseDynamics,
    VariableScaling,
)

from tests.utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_double_update_bounds_and_init(phase_dynamics):
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model = BiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
    nq = bio_model.nb_q
    ns = 10

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=phase_dynamics)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot
    u_init = InitialGuessList()
    u_init["tau"] = [0] * bio_model.nb_tau
    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        ns,
        1.0,
        x_init=x_init,
        u_init=u_init,
    )

    x_bounds = BoundsList()
    x_bounds["q"] = -np.ones((nq, 1)), np.ones((nq, 1))
    x_bounds["qdot"] = -np.ones((nq, 1)), np.ones((nq, 1))
    u_bounds = BoundsList()
    u_bounds["tau"] = -2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.array([[0.1] + [-1] * (nq * 2) * (ns + 1) + [-2] * nq * ns]).T
    npt.assert_almost_equal(ocp.bounds_vectors[0], expected)
    expected = np.array([[0.1] + [1] * (nq * 2) * (ns + 1) + [2] * nq * ns]).T
    npt.assert_almost_equal(ocp.bounds_vectors[1], expected)

    x_init = InitialGuessList()
    x_init["q"] = 0.5 * np.ones((nq, 1))
    x_init["qdot"] = 0.5 * np.ones((nq, 1))
    u_init = InitialGuessList()
    u_init["tau"] = -0.5 * np.ones((nq, 1))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.1] + [0.5] * (nq * 2) * (ns + 1) + [-0.5] * nq * ns]).T
    npt.assert_almost_equal(ocp.init_vector, expected)

    x_bounds = BoundsList()
    x_bounds["q"] = -2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))
    x_bounds["qdot"] = -2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))
    u_bounds = BoundsList()
    u_bounds["tau"] = -4.0 * np.ones((nq, 1)), 4.0 * np.ones((nq, 1))
    ocp.update_bounds(x_bounds=x_bounds)
    ocp.update_bounds(u_bounds=u_bounds)

    expected = np.array([[0.1] + [-2] * (nq * 2) * (ns + 1) + [-4] * nq * ns]).T
    npt.assert_almost_equal(ocp.bounds_vectors[0], expected)
    expected = np.array([[0.1] + [2] * (nq * 2) * (ns + 1) + [4] * nq * ns]).T
    npt.assert_almost_equal(ocp.bounds_vectors[1], expected)

    x_init = InitialGuessList()
    x_init["q"] = 0.25 * np.ones((nq, 1))
    x_init["qdot"] = 0.25 * np.ones((nq, 1))
    u_init = InitialGuessList()
    u_init["tau"] = -0.25 * np.ones((nq, 1))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.1] + [0.25] * (nq * 2) * (ns + 1) + [-0.25] * nq * ns]).T
    npt.assert_almost_equal(ocp.init_vector, expected)

    with pytest.raises(RuntimeError, match="x_init should be built from a InitialGuessList"):
        ocp.update_initial_guess(x_bounds, u_bounds)
    with pytest.raises(RuntimeError, match="u_init should be built from a InitialGuessList"):
        ocp.update_initial_guess(None, u_bounds)
    with pytest.raises(RuntimeError, match="x_bounds should be built from a BoundsList"):
        ocp.update_bounds(x_init, u_init)
    with pytest.raises(RuntimeError, match="u_bounds should be built from a BoundsList"):
        ocp.update_bounds(None, u_init)
    with pytest.raises(
        ValueError, match="bad_key is not a state variable, please check for typos in the declaration of x_bounds"
    ):
        x_bounds = BoundsList()
        x_bounds.add("bad_key", [1, 2])
        ocp.update_bounds(x_bounds, u_bounds)
    with pytest.raises(
        ValueError, match="bad_key is not a control variable, please check for typos in the declaration of u_bounds"
    ):
        x_bounds = BoundsList()
        x_bounds["q"] = -np.ones((nq, 1)), np.ones((nq, 1))
        u_bounds = BoundsList()
        u_bounds.add("bad_key", [1, 2])
        ocp.update_bounds(x_bounds, u_bounds)
    with pytest.raises(
        ValueError, match="bad_key is not a state variable, please check for typos in the declaration of x_init"
    ):
        x_init = InitialGuessList()
        x_init.add("bad_key", [1, 2])
        ocp.update_initial_guess(x_init, u_init)
    with pytest.raises(
        ValueError, match="bad_key is not a control variable, please check for typos in the declaration of u_init"
    ):
        x_init = InitialGuessList()
        x_init["q"] = 0.5 * np.ones((nq, 1))
        u_init = InitialGuessList()
        u_init.add("bad_key", [1, 2])
        ocp.update_initial_guess(x_init, u_init)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_update_bounds_and_init_with_param(phase_dynamics):
    def my_parameter_function(bio_model, value, extra_value):
        new_gravity = MX.zeros(3, 1)
        new_gravity[2] = value + extra_value
        bio_model.set_gravity(new_gravity)

    bio_model = BiorbdModel(TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod")
    nq = bio_model.nb_q
    ns = 10
    g_min, g_max, g_init = -10, -6, -8

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=phase_dynamics)

    parameters = ParameterList(use_sx=False)
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    parameters.add(
        "gravity_z",
        my_parameter_function,
        size=1,
        extra_value=1,
        scaling=VariableScaling("gravity_z", np.array([1])),
    )
    parameter_bounds.add("gravity_z", min_bound=[g_min], max_bound=[g_max], interpolation=InterpolationType.CONSTANT)
    parameter_init["gravity_z"] = [g_init]

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        ns,
        1.0,
        parameters=parameters,
        parameter_init=parameter_init,
        parameter_bounds=parameter_bounds,
    )

    # Before modifying
    expected = np.array([[0.1] + [-np.inf] * (nq * 2) * (ns + 1) + [-np.inf] * nq * ns + [g_min]]).T
    npt.assert_almost_equal(ocp.bounds_vectors[0], expected)
    expected = np.array([[0.1] + [np.inf] * (nq * 2) * (ns + 1) + [np.inf] * nq * ns + [g_max]]).T
    npt.assert_almost_equal(ocp.bounds_vectors[1], expected)

    x_bounds = BoundsList()
    x_bounds["q"] = -np.ones((nq, 1)), np.ones((nq, 1))
    x_bounds["qdot"] = -np.ones((nq, 1)), np.ones((nq, 1))
    u_bounds = BoundsList()
    u_bounds["tau"] = -2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.array([[0.1] + [-1] * (nq * 2) * (ns + 1) + [-2] * nq * ns + [g_min]]).T
    npt.assert_almost_equal(ocp.bounds_vectors[0], expected)
    expected = np.array([[0.1] + [1] * (nq * 2) * (ns + 1) + [2] * nq * ns + [g_max]]).T
    npt.assert_almost_equal(ocp.bounds_vectors[1], expected)

    x_init = InitialGuessList()
    x_init["q"] = 0.5 * np.ones((nq, 1))
    x_init["qdot"] = 0.5 * np.ones((nq, 1))
    u_init = InitialGuessList()
    u_init["tau"] = -0.5 * np.ones((nq, 1))
    ocp.update_initial_guess(x_init, u_init)

    expected = np.array([[0.1] + [0.5] * (nq * 2) * (ns + 1) + [-0.5] * nq * ns + [g_init]]).T
    npt.assert_almost_equal(ocp.init_vector, expected)

    # Try on parameters too
    parameter_bounds = BoundsList()
    parameter_bounds.add(
        "gravity_z", min_bound=[g_min * 2], max_bound=[g_max * 2], interpolation=InterpolationType.CONSTANT
    )
    parameter_init = InitialGuessList()
    parameter_init["gravity_z"] = [g_init * 2]
    ocp.update_bounds(parameter_bounds=parameter_bounds)
    ocp.update_initial_guess(parameter_init=parameter_init)

    expected = np.array([[0.1] + [-1] * (nq * 2) * (ns + 1) + [-2] * nq * ns + [g_min * 2]]).T
    npt.assert_almost_equal(ocp.bounds_vectors[0], expected)
    expected = np.array([[0.1] + [1] * (nq * 2) * (ns + 1) + [2] * nq * ns + [g_max * 2]]).T
    npt.assert_almost_equal(ocp.bounds_vectors[1], expected)

    expected = np.array([[0.1] + [0.5] * (nq * 2) * (ns + 1) + [-0.5] * nq * ns + [g_init * 2]]).T
    npt.assert_almost_equal(ocp.init_vector, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("interpolation", [*InterpolationType])
def test_update_noised_init_rk4(interpolation, phase_dynamics):
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=phase_dynamics)

    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot
    u_init = InitialGuessList()
    u_init["tau"] = [0] * bio_model.nb_tau
    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting=ns,
        phase_time=phase_time,
        ode_solver=OdeSolver.RK4(),
        x_init=x_init,
        u_init=u_init,
    )

    # Path constraint and control path constraints
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["q"][2, -1] = 1.57
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * ntau, [tau_max] * ntau

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if interpolation == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [tau_init] * ntau
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.LINEAR:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.EACH_FRAME:
        x = np.zeros((nq * 2, ns + 1))
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.random.random((nq + nqdot, ns + 1))
        u = np.random.random((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.random.random((nq + nqdot, ns + 1))
        u = np.random.random((ntau, ns))
    elif interpolation == InterpolationType.SPLINE:
        # Bound , assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))
    elif interpolation == InterpolationType.CUSTOM:
        x = list()
        x.append(lambda y: np.array([0] * nq))
        x.append(lambda y: np.array([0] * nqdot))
        u = lambda y: np.array([tau_init] * ntau)
    else:
        raise NotImplementedError("Interpolation type not implemented")

    np.random.seed(0)
    x_init = InitialGuessList()
    if interpolation == InterpolationType.CUSTOM:
        x_init.add("q", x[0], interpolation=interpolation, t=t)
        x_init.add("qdot", x[1], interpolation=interpolation, t=t)
    else:
        x_init.add("q", x[:nq], interpolation=interpolation, t=t)
        x_init.add("qdot", x[nq:], interpolation=interpolation, t=t)
    x_init.add_noise(
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        **extra_params_x,
    )
    u_init = InitialGuessList()
    u_init.add("tau", u, interpolation=interpolation, t=t)
    u_init.add_noise(
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        **extra_params_u,
    )

    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x_init, u_init)
    else:
        ocp.update_initial_guess(x_init, u_init)

        if interpolation == InterpolationType.CONSTANT:
            expected = np.array(
                [
                    [0.33333333],
                    [0.00292881],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.01291136],
                    [0.00583576],
                    [-0.01464717],
                    [0.53482051],
                    [0.41798243],
                    [0.37593374],
                    [0.0061658],
                    [-0.00249651],
                    [0.03665925],
                    [-0.53905199],
                    [0.34954208],
                    [-0.04840646],
                    [0.00269299],
                    [0.0],
                    [1.67],
                    [0.0],
                    [0.0],
                    [0.0],
                    [-1.5269023],
                    [1.77867567],
                    [-0.94177755],
                    [0.55968409],
                    [0.08739329],
                    [1.09693476],
                    [-1.42658685],
                    [-0.34135224],
                    [-0.17539867],
                ]
            )

        elif interpolation == InterpolationType.LINEAR:
            expected = np.array(
                [
                    [0.33333333],
                    [1.00292881e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [1.26291136e00],
                    [5.83576452e-03],
                    [3.77852829e-01],
                    [5.34820509e-01],
                    [4.17982425e-01],
                    [3.75933739e-01],
                    [1.50616580e00],
                    [-2.49651155e-03],
                    [8.21659249e-01],
                    [-5.39051987e-01],
                    [3.49542082e-01],
                    [-4.84064610e-02],
                    [1.75269299e00],
                    [0.00000000e00],
                    [1.67000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [-7.69022965e-02],
                    [1.15886757e01],
                    [1.33822245e00],
                    [1.04301742e00],
                    [9.89739329e00],
                    [1.85693476e00],
                    [-1.90992018e00],
                    [9.46864776e00],
                    [-9.35398671e-01],
                ]
            )

        elif interpolation == InterpolationType.SPLINE:
            expected = np.array(
                [
                    [0.33333333],
                    [0.61502453],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [0.76732112],
                    [0.43533947],
                    [0.19273203],
                    [1.15035619],
                    [0.76088147],
                    [0.81430269],
                    [0.90922359],
                    [0.13708974],
                    [0.32886699],
                    [-0.32665397],
                    [0.78933071],
                    [0.15428881],
                    [0.57293724],
                    [-0.1],
                    [1.67],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.70590907],
                    [2.24732687],
                    [-0.65897059],
                    [1.08074616],
                    [0.85131915],
                    [1.31781816],
                    [-1.21759843],
                    [0.30813958],
                    [-0.03112013],
                ]
            )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            expected = np.array(
                [
                    [0.33333333],
                    [1.00292881e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [1.51291136e00],
                    [5.83576452e-03],
                    [7.70352829e-01],
                    [5.34820509e-01],
                    [4.17982425e-01],
                    [3.75933739e-01],
                    [1.50616580e00],
                    [-2.49651155e-03],
                    [8.21659249e-01],
                    [-5.39051987e-01],
                    [3.49542082e-01],
                    [-4.84064610e-02],
                    [1.50269299e00],
                    [0.00000000e00],
                    [1.67000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [-7.69022965e-02],
                    [1.15886757e01],
                    [1.33822245e00],
                    [5.59684085e-01],
                    [9.89739329e00],
                    [1.09693476e00],
                    [-1.42658685e00],
                    [9.46864776e00],
                    [-1.75398671e-01],
                ]
            )

        elif interpolation in (InterpolationType.EACH_FRAME, InterpolationType.ALL_POINTS):
            expected = np.array(
                [
                    [0.33333333],
                    [0.00292881],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.01291136],
                    [0.00583576],
                    [-0.01464717],
                    [0.53482051],
                    [0.41798243],
                    [0.37593374],
                    [0.0061658],
                    [-0.00249651],
                    [0.03665925],
                    [-0.53905199],
                    [0.34954208],
                    [-0.04840646],
                    [0.00269299],
                    [0.0],
                    [1.67],
                    [0.0],
                    [0.0],
                    [0.0],
                    [-1.5269023],
                    [1.77867567],
                    [-0.94177755],
                    [0.55968409],
                    [0.08739329],
                    [1.09693476],
                    [-1.42658685],
                    [-0.34135224],
                    [-0.17539867],
                ]
            )

        elif interpolation == InterpolationType.CUSTOM:
            expected = np.array(
                [
                    [0.33333333],
                    [0.00292881],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.01291136],
                    [0.00583576],
                    [-0.01464717],
                    [0.53482051],
                    [0.41798243],
                    [0.37593374],
                    [0.0061658],
                    [-0.00249651],
                    [0.03665925],
                    [-0.53905199],
                    [0.34954208],
                    [-0.04840646],
                    [0.00269299],
                    [0.0],
                    [1.67],
                    [0.0],
                    [0.0],
                    [0.0],
                    [-1.5269023],
                    [1.77867567],
                    [-0.94177755],
                    [0.55968409],
                    [0.08739329],
                    [1.09693476],
                    [-1.42658685],
                    [-0.34135224],
                    [-0.17539867],
                ]
            )
        else:
            raise NotImplementedError("Interpolation type not implemented in the tests")

        npt.assert_almost_equal(ocp.init_vector, expected)

        with pytest.raises(RuntimeError, match="x_bounds should be built from a BoundsList"):
            ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("interpolation", [*InterpolationType])
def test_update_noised_initial_guess_rk4(interpolation, phase_dynamics):
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0

    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot
    u_init = InitialGuessList()
    u_init["tau"] = [0] * bio_model.nb_tau
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=phase_dynamics)
    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting=ns,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
    )

    # Path constraint and control path constraints
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["q"][2, -1] = 1.57
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * ntau, [tau_max] * ntau

    # Initial guesses
    x = InitialGuessList()
    u = InitialGuessList()
    if interpolation == InterpolationType.CONSTANT:
        x.add("q", [0] * nq, interpolation=interpolation)
        x.add("qdot", [0] * nq, interpolation=interpolation)
        u.add("tau", [tau_init] * ntau, interpolation=interpolation)
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x.add("q", np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.785], [2.0, 0.0, 1.57]]).T, interpolation=interpolation)
        x.add("qdot", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).T, interpolation=interpolation)
        u.add("tau", np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation)
    elif interpolation == InterpolationType.LINEAR:
        x.add("q", np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 1.57]]).T, interpolation=interpolation)
        x.add("qdot", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).T, interpolation=interpolation)
        u.add("tau", np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation)
    elif interpolation == InterpolationType.EACH_FRAME:
        x_init = np.zeros((nq * 2, ns + 1))
        for i in range(ns + 1):
            x_init[i, :] = np.linspace(i, i + 1, ns + 1)
        x.add("q", x_init[:nq, :], interpolation=interpolation)
        x.add("qdot", x_init[nq:, :], interpolation=interpolation)
        u.add("tau", np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.ALL_POINTS:
        x_init = np.zeros((nq * 2, ns + 1))
        for i in range(ns + 1):
            x_init[i, :] = np.linspace(i, i + 1, ns + 1)
        x.add("q", x_init[:nq, :], interpolation=interpolation)
        x.add("qdot", x_init[nq:, :], interpolation=interpolation)
        u.add("tau", np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        np.random.seed(42)
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x.add("q", np.random.random((nq, 5)), interpolation=interpolation, t=t)
        x.add("qdot", np.random.random((nqdot, 5)), interpolation=interpolation, t=t)
        u.add("tau", np.random.random((ntau, 5)), interpolation=interpolation, t=t)
    elif interpolation == InterpolationType.CUSTOM:
        x.add("q", lambda y: np.array([0] * nq), interpolation=interpolation)
        x.add("qdot", lambda y: np.array([0] * nqdot), interpolation=interpolation)
        u.add("tau", lambda y: np.array([tau_init] * ntau), interpolation=interpolation)

    else:
        raise NotImplementedError("This interpolation is not implemented yet")

    x.add_noise(
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        seed=42,
    )
    u.add_noise(
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
    )
    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x, u)
    else:
        ocp.update_initial_guess(x, u)

        if interpolation == InterpolationType.CONSTANT:
            expected = np.array(
                [
                    0.33333333,
                    -0.00752759,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.02704286,
                    -0.01376022,
                    0.02614717,
                    -0.36148533,
                    0.03110985,
                    -0.45302538,
                    0.01391964,
                    -0.01767666,
                    -0.06024513,
                    -0.39983054,
                    -0.08552041,
                    -0.26119874,
                    0.00591951,
                    0.0,
                    1.67,
                    0.0,
                    0.0,
                    0.0,
                    -0.50183952,
                    0.39463394,
                    -1.76766555,
                    1.80285723,
                    -1.37592544,
                    1.46470458,
                    0.92797577,
                    -1.37602192,
                    0.40446005,
                ],
            )
        elif interpolation == InterpolationType.LINEAR:
            expected = np.array(
                [
                    0.33333333,
                    0.99247241,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.27704286,
                    -0.01376022,
                    0.41864717,
                    -0.36148533,
                    0.03110985,
                    -0.45302538,
                    1.51391964,
                    -0.01767666,
                    0.72475487,
                    -0.39983054,
                    -0.08552041,
                    -0.26119874,
                    1.75591951,
                    0.0,
                    1.67,
                    0.0,
                    0.0,
                    0.0,
                    0.94816048,
                    10.20463394,
                    0.51233445,
                    2.28619056,
                    8.43407456,
                    2.22470458,
                    0.44464243,
                    8.43397808,
                    -0.35553995,
                ]
            )
        elif interpolation == InterpolationType.SPLINE:
            expected = np.array(
                [
                    0.33333333,
                    0.59113089,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    0.33024578,
                    0.65874739,
                    0.21811854,
                    -0.02346609,
                    0.45735055,
                    -0.22503382,
                    0.16992981,
                    0.44909993,
                    0.12213423,
                    0.00393179,
                    0.48605987,
                    -0.01781424,
                    0.15385356,
                    -0.1,
                    1.67,
                    -0.1,
                    -0.1,
                    -0.1,
                    0.44704601,
                    1.07886696,
                    -0.85834515,
                    2.76664681,
                    -0.90891928,
                    1.79505682,
                    1.76510889,
                    -1.195846,
                    0.9931955,
                ]
            )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            expected = np.array(
                [
                    0.33333333,
                    0.99247241,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.52704286,
                    -0.01376022,
                    0.81114717,
                    -0.36148533,
                    0.03110985,
                    -0.45302538,
                    1.51391964,
                    -0.01767666,
                    0.72475487,
                    -0.39983054,
                    -0.08552041,
                    -0.26119874,
                    1.50591951,
                    0.0,
                    1.67,
                    0.0,
                    0.0,
                    0.0,
                    0.94816048,
                    10.20463394,
                    0.51233445,
                    1.80285723,
                    8.43407456,
                    1.46470458,
                    0.92797577,
                    8.43397808,
                    0.40446005,
                ]
            )

        elif interpolation == InterpolationType.EACH_FRAME:
            expected = np.array(
                [
                    0.33333333,
                    -0.00752759,
                    -0.1,
                    -0.1,
                    -0.1,
                    0.0,
                    0.0,
                    0.36037619,
                    0.9,
                    2.3594805,
                    2.971848,
                    0.03110985,
                    -0.45302538,
                    0.6805863,
                    0.9,
                    2.60642154,
                    3.26683613,
                    -0.08552041,
                    -0.26119874,
                    1.00591951,
                    -0.1,
                    1.47,
                    -0.1,
                    0.0,
                    0.0,
                    -0.50183952,
                    0.39463394,
                    -1.76766555,
                    1.80285723,
                    -1.37592544,
                    1.46470458,
                    0.92797577,
                    -1.37602192,
                    0.40446005,
                ]
            )

        elif interpolation == InterpolationType.CUSTOM:
            expected = np.array(
                [
                    0.33333333,
                    -0.00752759,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.02704286,
                    -0.01376022,
                    0.02614717,
                    -0.36148533,
                    0.03110985,
                    -0.45302538,
                    0.01391964,
                    -0.01767666,
                    -0.06024513,
                    -0.39983054,
                    -0.08552041,
                    -0.26119874,
                    0.00591951,
                    0.0,
                    1.67,
                    0.0,
                    0.0,
                    0.0,
                    -0.50183952,
                    0.39463394,
                    -1.76766555,
                    1.80285723,
                    -1.37592544,
                    1.46470458,
                    0.92797577,
                    -1.37602192,
                    0.40446005,
                ]
            )
        else:
            raise NotImplementedError("Interpolation type not implemented yet")

        npt.assert_almost_equal(ocp.init_vector, expected[:, np.newaxis])

        with pytest.raises(RuntimeError, match="x_bounds should be built from a BoundsList"):
            ocp.update_bounds(x, u)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("interpolation", [*InterpolationType])
def test_update_noised_initial_guess_collocation(interpolation, phase_dynamics):
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=phase_dynamics)

    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot
    u_init = InitialGuessList()
    u_init["tau"] = [0] * bio_model.nb_tau
    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting=ns,
        phase_time=phase_time,
        ode_solver=solver,
        x_init=x_init,
        u_init=u_init,
    )

    # Path constraint and control path constraints
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:-1, [0, -1]] = 0
    x_bounds["qdot"][2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * ntau, [tau_max] * ntau

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    x = InitialGuessList()
    u = InitialGuessList()
    if interpolation == InterpolationType.CONSTANT:
        x.add("q", [0] * nq, interpolation=interpolation)
        x.add("qdot", [0] * nq, interpolation=interpolation)
        u.add("tau", [tau_init] * ntau, interpolation=interpolation)
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x.add("q", np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.785], [2.0, 0.0, 1.57]]).T, interpolation=interpolation)
        x.add("qdot", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).T, interpolation=interpolation)
        u.add("tau", np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation)
    elif interpolation == InterpolationType.LINEAR:
        x.add("q", np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 1.57]]).T, interpolation=interpolation)
        x.add("qdot", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).T, interpolation=interpolation)
        u.add("tau", np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation)
    elif interpolation == InterpolationType.EACH_FRAME:
        x_init = np.zeros((nq * 2, ns + 1))
        for i in range(ns + 1):
            x_init[i, :] = np.linspace(i, i + 1, ns + 1)
        x.add("q", x_init[:nq, :], interpolation=interpolation)
        x.add("qdot", x_init[nq:, :], interpolation=interpolation)
        u.add("tau", np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.ALL_POINTS:
        x_init = np.zeros((nq * 2, ns * (solver.polynomial_degree + 1) + 1))
        for i in range(nq * 2):
            x_init[i, :] = np.linspace(0, 1, ns * (solver.polynomial_degree + 1) + 1)
        x.add("q", x_init[:nq, :], interpolation=interpolation)
        x.add("qdot", x_init[nq:, :], interpolation=interpolation)
        u.add("tau", np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        np.random.seed(42)
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x.add("q", np.random.random((nq, 5)), interpolation=interpolation, t=t)
        x.add("qdot", np.random.random((nqdot, 5)), interpolation=interpolation, t=t)
        u.add("tau", np.random.random((ntau, 5)), interpolation=interpolation, t=t)
    elif interpolation == InterpolationType.CUSTOM:
        x.add("q", lambda y: np.array([0] * nq), interpolation=interpolation)
        x.add("qdot", lambda y: np.array([0] * nqdot), interpolation=interpolation)
        u.add("tau", lambda y: np.array([tau_init] * ntau), interpolation=interpolation)
    else:
        raise NotImplementedError("This interpolation is not implemented yet")

    x.add_noise(
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        seed=42,
    )
    u.add_noise(
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
    )
    ocp.update_initial_guess(x, u)

    with pytest.raises(RuntimeError, match="x_bounds should be built from a BoundsList"):
        ocp.update_bounds(x, u)
