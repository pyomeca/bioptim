"""
Characterization tests for the "configuration manifold" refactor (issue #1037).

The first part pins what must keep working while the OCP layer stops parsing DoF names / asking about quaternions:
sizes and names of q / qdot / tau for Euclidean and quaternion models, quaternion normalization during integration,
custom models and Pinocchio models.
Names are not stored on the OptimizationVariable: they are read from the plot legend ("<variable>_<name>-<col>") of
nlp.plot["<variable>_states" | "<variable>_controls"], the sizes from len(nlp.states[key]) / len(nlp.controls[key]).
"""

import numpy as np
import numpy.testing as npt
import pytest

from bioptim import (
    OptimalControlProgram,
    PhaseDynamics,
    TorqueBiorbdModel,
)
from ..utils import TestUtils

QUAT_MODEL = "/examples/models/trunk_and_2arm_quaternion.bioMod"
ROOT_NAMES = ["Pelvis_TransX", "Pelvis_TransY", "Pelvis_TransZ", "Pelvis_RotX", "Pelvis_RotY", "Pelvis_RotZ"]
QDOT_NAMES = ROOT_NAMES + [f"{arm}_omega{axis}" for arm in ("RightArm", "LeftArm") for axis in "XYZ"]


def _legend_names(nlp, plot_key: str, variable: str) -> list[str]:
    """Names of the elements of a variable, read from its plot legend ('<variable>_<name>-<col>')."""
    prefix = f"{variable}_"
    names = []
    for legend in nlp.plot[plot_key].legend:
        assert legend.startswith(prefix) and legend.endswith("-0")
        names.append(legend[len(prefix) : -len("-0")])
    return names


def _quaternion_ocp(model_class=TorqueBiorbdModel, **kwargs):
    model = model_class(TestUtils.bioptim_folder() + QUAT_MODEL, **kwargs)
    return OptimalControlProgram(model, 5, 1.0)


def test_quaternion_torque_model_sizes_and_names():
    ocp = _quaternion_ocp()
    nlp = ocp.nlp[0]
    model = nlp.model

    assert len(model.name_dofs) == 14
    assert len(nlp.states["q"]) == 14
    assert len(nlp.states["qdot"]) == 12
    assert len(nlp.controls["tau"]) == 12
    assert nlp.states["q"].shape == 14
    assert nlp.states["qdot"].shape == 12
    assert nlp.controls["tau"].shape == 12

    assert list(model.name_dofs[-2:]) == ["RightArm_QuatW", "LeftArm_QuatW"]
    assert _legend_names(nlp, "q_states", "q") == list(model.name_dofs)
    assert _legend_names(nlp, "qdot_states", "qdot") == QDOT_NAMES

    legend = nlp.plot["qdot_states"].legend
    assert len(legend) == 12
    assert "qdot_RightArm_omegaX-0" in legend
    assert "qdot_LeftArm_omegaZ-0" in legend


def test_euclidean_torque_model_sizes_and_names():
    model = TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/cube.bioMod")
    ocp = OptimalControlProgram(model, 5, 1.0)
    nlp = ocp.nlp[0]
    names = list(nlp.model.name_dofs)

    assert len(names) > 0
    assert len(nlp.states["q"]) == len(nlp.states["qdot"]) == len(nlp.controls["tau"]) == len(names)
    assert _legend_names(nlp, "q_states", "q") == names
    assert _legend_names(nlp, "qdot_states", "qdot") == names
    assert _legend_names(nlp, "tau_controls", "tau") == names


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
def test_free_floating_base_quaternion_sizes_and_names(phase_dynamics):
    from bioptim.examples.toy_examples.torque_driven_ocp import example_quaternions as ocp_module

    ocp = ocp_module.prepare_ocp(
        TestUtils.bioptim_folder() + QUAT_MODEL, 5, 0.25, phase_dynamics=phase_dynamics, expand_dynamics=True
    )
    nlp = ocp.nlp[0]

    assert {key: len(nlp.states[key]) for key in nlp.states.keys()} == {
        "q_roots": 6,
        "q_joints": 8,
        "qdot_roots": 6,
        "qdot_joints": 6,
    }
    assert {key: len(nlp.controls[key]) for key in nlp.controls.keys()} == {"tau_joints": 6}

    assert _legend_names(nlp, "q_roots_states", "q_roots") == ROOT_NAMES
    assert _legend_names(nlp, "q_joints_states", "q_joints") == [
        "RightArm_QuatX",
        "RightArm_QuatY",
        "RightArm_QuatZ",
        "LeftArm_QuatX",
        "LeftArm_QuatY",
        "LeftArm_QuatZ",
        "RightArm_QuatW",
        "LeftArm_QuatW",
    ]
    assert _legend_names(nlp, "qdot_roots_states", "qdot_roots") == ROOT_NAMES
    assert _legend_names(nlp, "qdot_joints_states", "qdot_joints") == [
        f"{arm}_omega{axis}" for arm in ("RightArm", "LeftArm") for axis in "XYZ"
    ]
    # The tau labels of quaternion DoFs are intentionally not pinned, only the size is.
    assert len(_legend_names(nlp, "tau_joints_controls", "tau_joints")) == 6


def test_quaternion_is_normalized_by_the_integrator():
    ocp = _quaternion_ocp()
    nlp = ocp.nlp[0]
    integrator = nlp.dynamics[0]
    assert integrator.function.name_in() == ["t_span", "x0", "u", "p", "a", "d"]
    assert integrator.function.name_out()[0] == "xf"

    x0 = np.zeros(26)
    x0[[6, 7, 8, 12]] = 2 * np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])  # not a unit quaternion
    x0[[9, 10, 11, 13]] = 2 * np.array([0.3, 0.1, 0.2, np.sqrt(1 - 0.14)])
    x0[14:] = 0.1
    assert abs(np.linalg.norm(x0[[6, 7, 8, 12]]) - 1) > 0.5

    xf = np.array(integrator.function(np.array([0.0, 1.0]), x0, np.zeros(12), [], [], [])[0]).reshape(-1)
    assert xf.shape == (26,)
    npt.assert_allclose(np.linalg.norm(xf[[6, 7, 8, 12]]), 1.0, atol=1e-8)
    npt.assert_allclose(np.linalg.norm(xf[[9, 10, 11, 13]]), 1.0, atol=1e-8)


def test_custom_model_still_builds():
    from bioptim.examples.toy_examples.custom_model import main as ocp_module
    from bioptim.examples.toy_examples.custom_model.custom_package import my_model as model

    ocp = ocp_module.prepare_ocp(
        model=model.MyModel(),
        final_time=1,
        n_shooting=30,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        n_threads=1,
        expand_dynamics=True,
    )
    assert ocp.nlp[0].model.name_dofs == ["rotx"]


# ----------------------------------------------------------------------------------------------------------------------
# Pinocchio (every test imports pinocchio on its own so that the biorbd tests run without it)
# ----------------------------------------------------------------------------------------------------------------------
def _col(x):
    return np.asarray(x, dtype=float).reshape(-1, 1)


def _inertia():
    return pytest.importorskip("pinocchio").Inertia(1.0, np.array([0.0, 0.0, 0.1]), np.diag([0.02, 0.02, 0.01]))


def _build_pin_model(*joints):
    """Chain of (name, JointModel) pairs, each attached to the previous one with a 0.3 m offset along z."""
    pin = pytest.importorskip("pinocchio")
    model = pin.Model()
    parent = 0
    for i, (name, joint) in enumerate(joints):
        placement = pin.SE3.Identity() if i == 0 else pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.3]))
        parent = model.addJoint(parent, joint, placement, name)
        model.appendBodyToJoint(parent, _inertia(), pin.SE3.Identity())
    return model


def _freeflyer_rx_model():
    pin = pytest.importorskip("pinocchio")
    return _build_pin_model(("root", pin.JointModelFreeFlyer()), ("arm", pin.JointModelRX()))


def _spherical_model():
    pin = pytest.importorskip("pinocchio")
    return _build_pin_model(("sph", pin.JointModelSpherical()))


def test_pinocchio_freeflyer_sizes_and_normalization():
    from bioptim import PinocchioModel

    model = PinocchioModel(_freeflyer_rx_model())
    assert model.nb_q == 8
    assert model.nb_qdot == 7

    q = np.array([1.0, 2.0, 3.0, 0.2, 0.4, 0.4, 0.8, 0.7])  # (x, y, z, qx, qy, qz, qw, angle)
    q[3:7] *= 3.0
    q_norm = np.array(model.normalize_state_quaternions()(_col(q))).reshape(-1)

    npt.assert_allclose(np.linalg.norm(q_norm[3:7]), 1.0, atol=1e-12)
    npt.assert_allclose(q_norm[3:7], q[3:7] / np.linalg.norm(q[3:7]), atol=1e-12)
    npt.assert_array_equal(q_norm[:3], q[:3])
    assert q_norm[7] == q[7]


def test_pinocchio_spherical_normalization():
    from bioptim import PinocchioModel

    model = PinocchioModel(_spherical_model())
    assert model.nb_q == 4
    assert model.nb_qdot == 3

    q = np.array([0.0, 0.0, 3.0, 4.0])
    q_norm = np.array(model.normalize_state_quaternions()(_col(q))).reshape(-1)
    npt.assert_allclose(q_norm, [0.0, 0.0, 0.6, 0.8], atol=1e-12)


def test_pinocchio_rx_plus_rubx_sizes():
    pin = pytest.importorskip("pinocchio")
    from bioptim import PinocchioModel

    model = PinocchioModel(_build_pin_model(("rx", pin.JointModelRX()), ("rubx", pin.JointModelRUBX())))
    assert model.nb_q == 3
    assert model.nb_qdot == 2


# ----------------------------------------------------------------------------------------------------------------------
# Strict xfail reproducers: each asserts the CORRECT behaviour and must pass once the corresponding bug is fixed.
# ----------------------------------------------------------------------------------------------------------------------
def test_b2_joint_acceleration_quaternion_ocp_builds():
    from bioptim import JointAccelerationBiorbdModel

    ocp = _quaternion_ocp(JointAccelerationBiorbdModel)
    assert len(ocp.nlp[0].states["q"]) == 14
    assert len(ocp.nlp[0].states["qdot"]) == 12


def test_b2b_joint_acceleration_dynamics_uses_reshape_qdot():
    from bioptim import JointAccelerationBiorbdModel

    ocp = _quaternion_ocp(JointAccelerationBiorbdModel)
    nlp = ocp.nlp[0]
    n_q, n_qdot = nlp.model.nb_q, nlp.model.nb_qdot

    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, n_q)
    for idx in ([6, 7, 8, 12], [9, 10, 11, 13]):
        q[idx] /= np.linalg.norm(q[idx])
    qdot = rng.uniform(0.5, 1.5, n_qdot)
    x = np.concatenate((q, qdot))
    u = np.zeros(nlp.controls.shape)

    xdot = np.array(nlp.dynamics_func(0.0, x, u, [], [], [])).reshape(-1)
    expected = np.array(nlp.model.reshape_qdot()(q, qdot, [])).reshape(-1)
    assert expected.shape == (n_q,)
    npt.assert_allclose(xdot[:n_q], expected, atol=1e-8)


@pytest.mark.xfail(
    strict=True, reason="B3: phase_mapping indices into the cross-phase union are used on per-phase names"
)
def test_b3_two_phases_with_different_models():
    folder = TestUtils.bioptim_folder() + "/examples/models/"
    models = [TorqueBiorbdModel(folder + "cube.bioMod"), TorqueBiorbdModel(folder + "double_pendulum.bioMod")]
    ocp = OptimalControlProgram(models, [5, 5], [1.0, 1.0])

    for nlp in ocp.nlp:
        assert _legend_names(nlp, "qdot_states", "qdot") == list(nlp.model.name_dofs)
        assert len(nlp.states["qdot"]) == len(nlp.model.name_dofs)


def _pin_ocp(pin_model, n_shooting=5, **kwargs):
    from bioptim.models.pinocchio import TorquePinocchioModel

    return OptimalControlProgram(TorquePinocchioModel(pin_model), n_shooting, 1.0, **kwargs)


@pytest.mark.xfail(strict=True, reason="B4: Pinocchio OCP declares q with nb_qdot elements instead of nb_q")
def test_b4_pinocchio_freeflyer_ocp_sizes():
    nlp = _pin_ocp(_freeflyer_rx_model()).nlp[0]
    assert len(nlp.states["q"]) == 8
    assert len(nlp.states["qdot"]) == 7


@pytest.mark.xfail(
    strict=True, reason="B4: Pinocchio free-flyer quaternion is not kept on the unit sphere along the OCP"
)
def test_b4_pinocchio_freeflyer_quaternion_stays_unit_after_solve():
    from bioptim import BoundsList, InitialGuessList, InterpolationType, Solver, SolutionMerge

    pin = pytest.importorskip("pinocchio")
    pin_model = _freeflyer_rx_model()
    q0 = pin.neutral(pin_model)
    n_q, n_qdot = pin_model.nq, pin_model.nv

    x_bounds = BoundsList()
    q_min, q_max = np.full((n_q, 3), -5.0), np.full((n_q, 3), 5.0)
    q_min[:, 0] = q_max[:, 0] = q0  # start at the neutral configuration
    x_bounds.add(
        "q", min_bound=q_min, max_bound=q_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )
    qdot_min, qdot_max = np.full((n_qdot, 3), -10.0), np.full((n_qdot, 3), 10.0)
    qdot_min[:, 0] = qdot_max[:, 0] = 0.0
    x_bounds.add(
        "qdot",
        min_bound=qdot_min,
        max_bound=qdot_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_init = InitialGuessList()
    x_init.add("q", q0, interpolation=InterpolationType.CONSTANT)
    x_init.add("qdot", np.ones(n_qdot), interpolation=InterpolationType.CONSTANT)

    ocp = _pin_ocp(pin_model, x_bounds=x_bounds, x_init=x_init)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(2)
    sol = ocp.solve(solver)

    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
    assert q.shape[0] == n_q
    for node in range(q.shape[1]):
        npt.assert_allclose(np.linalg.norm(q[3:7, node]), 1.0, atol=1e-6)


@pytest.mark.xfail(
    strict=True, reason="B4: Pinocchio reshape_qdot returns qdot (nv entries) instead of the nq derivative of q"
)
def test_b4_pinocchio_reshape_qdot_matches_finite_difference():
    from bioptim import PinocchioModel

    pin = pytest.importorskip("pinocchio")
    pin_model = _freeflyer_rx_model()
    model = PinocchioModel(pin_model)

    rng = np.random.default_rng(1)
    q = pin.integrate(pin_model, pin.neutral(pin_model), rng.uniform(-0.5, 0.5, pin_model.nv))
    v = rng.uniform(-1.0, 1.0, pin_model.nv)
    eps = 1e-7

    q_dot = np.array(model.reshape_qdot()(_col(q), _col(v), [])).reshape(-1)
    assert q_dot.shape == (pin_model.nq,)
    expected = (pin.integrate(pin_model, q, eps * v) - q) / eps
    npt.assert_allclose(q_dot, expected, atol=1e-5)


@pytest.mark.xfail(strict=True, reason="B5: Pinocchio nb_quaternions counts nq - nv, wrong for RUBX and Planar joints")
@pytest.mark.parametrize("joint_name", ["RUBX", "Planar"])
def test_b5_pinocchio_nb_quaternions_is_zero_without_quaternion(joint_name):
    from bioptim import PinocchioModel

    pin = pytest.importorskip("pinocchio")
    joint = pin.JointModelRUBX() if joint_name == "RUBX" else pin.JointModelPlanar()
    model = PinocchioModel(_build_pin_model(("joint", joint)))
    assert model.nb_q > model.nb_qdot
    assert model.nb_quaternions == 0


@pytest.mark.xfail(
    strict=True, reason="B5: Pinocchio normalizes the last 4 q of any joint with nq = nv + 1 (RUBX gives NaN)"
)
def test_b5_pinocchio_normalize_rx_plus_rubx():
    from bioptim import PinocchioModel

    pin = pytest.importorskip("pinocchio")
    model = PinocchioModel(_build_pin_model(("rx", pin.JointModelRX()), ("rubx", pin.JointModelRUBX())))
    q_norm = np.array(model.normalize_state_quaternions()(_col([3.0, 4.0, 0.0]))).reshape(-1)
    npt.assert_allclose(q_norm, [3.0, 1.0, 0.0], atol=1e-12)


def _quaternion_model_with_actuators(tmp_path) -> str:
    """The quaternion bioMod with a constant actuator (both directions) on every DoF."""
    text = open(TestUtils.bioptim_folder() + QUAT_MODEL).read()
    dofs = {
        "Pelvis": ["TransX", "TransY", "TransZ", "RotX", "RotY", "RotZ"],
        "RightArm": ["QuatX", "QuatY", "QuatZ"],
        "LeftArm": ["QuatX", "QuatY", "QuatZ"],
    }
    for segment, names in dofs.items():
        for name in names:
            for direction in ("positive", "negative"):
                text += (
                    f"\nactuator {segment}\n type Constant\n dof {name}\n direction {direction}\n Tmax 10\n"
                    "endactuator\n"
                )
    path = tmp_path / "quaternion_with_actuators.bioMod"
    path.write_text(text)
    return str(path)


@pytest.mark.xfail(
    strict=True, reason="B6: residual_tau of a quaternion model is sized/named with 6 elements instead of 12"
)
def test_b6_residual_tau_has_nb_qdot_elements(tmp_path):
    from bioptim import TorqueActivationBiorbdModel

    model = TorqueActivationBiorbdModel(_quaternion_model_with_actuators(tmp_path), with_residual_torque=True)
    nlp = OptimalControlProgram(model, 5, 1.0).nlp[0]
    assert len(nlp.controls["residual_tau"]) == 12


@pytest.mark.xfail(strict=True, reason="B7: plot mapping of qdot/tau maps nb_q entries instead of nb_qdot")
def test_b7_plot_phase_mappings_have_nb_qdot_entries():
    nlp = _quaternion_ocp().nlp[0]
    assert len(nlp.plot["qdot_states"].phase_mappings.to_first.map_idx) == 12
    assert len(nlp.plot["tau_controls"].phase_mappings.to_first.map_idx) == 12


@pytest.mark.xfail(strict=True, reason="B10: out of scope of #1037, follow-up issue")
def test_b10_quaternion_root_with_free_floating_base(tmp_path):
    from bioptim import TorqueFreeFloatingBaseBiorbdModel

    path = tmp_path / "quaternion_root.bioMod"
    path.write_text("""version 3
root_actuated 0
external_forces 0
gravity 0 0 -9.81

segment Root
    translations xyz
    rotations q
    mass 10
    inertia
        1 0 0
        0 1 0
        0 0 1
    com 0 0 0
    mesh 0 0 -0.5
    mesh 0 0 0.5
endsegment

segment Child
    parent Root
    RT 0 0 0 xyz 0 0 0.5
    rotations x
    mass 1
    inertia
        0.1 0 0
        0 0.1 0
        0 0 0.01
    com 0 0 -0.2
    mesh 0 0 0
    mesh 0 0 -0.4
endsegment
""")
    model = TorqueFreeFloatingBaseBiorbdModel(str(path))
    nlp = OptimalControlProgram(model, 5, 1.0).nlp[0]
    assert len(nlp.states["q_roots"]) == 7
