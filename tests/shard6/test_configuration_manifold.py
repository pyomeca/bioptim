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
