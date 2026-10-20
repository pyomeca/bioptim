"""
Tests of the Euclidean configuration-space defaults provided by StateDynamics (issue #1037, step 3).
"""

import numpy as np
import numpy.testing as npt
import pytest
from casadi import MX, SX

from bioptim import StateDynamics, TorqueBiorbdModel
from bioptim.models.protocols.biomodel import BioModel
from bioptim.examples.toy_examples.custom_model.custom_package.my_model import MyModel
from bioptim.examples.toy_examples.stochastic_optimal_control.models.mass_point_model import MassPointModel
from ..utils import TestUtils

NEW_NAMES = ("nb_q", "nb_qdot", "name_q", "name_qdot", "normalize_q")


def _check_identity(model):
    f = model.normalize_q()
    n = model.nb_q
    assert f.size1_in(0) == n and f.size1_out(0) == n
    q = MX.sym("q", n)
    assert f(q).shape == (n, 1)
    sx = SX.sym("q", n)
    assert f(sx).shape == (n, 1)
    x = np.arange(1, n + 1, dtype=float)
    npt.assert_array_equal(np.array(f(x)).squeeze(), x)


def test_custom_model_defaults():
    model = MyModel()
    assert model.name_q == ("rotx",)
    assert model.name_qdot == ("rotx",)
    assert model.nb_q == model.nb_qdot == 1
    _check_identity(model)
    assert model.normalize_q() is model.normalize_q()
    npt.assert_array_equal(np.array(model.normalize_q()(np.array([0.3]))), [[0.3]])


def test_euclidean_biorbd_defaults():
    model = TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/cube.bioMod")
    names = tuple(model.name_dofs)
    assert model.name_q == names
    assert model.name_qdot == names
    assert model.nb_q == model.nb_qdot == len(names)
    _check_identity(model)


def test_quaternion_biorbd_name_qdot_raises():
    model = TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/trunk_and_2arm_quaternion.bioMod")
    assert (model.nb_q, model.nb_qdot) == (14, 12)
    assert len(model.name_q) == 14
    with pytest.raises(NotImplementedError, match="non-Euclidean"):
        model.name_qdot


def test_non_euclidean_adhoc_subclass_raises():
    class Tiny(StateDynamics):
        nb_q = 2
        nb_qdot = 1
        name_dofs = ("a", "b")

    Tiny.__abstractmethods__ = frozenset()  # only the configuration space is under test
    with pytest.raises(NotImplementedError, match="name_q and name_qdot"):
        Tiny().name_qdot


def _all_subclasses(cls):
    seen = set()
    stack = [cls]
    while stack:
        c = stack.pop()
        for sub in c.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


def test_no_mro_clash():
    assert MassPointModel is not None and MyModel is not None
    subclasses = _all_subclasses(StateDynamics)
    assert len(subclasses) > 0
    for cls in subclasses:
        mro = cls.__mro__
        for name in NEW_NAMES:
            assert any(name in vars(c) for c in mro), f"{cls} does not resolve {name}"
            getattr(cls, name)  # resolves at class level without error
            if BioModel in mro:
                owner = next(c for c in mro if name in vars(c))
                assert owner is not BioModel, f"{cls}: BioModel stub shadows {name}"
