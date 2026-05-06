import importlib.util

import pytest

from bioptim import PinocchioModel


def test_pinocchio_model_import_is_lazy():
    with pytest.raises(ValueError, match="The model should be of type 'str' or 'pinocchio.Model'"):
        PinocchioModel(1)


def test_pinocchio_missing_dependency_message():
    if importlib.util.find_spec("pinocchio") is not None:
        pytest.skip("Pinocchio is installed in this environment")

    with pytest.raises(ModuleNotFoundError, match="requires the optional dependency 'pinocchio'"):
        PinocchioModel("model.urdf")

