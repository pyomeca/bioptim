import pytest
from bioptim import ParameterList, VariableScaling
import numpy as np


def my_parameter_function():
    return 1


def test_wrong_parameter():
    param = ParameterList(use_sx=True)

    with pytest.raises(
        ValueError,
        match="Parameters are declared for all phases at once. You must therefore "
        "not use 'phase' but 'list_index' instead",
    ):
        param.add(
            "gravity_z",
            function=my_parameter_function,
            size=1,
            phase=0,
            scaling=VariableScaling("gravity_z", np.array([1])),
        )


def test_param_scaling():
    param = ParameterList(use_sx=True)

    with pytest.raises(
        ValueError,
        match="Scaling must be a VariableScaling, not <class 'str'>",
    ):
        param.add("gravity_z", function=my_parameter_function, size=1, scaling="a")

    with pytest.raises(
        ValueError,
        match="Scaling must be a VariableScaling, not <class 'float'>",
    ):
        param.add("gravity_z", function=my_parameter_function, size=1, scaling=1.0)

    with pytest.raises(
        ValueError,
        match="Scaling must be a VariableScaling, not <class 'list'>",
    ):
        param.add("gravity_z", function=my_parameter_function, size=1, scaling=[])

    with pytest.raises(
        ValueError,
        match="Scaling factors must be strictly greater than zero.",
    ):
        param.add(
            "gravity_z", function=my_parameter_function, size=1, scaling=VariableScaling("gravity_z", np.array([-1]))
        )

    with pytest.raises(
        ValueError,
        match="Scaling must be a 1- or 2- dimensional numpy array",
    ):
        param.add(
            "gravity_z", function=my_parameter_function, size=1, scaling=VariableScaling("gravity_z", np.array([[[1]]]))
        )

    with pytest.raises(ValueError, match=f"Parameter scaling must be of size 3, not 2."):
        param.add(
            "gravity_z", function=my_parameter_function, size=3, scaling=VariableScaling("gravity_z", np.array([1, 2]))
        )

    with pytest.raises(ValueError, match=f"Parameter scaling must have exactly one column"):
        param.add(
            "gravity_z", function=my_parameter_function, size=3, scaling=VariableScaling("gravity_z", np.ones((3, 2)))
        )
