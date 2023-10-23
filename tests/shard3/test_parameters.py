import pytest
from bioptim import ParameterList
import numpy as np


def my_parameter_function():
    return 1


def test_wrong_parameter():
    param = ParameterList()

    with pytest.raises(
        ValueError,
        match="Parameters are declared for all phases at once. You must therefore "
        "not use 'phase' but 'list_index' instead",
    ):
        param.add("gravity_z", my_parameter_function, size=1, phase=0)


def test_param_scaling():
    param = ParameterList()

    with pytest.raises(
        ValueError,
        match="Parameter scaling must be a numpy array",
    ):
        param.add("gravity_z", my_parameter_function, size=1, scaling="a")

    with pytest.raises(
        ValueError,
        match="Parameter scaling must be a numpy array",
    ):
        param.add("gravity_z", my_parameter_function, size=1, scaling=1.0)

    with pytest.raises(
        ValueError,
        match="Parameter scaling must be a numpy array",
    ):
        param.add("gravity_z", my_parameter_function, size=1, scaling=[])

    with pytest.raises(
        ValueError,
        match="Parameter scaling must contain only positive values",
    ):
        param.add("gravity_z", my_parameter_function, size=1, scaling=np.array([-1]))

    with pytest.raises(
        ValueError,
        match="Parameter scaling must be a 1- or 2- dimensional numpy array",
    ):
        param.add("gravity_z", my_parameter_function, size=1, scaling=np.array([[[1]]]))

    with pytest.raises(
        ValueError, match=f"The shape \(2\) of the scaling of parameter gravity_z does not match the params shape."
    ):
        param.add("gravity_z", my_parameter_function, size=3, scaling=np.array([1, 2]))

    with pytest.raises(
        ValueError, match=f"Invalid ncols for Parameter Scaling \(ncols = 2\), the expected number of column is 1"
    ):
        param.add("gravity_z", my_parameter_function, size=3, scaling=np.ones((1, 2)))
