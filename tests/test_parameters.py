import pytest
from bioptim import ParameterList
import numpy as np


def test_wrong_parameter():
    param = ParameterList()

    my_parameter_function = 1
    initial_gravity = 1
    bounds = 1

    with pytest.raises(
        ValueError,
        match="Parameters are declared for all phases at once. You must therefore "
        "not use 'phase' but 'list_index' instead",
    ):
        param.add("gravity_z", my_parameter_function, initial_gravity, bounds, size=1, phase=0)


def test_param_scaling():
    param = ParameterList()

    my_parameter_function = 1
    initial_gravity = 1
    bounds = 1

    with pytest.raises(
        ValueError,
        match="Parameter scaling must be a numpy array",
    ):
        param.add("gravity_z", my_parameter_function, initial_gravity, bounds, size=1, scaling="a")

    with pytest.raises(
        ValueError,
        match="Parameter scaling must be a numpy array",
    ):
        param.add("gravity_z", my_parameter_function, initial_gravity, bounds, size=1, scaling=1.0)

    with pytest.raises(
        ValueError,
        match="Parameter scaling must be a numpy array",
    ):
        param.add("gravity_z", my_parameter_function, initial_gravity, bounds, size=1, scaling=[])

    with pytest.raises(
        ValueError,
        match="Parameter scaling must contain only positive values",
    ):
        param.add("gravity_z", my_parameter_function, initial_gravity, bounds, size=1, scaling=np.array([-1]))
