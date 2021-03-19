import pytest
from bioptim import ParameterList


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
