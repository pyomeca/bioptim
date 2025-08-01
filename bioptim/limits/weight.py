import numpy as np

from .path_conditions import PathCondition
from ..misc.parameters_types import NpArray, Callable, FloatListOptional
from ..misc.enums import InterpolationType


class Weight(PathCondition):
    def __new__(
        cls,
        input_array: NpArray | Callable,
        t: FloatListOptional = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **extra_params,
    ):
        """
        Initialize the weight with a list of values or a single value.

        Parameters
        ----------
        input_array: np.ndarray | Callable
            An array of values to use in the interpolation of the weight.
        t: list[float]
            The time stamps
        interpolation
            The type of interpolation to use
        extra_params: dict
            Any parameters to pass to the path condition
        """
        obj = super().__new__(cls, input_array, t, interpolation, **extra_params)
        return obj
