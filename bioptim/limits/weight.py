import numpy as np
from numpy import array, ndarray
from scipy.interpolate import interp1d

from ..misc.parameters_types import (
    NpArray,
    Callable,
    FloatListOptional,
    AnyTuple,
    Int,
    Str,
    CX,
)
from ..misc.enums import InterpolationType


class Weight(ndarray):
    """
    A vector for the weight at a specific penalty index

    Attributes
    ----------
    n_nodes: int
        Number of indices in the penalty
    type: InterpolationType
        Type of interpolation
    t: list[float]
        Time vector
    extra_params: dict
        Any extra parameters that is associated to the path condition
    custom_function: Callable
        Custom function to describe the path condition interpolation

    Methods
    -------
    __array_finalize__(self, obj: "Weight")
        Finalize the array. This is required since Weight inherits from np.ndarray
    __reduce__(self) -> tuple
        Adding some attributes to the reduced state
    __setstate__(self, state: tuple, *args, **kwargs)
        Adding some attributes to the expanded state
    check_and_adjust_dimensions(self, n_nodes: int, element_name: str)
        Sanity check if the dimension of the vector is sounds when compare to the number
        of required nodes. If the function exit, then everything is okay
    evaluate_at(self, node: int)
        Evaluate the interpolation at a specific index
    """

    def __new__(
        cls,
        value: NpArray | Callable = 1,
        t: FloatListOptional = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **extra_params,
    ):
        """
        Initialize the weight with a list of values or a single value.

        Parameters
        ----------
        value: np.ndarray | Callable
            An array of values to use in the interpolation of the weight.
        t: list[float]
            The time stamps
        interpolation
            The type of interpolation to use
        extra_params: dict
            Any parameters to pass to the path condition
        """
        # Check and reinterpret input
        custom_function = None
        if interpolation == InterpolationType.CUSTOM:
            if not callable(value):
                raise TypeError("The input when using InterpolationType.CUSTOM should be a callable function")
            custom_function = value
            value = np.array(())
        if not isinstance(value, CX):
            value = np.asarray(value, dtype=float)

        if len(value.shape) == 0:
            value = value[np.newaxis]

        if interpolation == InterpolationType.CONSTANT:
            if value.shape[0] != 1:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.CONSTANT "
                    f"(ncols = {value.shape[0]}), the expected number of column is 1"
                )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if value.shape[0] != 1 and value.shape[0] != 3:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT "
                    f"(ncols = {value.shape[0]}), the expected number of column is 1 or 3"
                )
            if value.shape[0] == 1:
                value = np.repeat(value, 3, axis=1)

        elif interpolation == InterpolationType.LINEAR:
            if value.shape[0] != 2:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.LINEAR "
                    f"(ncols = {value.shape[0]}), the expected number of column is 2"
                )

        elif interpolation == InterpolationType.EACH_FRAME:
            # This will be verified in check_and_adjust_dimensions
            pass

        elif interpolation == InterpolationType.ALL_POINTS:
            raise ValueError(
                "The interpolation type ALL_POINTS is not allowed for Weight since the objective is "
                "evaluated only at the node and not at the collocation points. Use EACH_FRAME instead."
            )

        elif interpolation == InterpolationType.SPLINE:
            if value.shape[0] < 2:
                raise RuntimeError("Value for InterpolationType.SPLINE must have at least 2 columns")
            if t is None:
                raise RuntimeError("Spline necessitate a time vector")
            t = np.asarray(t)
            if value.shape[0] != t.shape[0]:
                raise RuntimeError("Spline necessitate a time vector which as the same length as column of data")

        elif interpolation == InterpolationType.CUSTOM:
            # We have to assume dimensions are those the user wants
            pass
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")
        if not isinstance(value, CX):
            obj = np.asarray(value).view(cls)
        else:
            obj = value

        # Additional information (do not forget to update __reduce__ and __setstate__)
        obj.n_nodes = None  # This will be set in check_and_adjust_dimensions
        obj.type = interpolation
        obj.t = t
        obj.extra_params = extra_params
        if interpolation == InterpolationType.CUSTOM:
            obj.custom_function = custom_function

        return obj

    def __array_finalize__(self, obj: "Weight"):
        """
        Finalize the array. This is required since Weight inherits from np.ndarray

        Parameters
        ----------
        obj: Weight
            The current object to finalize
        """

        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.n_nodes = getattr(obj, "n_nodes", None)
        self.type = getattr(obj, "type", None)
        self.t = getattr(obj, "t", None)
        self.extra_params = getattr(obj, "extra_params", None)

    def __array__(self) -> NpArray:
        return array([self])

    def __reduce__(self) -> AnyTuple:
        """
        Adding some attributes to the reduced state

        Returns
        -------
        The reduced state of the class
        """

        pickled_state = super(Weight, self).__reduce__()
        new_state = pickled_state[2] + (self.n_nodes, self.type, self.t, self.extra_params)
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state: AnyTuple, *args, **kwargs):
        """
        Adding some attributes to the expanded state

        Parameters
        ----------
        state: tuple
            The state as described by __reduce__
        """

        self.n_nodes = state[-4]
        self.type = state[-3]
        self.t = state[-2]
        self.extra_params = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(Weight, self).__setstate__(state[0:-4], *args, **kwargs)

    def check_and_adjust_dimensions(self, n_nodes: Int, element_name: Str):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        Parameters
        ----------
        n_nodes: int
            The number of nodes considered in the objective
        element_name: str
            The human readable name of the data structure
        """
        self.n_nodes = n_nodes
        if self.type == InterpolationType.EACH_FRAME:
            if self.shape[0] != self.n_nodes:
                raise RuntimeError(
                    f"Invalid number of column for {self.type} (ncols = {self.shape[0]}), the expected number of column is {self.n_nodes} for {element_name}."
                )

    def evaluate_at(self, node: Int, n_elements: int):
        """
        Evaluate the interpolation at a specific node

        Parameters
        ----------
        node: int
            The index at which to evaluate the objective weight.
            example: for node=Node.ALL -> node ranges from 0 to ns + 1
                     for node=Node.START -> node is 0
                     for node=Node.END -> node is 0
                     for node=[0, 4, 7, 9] -> node is 0, 1, 2, 3, respectively
        n_elements: int
            The number of components in the objective (e.g., the number of DoFs)

        Returns
        -------
        The values of the components at a specific index
        """

        # Sanity checks
        if self.n_nodes is None:
            raise RuntimeError(f"check_and_adjust_dimensions must be called at least once before evaluating at")
        if node > self.n_nodes:
            raise RuntimeError("index too high for evaluate at")

        if self.type == InterpolationType.CONSTANT:
            value = self[0]

        elif self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if node == 0:
                value = self[0]
            elif node == self.n_nodes - 1:
                value = self[2]
            else:
                value = self[1]

        elif self.type == InterpolationType.LINEAR:
            if self.n_nodes == 1:
                value = self[0]
            else:
                value = self[0] + (self[1] - self[0]) * node / ((self.n_nodes - 1))

        elif self.type == InterpolationType.EACH_FRAME:
            value = self[node]

        elif self.type == InterpolationType.SPLINE:
            spline = interp1d(self.t, self)
            value = spline(node / self.n_nodes * (self.t[-1] - self.t[0]))

        elif self.type == InterpolationType.CUSTOM:
            parameters = {}
            for key in self.extra_params:
                if key == "phase" or key == "option_type":
                    continue
                parameters[key] = self.extra_params[key]

            value = self.custom_function(node, **parameters)
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")

        repeated_value = np.repeat(value, n_elements)

        return repeated_value


class ObjectiveWeight(Weight):
    """
    A class to represent an objective weight.
    """

    def __new__(
        cls,
        value: NpArray | Callable = None,
        t: FloatListOptional = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **extra_params,
    ) -> "ObjectiveWeight":
        value = value if value is not None else 1
        obj = super().__new__(cls, value, t, interpolation, **extra_params)
        return obj


class ConstraintWeight(Weight):
    """
    A class to represent a constraint weight.
    """

    def __new__(
        cls,
        value: NpArray | Callable = None,
        t: FloatListOptional = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **extra_params,
    ) -> "ConstraintWeight":
        value = value if value is not None else 1
        obj = super().__new__(cls, value, t, interpolation, **extra_params)
        return obj
