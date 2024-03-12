from typing import Callable, Any, List, Tuple

import numpy as np
from casadi import MX, SX, vertcat
from scipy.interpolate import interp1d
from numpy import array, ndarray

from ..misc.enums import InterpolationType, MagnitudeType
from ..misc.options import OptionGeneric, OptionDict
from ..optimization.variable_scaling import VariableScaling


class PathCondition(np.ndarray):
    """
    A matrix for any component (rows) and time (columns) conditions

    Attributes
    ----------
    n_shooting: int
        Number of shooting points
    type: InterpolationType
        Type of interpolation
    t: list[float]
        Time vector
    extra_params: dict
        Any extra parameters that is associated to the path condition
    slice_list: slice
        Slice of the array
    custom_function: Callable
        Custom function to describe the path condition interpolation

    Methods
    -------
    __array_finalize__(self, obj: "PathCondition")
        Finalize the array. This is required since PathCondition inherits from np.ndarray
    __reduce__(self) -> tuple
        Adding some attributes to the reduced state
    __setstate__(self, state: tuple, *args, **kwargs)
        Adding some attributes to the expanded state
    check_and_adjust_dimensions(self, n_elements: int, n_shooting: int, element_name: str)
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay
    evaluate_at(self, shooting_point: int)
        Evaluate the interpolation at a specific shooting point
    """

    def __new__(
        cls,
        input_array: np.ndarray | Callable,
        t: list = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        slice_list: slice | list | tuple = None,
        **extra_params,
    ):
        """
        Parameters
        ----------
        input_array: np.ndarray | Callable
            The matrix of interpolation, rows are the components, columns are the time
        t: list[float]
            The time stamps
        interpolation: InterpolationType
            The type of interpolation. It determines how many timestamps are required
        slice_list: slice | list | tuple
            If the data should be sliced. It is more relevant for custom functions
        extra_params: dict
            Any parameters to pass to the path condition
        """

        # Check and reinterpret input
        custom_function = None
        if interpolation == InterpolationType.CUSTOM:
            if not callable(input_array):
                raise TypeError("The input when using InterpolationType.CUSTOM should be a callable function")
            custom_function = input_array
            input_array = np.array(())
        if not isinstance(input_array, (MX, SX)):
            input_array = np.asarray(input_array, dtype=float)

        if len(input_array.shape) == 0:
            input_array = input_array[np.newaxis, np.newaxis]

        if interpolation == InterpolationType.CONSTANT:
            if len(input_array.shape) == 1:
                input_array = input_array[:, np.newaxis]
            if input_array.shape[1] != 1:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.CONSTANT "
                    f"(ncols = {input_array.shape[1]}), the expected number of column is 1"
                )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if len(input_array.shape) == 1:
                input_array = input_array[:, np.newaxis]
            if input_array.shape[1] != 1 and input_array.shape[1] != 3:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT "
                    f"(ncols = {input_array.shape[1]}), the expected number of column is 1 or 3"
                )
            if input_array.shape[1] == 1:
                input_array = np.repeat(input_array, 3, axis=1)
        elif interpolation == InterpolationType.LINEAR:
            if input_array.shape[1] != 2:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.LINEAR "
                    f"(ncols = {input_array.shape[1]}), the expected number of column is 2"
                )
        elif interpolation == InterpolationType.EACH_FRAME or interpolation == InterpolationType.ALL_POINTS:
            # This will be verified when the expected number of columns is set
            pass
        elif interpolation == InterpolationType.SPLINE:
            if input_array.shape[1] < 2:
                raise RuntimeError("Value for InterpolationType.SPLINE must have at least 2 columns")
            if t is None:
                raise RuntimeError("Spline necessitate a time vector")
            t = np.asarray(t)
            if input_array.shape[1] != t.shape[0]:
                raise RuntimeError("Spline necessitate a time vector which as the same length as column of data")

        elif interpolation == InterpolationType.CUSTOM:
            # We have to assume dimensions are those the user wants
            pass
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")
        if not isinstance(input_array, (MX, SX)):
            obj = np.asarray(input_array).view(cls)
        else:
            obj = input_array

        # Additional information (do not forget to update __reduce__ and __setstate__)
        obj.n_shooting = None
        obj.type = interpolation
        obj.t = t
        obj.extra_params = extra_params
        obj.slice_list = slice_list
        if interpolation == InterpolationType.CUSTOM:
            obj.custom_function = custom_function

        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array. This is required since PathCondition inherits from np.ndarray

        Parameters
        ----------
        obj: PathCondition
            The current object to finalize
        """

        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.n_shooting = getattr(obj, "n_shooting", None)
        self.type = getattr(obj, "type", None)
        self.t = getattr(obj, "t", None)
        self.extra_params = getattr(obj, "extra_params", None)
        self.slice_list = getattr(obj, "slice_list", None)

    def __array__(self) -> ndarray:
        return array([self])

    def __reduce__(self) -> tuple:
        """
        Adding some attributes to the reduced state

        Returns
        -------
        The reduced state of the class
        """

        pickled_state = super(PathCondition, self).__reduce__()
        new_state = pickled_state[2] + (self.n_shooting, self.type, self.t, self.extra_params, self.slice_list)
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state: tuple, *args, **kwargs):
        """
        Adding some attributes to the expanded state

        Parameters
        ----------
        state: tuple
            The state as described by __reduce__
        """

        self.n_shooting = state[-5]
        self.type = state[-4]
        self.t = state[-3]
        self.extra_params = state[-2]
        self.slice_list = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(PathCondition, self).__setstate__(state[0:-5], *args, **kwargs)

    def check_and_adjust_dimensions(self, n_elements: int, n_shooting: int, element_name: str):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        Parameters
        ----------
        n_elements: int
            The expected number of rows
        n_shooting: int
            The number of shooting points in the ocp
        element_name: str
            The human readable name of the data structure
        """

        if (
            self.type == InterpolationType.CONSTANT
            or self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            or self.type == InterpolationType.LINEAR
            or self.type == InterpolationType.SPLINE
            or self.type == InterpolationType.CUSTOM
        ):
            self.n_shooting = n_shooting
        elif self.type == InterpolationType.EACH_FRAME or self.type == InterpolationType.ALL_POINTS:
            self.n_shooting = n_shooting + 1
        else:
            if self.n_shooting != n_shooting:
                raise RuntimeError(
                    f"Invalid number of shooting ({self.n_shooting}), the expected number is {n_shooting}"
                )

        if self.type == InterpolationType.CUSTOM:
            parameters = {}
            for key in self.extra_params:
                if key == "phase" or key == "option_type":
                    continue
                parameters[key] = self.extra_params[key]

            slice_list = self.slice_list
            if slice_list is not None:
                val_size = self.custom_function(0, **parameters)[
                    slice_list.start : slice_list.stop : slice_list.step
                ].shape[0]
            else:
                val_size = self.custom_function(0, **parameters).shape[0]
        else:
            val_size = self.shape[0]
        if val_size != n_elements:
            raise RuntimeError(f"Invalid number of {element_name} ({val_size}), the expected size is {n_elements}")

        if self.type == InterpolationType.EACH_FRAME:
            if self.shape[1] != self.n_shooting:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.EACH_FRAME (ncols = {self.shape[1]}), "
                    f"the expected number of column is {self.n_shooting}"
                )
        elif self.type == InterpolationType.ALL_POINTS:
            if self.shape[1] != self.n_shooting:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.ALL_POINTS (ncols = {self.shape[1]}), "
                    f"the expected number of column is {self.n_shooting}"
                )

    def evaluate_at(self, shooting_point: int, repeat: int = 1):
        """
        Evaluate the interpolation at a specific shooting point

        Parameters
        ----------
        shooting_point: int
            The shooting point to evaluate the path condition at
        repeat: int
            The number of collocation points (only used for InterpolationType.LINEAR in collocations)

        Returns
        -------
        The values of the components at a specific time index
        """

        if self.n_shooting is None:
            raise RuntimeError(f"check_and_adjust_dimensions must be called at least once before evaluating at")

        if self.type == InterpolationType.CONSTANT:
            return self[:, 0]
        elif self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if shooting_point == 0:
                return self[:, 0]
            elif shooting_point == self.n_shooting:
                return self[:, 2]
            elif shooting_point > self.n_shooting:
                raise RuntimeError("shooting point too high")
            else:
                return self[:, 1]
        elif self.type == InterpolationType.LINEAR:
            return self[:, 0] + (self[:, 1] - self[:, 0]) * shooting_point / (
                self.n_shooting * repeat
            )  # see if repeat or repeat + 1
        elif self.type == InterpolationType.EACH_FRAME:
            return self[:, shooting_point]
        elif self.type == InterpolationType.ALL_POINTS:
            return self[:, shooting_point]
        elif self.type == InterpolationType.SPLINE:
            spline = interp1d(self.t, self)
            return spline(shooting_point / self.n_shooting * (self.t[-1] - self.t[0]))
        elif self.type == InterpolationType.CUSTOM:
            if self.slice_list is not None:
                slice_list = self.slice_list
                return self.custom_function(shooting_point, **self.extra_params)[
                    slice_list.start : slice_list.stop : slice_list.step
                ]
            else:
                parameters = {}
                for key in self.extra_params:
                    if key == "phase" or key == "option_type":
                        continue
                    parameters[key] = self.extra_params[key]

                return self.custom_function(shooting_point, **parameters)
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")


class Bounds(OptionGeneric):
    """
    A placeholder for bounds constraints

    Attributes
    ----------
    n_shooting: int
        The number of shooting of the ocp
    min: PathCondition
        The minimal bound
    max: PathCondition
        The maximal bound
    type: InterpolationType
        The type of interpolation of the bound
    t: list[float]
        The time stamps
    extra_params: dict
        Any parameters to pass to the path condition

    Methods
    -------
    check_and_adjust_dimensions(self, n_elements: int, n_shooting: int)
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay
    concatenate(self, other: "Bounds")
        Vertical concatenate of two Bounds
    scale(self, scaling: float | np.ndarray)
        Scaling a Bound
    __getitem__(self, slice_list: slice) -> "Bounds"
        Allows to get from square brackets
    __setitem__(self, slice: slice, value: np.ndarray | list | float)
        Allows to set from square brackets
    __bool__(self) -> bool
        Get if the Bounds is empty
    shape(self) -> int
        Get the size of the Bounds
    """

    def __init__(
        self,
        key,
        min_bound: Callable | PathCondition | np.ndarray | list | tuple | float = None,
        max_bound: Callable | PathCondition | np.ndarray | list | tuple | float = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        slice_list: slice | list | tuple = None,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        min_bound: Callable | PathCondition | np.ndarray | list | tuple
            The minimal bound
        max_bound: Callable | PathCondition | np.ndarray | list | tuple
            The maximal bound
        interpolation: InterpolationType
            The type of interpolation of the bound
        slice_list: slice | list | tuple
            Slice of the array
        parameters: dict
            Any extra parameters that is associated to the path condition
        """
        min_bound = min_bound if min_bound is not None else ()
        max_bound = max_bound if max_bound is not None else ()

        if isinstance(min_bound, PathCondition):
            self.min = min_bound
        else:
            self.min = PathCondition(min_bound, interpolation=interpolation, slice_list=slice_list, **parameters)

        if isinstance(max_bound, PathCondition):
            self.max = max_bound
        else:
            self.max = PathCondition(max_bound, interpolation=interpolation, slice_list=slice_list, **parameters)

        super(Bounds, self).__init__(**parameters)
        self.type = interpolation
        self.t = None
        self.extra_params = self.min.extra_params
        self.n_shooting = self.min.n_shooting
        self.key = key

    def check_and_adjust_dimensions(self, n_elements: int, n_shooting: int):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        Parameters
        ----------
        n_elements: int
            The expected number of rows
        n_shooting: int
            The number of shooting points in the ocp
        """

        self.min.check_and_adjust_dimensions(n_elements, n_shooting, f"Bound min of key {self.key}")
        self.max.check_and_adjust_dimensions(n_elements, n_shooting, f"Bound max of key {self.key}")
        self.t = self.min.t
        self.n_shooting = self.min.n_shooting

    def concatenate(self, other: "Bounds"):
        """
        Vertical concatenate of two Bounds

        Parameters
        ----------
        other: Bounds
            The Bounds to concatenate with
        """

        if not isinstance(self.min, (MX, SX)) and not isinstance(other.min, (MX, SX)):
            self.min = PathCondition(np.concatenate((self.min, other.min)), interpolation=self.min.type)
        else:
            self.min = PathCondition(vertcat(self.min, other.min), interpolation=self.min.type)
        if not isinstance(self.max, (MX, SX)) and not isinstance(other.max, (MX, SX)):
            self.max = PathCondition(np.concatenate((self.max, other.max)), interpolation=self.max.type)
        else:
            self.max = PathCondition(vertcat(self.max, other.max), interpolation=self.max.type)

        self.type = self.min.type
        self.t = self.min.t
        self.extra_params = self.min.extra_params
        self.n_shooting = self.min.n_shooting

    def scale(self, scaling: float | np.ndarray | VariableScaling):
        """
        Scaling a Bound

        Parameters
        ----------
        scaling: float
            The scaling factor
        """

        return Bounds(None, self.min / scaling, self.max / scaling, interpolation=self.type)

    def __getitem__(self, slice_list: slice | list | tuple) -> "Bounds":
        """
        Allows to get from square brackets

        Parameters
        ----------
        slice_list: slice | list | tuple
            The slice to get

        Returns
        -------
        The bound sliced
        """
        if isinstance(slice_list, range):
            slice_list = slice(slice_list[0], slice_list[-1] + 1)

        if isinstance(slice_list, slice):
            t = self.min.t
            param = self.extra_params
            interpolation = self.type
            if interpolation == InterpolationType.CUSTOM:
                min_bound = self.min.custom_function
                max_bound = self.max.custom_function
            else:
                min_bound = np.array(self.min[slice_list.start : slice_list.stop : slice_list.step, :])
                max_bound = np.array(self.max[slice_list.start : slice_list.stop : slice_list.step, :])
            bounds_sliced = Bounds(
                min_bound=min_bound,
                max_bound=max_bound,
                interpolation=interpolation,
                slice_list=slice_list,
                t=t,
                **param,
            )
            # TODO: Verify if it is ok that slice_list arg sent is used only if it is a custom type
            #  (otherwise, slice_list is used before calling Bounds constructor)
            return bounds_sliced
        else:
            raise RuntimeError(
                "Invalid input for slicing bounds. Please note that columns should not be specified. "
                "Therefore, it should look like [a:b] or [a:b:c] where a is the starting index, "
                "b is the stopping index and c is the step for slicing."
            )

    def __setitem__(self, _slice: slice | list | tuple, value: np.ndarray | list | float):
        """
        Allows to set from square brackets

        Parameters
        ----------
        _slice: slice | list | tuple
            The slice where to put the data
        value: np.ndarray | float
            The value to set
        """

        self.min[_slice] = value
        self.max[_slice] = value

    def __bool__(self) -> bool:
        """
        Get if the Bounds is empty

        Returns
        -------
        If the Bounds is empty
        """

        return len(self.min) > 0

    @property
    def shape(self) -> list:
        """
        Get the size of the Bounds

        Returns
        -------
        The size of the Bounds
        """

        return self.min.shape

    @property
    def n_shooting(self):
        return self._n_shooting

    @n_shooting.setter
    def n_shooting(self, ns):
        self._n_shooting = ns
        self.min.n_shooting = ns
        self.max.n_shooting = ns


class BoundsList(OptionDict):
    """
    A list of Bounds if more than one is required

    Methods
    -------
    add(self, min_bound: PathCondition | np.ndarray | list | tuple = None,
            max_bound: PathCondition | np.ndarray | list | tuple = None, bounds: Bounds = None, **extra_arguments)
        Add a new constraint to the list, either [min_bound AND max_bound] OR [bounds] should be defined
    __getitem__(self, item) -> Bounds
        Get the ith option of the list
    print(self)
        Print the BoundsList to the console
    """

    def __init__(self, *args, **kwargs):
        super(BoundsList, self).__init__(sub_type=Bounds)
        self.type = InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT

    def __setitem__(self, key, value):
        if isinstance(value, (list, tuple)):
            self.add(key, min_bound=value[0], max_bound=value[1])
            return

        if isinstance(key, str):
            super(BoundsList, self).__setitem__(key, value)
            return

        raise KeyError("The required key in setting is invalid")

    def add(
        self,
        key,
        bounds: Bounds = None,
        min_bound: PathCondition | np.ndarray | float | list | tuple | Callable = None,
        max_bound: PathCondition | np.ndarray | float | list | tuple | Callable = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase: int = -1,
        **extra_arguments: Any,
    ):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        key: str
            The name of the optimization variable
        min_bound: PathCondition | np.ndarray | list | tuple
            The minimum path condition. If min_bound if defined, then max_bound must be so and bound should be None
        max_bound: [PathCondition, np.ndarray, list, tuple]
            The maximum path condition. If max_bound if defined, then min_bound must be so and bound should be None
        bounds: Bounds
            Copy a Bounds. If bounds is defined, min_bound and max_bound should be None
        interpolation: InterpolationType
            Type of interpolation do perform between shooting points
        phase: int
            The phase to add the bound to
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if bounds and (min_bound or max_bound):
            RuntimeError("min_bound/max_bound and bounds cannot be set alongside")
        if isinstance(bounds, (float, int)) and min_bound is None and max_bound is None:
            min_bound = bounds
            max_bound = bounds
            bounds = None
        if isinstance(bounds, Bounds):
            if phase == -1 and key in self.options[phase].keys():
                phase = len(self.options) if self.options[0] else 0

            previous_phase = bounds.phase
            bounds.phase = phase
            self.copy(bounds, key)
            bounds.phase = previous_phase
            self.type = bounds.type
        else:
            super(BoundsList, self)._add(
                key=key,
                min_bound=min_bound,
                max_bound=max_bound,
                option_type=Bounds,
                interpolation=interpolation,
                phase=phase,
                **extra_arguments,
            )
            self.type = interpolation

    def concatenate(self, other):
        for key in other.keys():
            self[key] = other[key]

    @property
    def param_when_copying(self):
        return {}

    def __getitem__(self, item: int | str) -> Any:
        """
        Get the ith option of the list

        Parameters
        ----------
        item: int
            The index of the option to get

        Returns
        -------
        The ith option of the list
        """

        return super(BoundsList, self).__getitem__(item)

    def print(self):
        """
        Print the BoundsList to the console
        """

        raise NotImplementedError("Printing of BoundsList is not ready yet")


class InitialGuess(OptionGeneric):
    """
    A placeholder for the initial guess

    Attributes
    ----------
    init: PathCondition
        The initial guess

    Methods
    -------
    check_and_adjust_dimensions(self, n_elements: int, n_shooting: int)
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay
    concatenate(self, other: "InitialGuess")
        Vertical concatenate of two InitialGuess
    scale(self, scaling: float)
        Scaling an InitialGuess
    __bool__(self) -> bool
        Get if the initial guess is empty
    shape(self) -> int
        Get the size of the initial guess
    __setitem__(self, _slice: slice | list | tuple, value: np.ndarray | list | float)
        Allows to set from square brackets
    """

    def __init__(
        self,
        key,
        initial_guess: np.ndarray | list | tuple | float | Callable = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        initial_guess: np.ndarray | list | tuple | float | Callable
            The initial guess
        interpolation: InterpolationType
            The type of interpolation of the initial guess
        parameters: dict
            Any extra parameters that is associated to the path condition
        """
        initial_guess = initial_guess if initial_guess is not None else ()

        if isinstance(initial_guess, PathCondition):
            self.init = initial_guess
        else:
            if "type" in parameters:
                interpolation = parameters["type"]
                del parameters["type"]
            self.init = PathCondition(initial_guess, interpolation=interpolation, **parameters)

        super(InitialGuess, self).__init__(**parameters)
        self.type = interpolation
        self.key = key

    def check_and_adjust_dimensions(self, n_elements: int, n_shooting: int):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        Parameters
        ----------
        n_elements: int
            The expected number of rows
        n_shooting: int
            The number of shooting points in the ocp
        """

        self.init.check_and_adjust_dimensions(n_elements, n_shooting, "InitialGuess")

    def concatenate(self, other: "InitialGuess"):
        """
        Vertical concatenate of two Bounds

        Parameters
        ----------
        other: InitialGuess
            The InitialGuess to concatenate with
        """

        self.init = PathCondition(
            np.concatenate((self.init, other.init)),
            interpolation=self.init.type,
        )

    def scale(self, scaling: float | np.ndarray | VariableScaling):
        """
        Scaling an InitialGuess

        Parameters
        ----------
        scaling: float
            The scaling factor
        """

        return InitialGuess(f"{self.key}_scaled", self.init / scaling, interpolation=self.type)

    def __bool__(self) -> bool:
        """
        Get if the InitialGuess is empty

        Returns
        -------
        If the InitialGuess is empty
        """

        return len(self.init) > 0

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the size of the InitialGuess

        Returns
        -------
        The size of the InitialGuess
        """

        return self.init.shape

    def __setitem__(self, _slice: slice | list | tuple, value: np.ndarray | list | float):
        """
        Allows to set from square brackets

        Parameters
        ----------
        _slice: slice | list | tuple
            The slice where to put the data
        value: np.ndarray | float
            The value to set
        """

        self.init[_slice] = value

    def add_noise(
        self,
        bounds: Bounds | BoundsList = None,
        magnitude: list | int | float | np.ndarray = 1,
        magnitude_type: MagnitudeType = MagnitudeType.RELATIVE,
        n_shooting: int = None,
        bound_push: list | int | float = 0.1,
        seed: int = None,
    ):
        """
        An interface for NoisedInitialGuess class

        Parameters
        ----------
        bounds: Bounds | BoundsList
            The bounds
        magnitude: list | int | float | np.ndarray
            The magnitude of the noised that must be applied between 0 and 1 (0 = no noise, 1 = continuous noise with a
            range defined between the bounds or between -magnitude and +magnitude for absolute noise
            If one value is given, applies this value to each initial guess
        magnitude_type: MagnitudeType
            The type of magnitude to apply : relative to the bounds or absolute
        n_shooting: int
            Number of nodes (second dim)
        bound_push: list | int | float
            The absolute minimal distance between the bound and the noised initial guess (if the originally generated
            initial guess is outside the bound-bound_push, this node is attributed the value bound-bound_push)
        seed: int
            The seed of the random generator
        """

        noised_guess = NoisedInitialGuess(
            key=self.key,
            initial_guess=self.init,
            interpolation=self.type,
            bounds=bounds,
            n_shooting=n_shooting,
            bound_push=bound_push,
            seed=seed,
            magnitude=magnitude,
            magnitude_type=magnitude_type,
            **self.params,
        )
        self.init = noised_guess.init
        self.type = noised_guess.type


class NoisedInitialGuess(InitialGuess):
    """
    A placeholder for the noised initial guess

    Attributes
    ----------
    init: InitialGuess
        The noised initial guess
    noise: np.array
        The noise to add to the initial guess
    noised_initial_guess: np.array
        The noised initial guess
    seed: int
        The seed of the random generator
    bound_push: float
        The bound to push the noise away from the bounds
    bounds: Bounds
        The bounds of the decision variables

    Methods
    -------
    _create_noise_matrix(self)
        Create the matrix of the initial guess + noise evaluated at each node
    """

    def __init__(
        self,
        key,
        initial_guess: np.ndarray | list | tuple | float | Callable | PathCondition | InitialGuess = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        bounds: Bounds | BoundsList = None,
        magnitude: list | int | float | np.ndarray = 1,
        magnitude_type: MagnitudeType = MagnitudeType.RELATIVE,
        n_shooting: int = None,
        bound_push: list | int | float = 0.1,
        seed: int = None,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        initial_guess: np.ndarray | list | tuple | float | Callable | PathCondition
            The initial guess
        init_interpolation: InterpolationType
            The type of interpolation of the initial guess
        bounds: Bounds | BoundsList
            The bounds
        magnitude: list | int | float | np.ndarray
            The magnitude of the noised that must be applied between 0 and 1 (0 = no noise, 1 = continuous noise with a
            range defined between the bounds or between -magnitude and +magnitude for absolute noise
            If one value is given, applies this value to each initial guess
        magnitude_type: MagnitudeType
            The type of magnitude to apply : relative to the bounds or absolute
        n_elements: int
            Number of elements (first dim)
        n_shooting: int
            Number of nodes (second dim)
        bound_push: list | int | float
            The absolute minimal distance between the bound and the noised initial guess (if the originally generated
            initial guess is outside the bound-bound_push, this node is attributed the value bound-bound_push)
        parameters: dict
            Any extra parameters that is associated to the path condition
        """

        if n_shooting is None:
            raise RuntimeError("n_shooting must be specified to generate noised initial guess")
        self.n_shooting = n_shooting

        if bounds is None:
            raise RuntimeError("'bounds' must be specified to generate noised initial guess")

        # test if initial_guess is a np.array, tuple or list
        if isinstance(initial_guess, (np.ndarray, tuple, list)):
            self.init = InitialGuess(key, initial_guess, interpolation=interpolation, **parameters)

        if isinstance(initial_guess, InitialGuess):
            interpolation = initial_guess.type
        if interpolation == InterpolationType.ALL_POINTS:
            bounds.n_shooting = initial_guess.shape[1]

        self.bounds = bounds
        self.n_elements = self.bounds.min.shape[0]
        self.bounds.check_and_adjust_dimensions(self.n_elements, n_shooting)
        self.bound_push = bound_push

        self.seed = seed

        self._check_magnitude(magnitude)
        self.noise = None

        self._create_noise_matrix(
            initial_guess=initial_guess, interpolation=interpolation, magnitude_type=magnitude_type, **parameters
        )

        super(NoisedInitialGuess, self).__init__(
            key=key,
            initial_guess=self.noised_initial_guess,
            interpolation=(
                interpolation
                if interpolation
                == InterpolationType.ALL_POINTS  # interpolation should always be done at each data point
                else InterpolationType.EACH_FRAME
            ),
            **parameters,
        )

    def _check_magnitude(self, magnitude: list | int | float | np.ndarray):
        """
        Check noise magnitude type and shape

        Parameters
        ----------
        magnitude: list | int | float | np.ndarray
            The magnitude of the noised that must be applied between 0 and 1 (0 = no noise, 1 = continuous noise with a
            standard deviation of the size of the range defined between the bounds
        """

        if isinstance(magnitude, (int, float)):
            magnitude = (magnitude,)

        if isinstance(magnitude, (list, tuple)):
            magnitude = np.array(magnitude)

        if len(magnitude.shape) == 1:
            magnitude = magnitude[:, np.newaxis]

        if magnitude.shape[0] == 1:
            magnitude = np.repeat(magnitude, self.n_elements, axis=0)

        if magnitude.shape[0] != 1 and magnitude.shape[0] != self.n_elements:
            raise ValueError("magnitude must be a float or list of float of the size of states or controls")

        self.magnitude = magnitude

    def _create_noise_matrix(
        self,
        initial_guess: np.ndarray | list | tuple | float | Callable | PathCondition | InitialGuess = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        magnitude_type: MagnitudeType = MagnitudeType.RELATIVE,
        polynomial_degree: int = 1,
        **parameters: Any,
    ):
        """
        Create the matrix of the initial guess + noise evaluated at each node

        Parameters
        ----------
        initial_guess: np.ndarray | list | tuple | float | Callable | PathCondition | InitialGuess
            The initial guess
        interpolation: InterpolationType
            The type of interpolation of the initial guess
        magnitude_type: MagnitudeType
            The type of magnitude to apply : relative to the bounds or an absolute value
        polynomial_degree: int
            The degree of the polynomial used in collocations
        """

        if isinstance(initial_guess, InitialGuess):
            tp = initial_guess
        else:
            tp = InitialGuess("init", initial_guess, interpolation=interpolation, **parameters)

        if tp.type == InterpolationType.EACH_FRAME:
            n_columns = self.n_shooting - 1  # As it will add 1 by itself later
        elif tp.type == InterpolationType.ALL_POINTS:
            n_columns = tp.shape[1] - 1  # As it will add 1 by itself later
        else:
            n_columns = self.n_shooting

        ns = n_columns + 1 if interpolation == InterpolationType.ALL_POINTS else self.n_shooting
        bounds_min_matrix = np.zeros((self.n_elements, ns))
        bounds_max_matrix = np.zeros((self.n_elements, ns))
        self.bounds.min.n_shooting = ns
        self.bounds.max.n_shooting = ns
        for shooting_point in range(ns):
            if shooting_point == ns - 1:
                bounds_min_matrix[:, shooting_point] = self.bounds.min.evaluate_at(
                    shooting_point + 1, repeat=polynomial_degree
                )
                bounds_max_matrix[:, shooting_point] = self.bounds.max.evaluate_at(
                    shooting_point + 1, repeat=polynomial_degree
                )
            else:
                bounds_min_matrix[:, shooting_point] = self.bounds.min.evaluate_at(
                    shooting_point, repeat=polynomial_degree
                )
                bounds_max_matrix[:, shooting_point] = self.bounds.max.evaluate_at(
                    shooting_point, repeat=polynomial_degree
                )

        if self.seed is not None:
            np.random.seed(self.seed)

        self.noise = (
            np.random.random((self.n_elements, ns)) * 2 - 1  # random noise
        ) * self.magnitude  # magnitude of the noise within the range defined by the bounds
        if magnitude_type == MagnitudeType.RELATIVE:
            self.noise *= bounds_max_matrix - bounds_min_matrix

        # building the noised initial guess
        if initial_guess is None:
            initial_guess_matrix = (bounds_min_matrix + bounds_max_matrix) / 2
        else:
            tp.check_and_adjust_dimensions(self.n_elements, n_columns)
            initial_guess_matrix = np.zeros((self.n_elements, ns))
            for shooting_point in range(ns):
                initial_guess_matrix[:, shooting_point] = tp.init.evaluate_at(shooting_point, repeat=polynomial_degree)

        init_instance = InitialGuess(
            "noised",
            initial_guess_matrix,
            interpolation=(
                interpolation if interpolation == InterpolationType.ALL_POINTS else InterpolationType.EACH_FRAME
            ),
        )

        self.noised_initial_guess = init_instance.init + self.noise

        # Make sure initial guess is inside the bounds at a distance of bound push
        idx_min = np.where(self.noised_initial_guess < bounds_min_matrix)
        idx_max = np.where(self.noised_initial_guess > bounds_max_matrix)
        self.noised_initial_guess[idx_min] = bounds_min_matrix[idx_min] + self.bound_push
        self.noised_initial_guess[idx_max] = bounds_max_matrix[idx_max] - self.bound_push


class InitialGuessList(OptionDict):
    """
    A list of InitialGuess if more than one is required

    Methods
    -------
    add(self, initial_guess: PathCondition | np.ndarray | list | tuple, **extra_arguments)
        Add a new initial guess to the list
    print(self)
        Print the InitialGuessList to the console
    _check_type_and_format_bounds(bounds, nb_phases)
        Check bounds type and format
    _check_type_and_format_magnitude(self, nb_phases)
        Check magnitude type and format
    _check_type_and_format_bound_push(self, nb_phases)
        Check bound_push type and format
    _check_type_and_format_seed(self, nb_phases)
        Check seed type and format
    _check_type_and_format_parameters(self, nb_phases)
        Check parameters type and format
    add_noise
        Add noise to each initial guesses from an InitialGuessList
    """

    def __init__(self, *args, **kwargs):
        super(InitialGuessList, self).__init__(sub_type=InitialGuess)
        self.type = InterpolationType.CONSTANT

    def add(
        self,
        key,
        initial_guess: InitialGuess | np.ndarray | list | tuple | Callable = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        phase: int = -1,
        **extra_arguments: Any,
    ):
        """
        Add a new initial guess to the list

        Parameters
        ----------
        key: str
            The name of the optimization variable
        initial_guess: InitialGuess | np.ndarray | list | tuple
            The initial guess to add
        extra_arguments: dict
            Any parameters to pass to the Bounds
        interpolation: InterpolationType
            Type of interpolation do perform between shooting points
        phase: int
            The phase to add the bound to
        """

        if isinstance(initial_guess, InitialGuess):
            if phase == -1 and key in self.options[phase].keys():
                phase = len(self.options) if self.options[0] else 0

            previous_phase = initial_guess.phase
            initial_guess.phase = phase
            self.copy(initial_guess, key)
            initial_guess.phase = previous_phase
            self.type = initial_guess.type
        else:
            super(InitialGuessList, self)._add(
                key,
                initial_guess=initial_guess,
                option_type=InitialGuess,
                interpolation=interpolation,
                phase=phase,
                **extra_arguments,
            )
            self.type = interpolation

    def concatenate(self, other):
        for key in other.keys():
            self[key] = other[key]

    def print(self):
        """
        Print the InitialGuessList to the console
        """
        raise NotImplementedError("Printing of InitialGuessList is not ready yet")

    @staticmethod
    def _check_type_and_format_bounds(bounds, nb_phases):
        """
        Check bounds type and format

        Parameters
        ----------
        nb_phases
            The number of phases
        """
        if bounds is None:
            raise ValueError("bounds must be specified to generate noised initial guess")

        if len(bounds) != nb_phases:
            raise ValueError(f"Invalid size of 'bounds', 'bounds' must be size {nb_phases}")

        return bounds

    @staticmethod
    def _check_type_and_format_magnitude(magnitude, nb_phases, keys):
        """
        Check magnitude type and format

        Parameters
        ----------
        nb_phases
            The number of phases
        """
        if magnitude is None:
            raise ValueError("'magnitude' must be specified to generate noised initial guess")

        if not isinstance(magnitude, (int, float, dict, list)):
            raise ValueError("'magnitude' must be an instance of int, float, list, or ndarray")

        if isinstance(magnitude, (int, float)):
            magnitude = [magnitude]

        # Deal with the list[int|float] case
        if isinstance(magnitude, list):
            for i, mag in enumerate(magnitude):
                if isinstance(mag, (int, float)):
                    magnitude[i] = {key: mag for key in keys}

        if isinstance(magnitude, dict):
            magnitude = [magnitude for _ in range(nb_phases)]

        if isinstance(magnitude, list):
            if len(magnitude) == 1:
                magnitude = [magnitude[0] for _ in range(nb_phases)]
            elif len(magnitude) != nb_phases:
                raise ValueError(f"Invalid size of 'magnitude', 'magnitude' as list must be size 1 or {nb_phases}")

        for mag in magnitude:
            for key in keys:
                if key not in mag.keys():
                    raise ValueError(f"Magnitude of all the elements must be specified, but {key} is missing")

        return magnitude

    @staticmethod
    def _check_type_and_format_bound_push(bound_push, nb_phases):
        """
        Check bound_push type and format

        Parameters
        ----------
        nb_phases
            The number of phases
        """
        if bound_push is None:
            raise ValueError("'bound_push' must be specified to generate noised initial guess")

        if not isinstance(bound_push, (int, float, list, ndarray)):
            raise ValueError("'bound_push' must be an instance of int, float, list or ndarray")

        if isinstance(bound_push, (float, int)):
            bound_push = [bound_push for j in range(nb_phases)]

        if isinstance(bound_push, list):
            if len(bound_push) == 1:
                bound_push = [bound_push[0] for j in range(nb_phases)]
            elif len(bound_push) != nb_phases:
                raise ValueError(f"Invalid size of 'bound_push', 'bound_push' as list must be size 1 or {nb_phases}")

        if isinstance(bound_push, ndarray):
            if bound_push.shape.__len__() > 1:
                # if todo: prepare the 2dimensional absolute_noise
                raise ValueError("'bound_push' must be a 1 dimension array'")
            if bound_push.shape == ():
                bound_push = bound_push[np.newaxis]
                bound_push = [bound_push[0] for j in range(nb_phases)]
            elif bound_push.shape[0] != nb_phases:
                raise ValueError(f"Invalid size of 'bound_push', 'bound_push' as array must be size 1 or {nb_phases}")

        return bound_push

    @staticmethod
    def _check_type_and_format_seed(seed, nb_phases, keys):
        """
        Check seed type and format

        Parameters
        ----------
        nb_phases
            The number of phases
        """

        if seed is not None and not isinstance(seed, (int, dict, list)):
            raise ValueError("Seed must be an integer, dict or a list of these")

        if seed is None:
            seed = {key: None for key in keys}

        if isinstance(seed, int):
            seed = [seed]

        # Deal with the list[int] case
        if isinstance(seed, list):
            for i, s in enumerate(seed):
                if isinstance(s, int):
                    seed[i] = {key: s if i == 0 else None for i, key in enumerate(keys)}

        if isinstance(seed, dict):
            seed = [seed]

        if isinstance(seed, list):
            if len(seed) == 1:
                empty = {key: None for key in keys}
                seed = [seed[0] if j == 0 else empty for j in range(nb_phases)]
            elif len(seed) != nb_phases:
                raise ValueError(f"Invalid size of 'seed', 'seed' as list must be size 1 or {nb_phases}")

        return seed

    def add_noise(
        self,
        bounds: BoundsList = None,
        n_shooting: int | List[int] | Tuple[int] = None,
        magnitude: int | float | dict | list = None,
        magnitude_type: MagnitudeType = MagnitudeType.RELATIVE,
        bound_push: int | float | List[int] | List[float] | ndarray = 0.1,
        seed: int | dict | list = None,
    ):
        """
        Add noise to each initial guesses from an InitialGuessList

        Parameters
        ----------
        bounds:
            The bounds of each phase
        n_shooting:
            Number of nodes (second dim) per initial guess
        magnitude:
            The magnitude of the noised that must be applied between 0 and 1 (0 = no noise, 1 = continuous noise with a
            range defined between the bounds or between -magnitude and +magnitude for absolute noise
            If one value is given, applies this value to each initial guess
        magnitude_type:
            The type of magnitude to apply : relative to the bounds or absolute
        bound_push:
            The absolute minimal distance between the bound and the noised initial guess (if the originally generated
            initial guess is outside the bound-bound_push, this node is attributed the value bound-bound_push).
            If one value is given, applies this value to each initial guess
        seed:
            The seed of the random generator
            If one value is given, applies this value to each initial guess
        """

        nb_phases = len(self)  # number of init guesses, i.e. number of phases

        if n_shooting is None:
            raise ValueError("n_shooting must be specified to generate noised initial guess")

        if nb_phases == 1 and isinstance(n_shooting, int):
            n_shooting = [n_shooting]
        if len(n_shooting) != nb_phases:
            raise ValueError(f"Invalid size of 'n_shooting', 'n_shooting' must be len {nb_phases}")

        bounds = self._check_type_and_format_bounds(bounds, nb_phases)
        magnitude = self._check_type_and_format_magnitude(magnitude, nb_phases, self.keys())
        bound_push = self._check_type_and_format_bound_push(bound_push, nb_phases)
        seed = self._check_type_and_format_seed(seed, nb_phases, self.keys())

        for i in range(nb_phases):
            for j, key in enumerate(self[i].keys()):
                self[i][key].add_noise(
                    bounds=bounds[i][key],
                    n_shooting=n_shooting[i],
                    magnitude=magnitude[i][key],
                    magnitude_type=magnitude_type,
                    bound_push=bound_push[i],
                    seed=seed[i][key],
                )

    @property
    def param_when_copying(self):
        return {"type": self.type}
