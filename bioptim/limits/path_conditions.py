from typing import Union, Callable, Any

import biorbd_casadi as biorbd
import numpy as np
from casadi import MX, SX, vertcat
from scipy.interpolate import interp1d
from numpy import array, ndarray

from ..misc.enums import InterpolationType
from ..misc.mapping import BiMapping, BiMappingList
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric


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
        input_array: Union[np.ndarray, Callable],
        t: list = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        slice_list: Union[slice, list, tuple] = None,
        **extra_params,
    ):
        """
        Parameters
        ----------
        input_array: Union[np.ndarray, Callable]
            The matrix of interpolation, rows are the components, columns are the time
        t: list[float]
            The time stamps
        interpolation: InterpolationType
            The type of interpolation. It determines how many timestamps are required
        slice_list: Union[slice, list, tuple]
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
                    f"Invalid number of column for InterpolationType.LINEAR_CONTINUOUS "
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
            slice_list = self.slice_list
            if slice_list is not None:
                val_size = self.custom_function(0, **self.extra_params)[
                    slice_list.start : slice_list.stop : slice_list.step
                ].shape[0]
            else:
                val_size = self.custom_function(0, **self.extra_params).shape[0]
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

    def evaluate_at(self, shooting_point: int):
        """
        Evaluate the interpolation at a specific shooting point

        Parameters
        ----------
        shooting_point: int
            The shooting point to evaluate the path condition at

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
            return self[:, 0] + (self[:, 1] - self[:, 0]) * shooting_point / self.n_shooting
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
                return self.custom_function(shooting_point, **self.extra_params)
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
    scale(self, scaling: Union[float, np.ndarray])
        Scaling a Bound
    __getitem__(self, slice_list: slice) -> "Bounds"
        Allows to get from square brackets
    __setitem__(self, slice: slice, value: Union[np.ndarray, list, float])
        Allows to set from square brackets
    __bool__(self) -> bool
        Get if the Bounds is empty
    shape(self) -> int
        Get the size of the Bounds
    """

    def __init__(
        self,
        min_bound: Union[Callable, PathCondition, np.ndarray, list, tuple, float] = None,
        max_bound: Union[Callable, PathCondition, np.ndarray, list, tuple, float] = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        slice_list: Union[slice, list, tuple] = None,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        min_bound: Union[Callable, PathCondition, np.ndarray, list, tuple]
            The minimal bound
        max_bound: Union[Callable, PathCondition, np.ndarray, list, tuple]
            The maximal bound
        interpolation: InterpolationType
            The type of interpolation of the bound
        slice_list: Union[slice, list, tuple]
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

        self.min.check_and_adjust_dimensions(n_elements, n_shooting, "Bound min")
        self.max.check_and_adjust_dimensions(n_elements, n_shooting, "Bound max")
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

    def scale(self, scaling: Union[float, np.ndarray]):
        """
        Scaling a Bound

        Parameters
        ----------
        scaling: float
            The scaling factor
        """

        self.min /= scaling
        self.max /= scaling
        return

    def __getitem__(self, slice_list: Union[slice, list, tuple]) -> "Bounds":
        """
        Allows to get from square brackets

        Parameters
        ----------
        slice_list: Union[slice, list, tuple]
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

    def __setitem__(self, _slice: Union[slice, list, tuple], value: Union[np.ndarray, list, float]):
        """
        Allows to set from square brackets

        Parameters
        ----------
        _slice: Union[slice, list, tuple]
            The slice where to put the data
        value: Union[np.ndarray, float]
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


class BoundsList(UniquePerPhaseOptionList):
    """
    A list of Bounds if more than one is required

    Methods
    -------
    add(self, min_bound: Union[PathCondition, np.ndarray, list, tuple] = None,
            max_bound: Union[PathCondition, np.ndarray, list, tuple] = None, bounds: Bounds = None, **extra_arguments)
        Add a new constraint to the list, either [min_bound AND max_bound] OR [bounds] should be defined
    __getitem__(self, item) -> Bounds
        Get the ith option of the list
    print(self)
        Print the BoundsList to the console
    """

    def add(
        self,
        min_bound: Union[PathCondition, np.ndarray, list, tuple] = None,
        max_bound: Union[PathCondition, np.ndarray, list, tuple] = None,
        bounds: Bounds = None,
        **extra_arguments,
    ):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        min_bound: Union[PathCondition, np.ndarray, list, tuple]
            The minimum path condition. If min_bound if defined, then max_bound must be so and bound should be None
        max_bound: [PathCondition, np.ndarray, list, tuple]
            The maximum path condition. If max_bound if defined, then min_bound must be so and bound should be None
        bounds: Bounds
            Copy a Bounds. If bounds is defined, min_bound and max_bound should be None
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if bounds and (min_bound or max_bound):
            RuntimeError("min_bound/max_bound and bounds cannot be set alongside")
        if isinstance(bounds, Bounds):
            if bounds.phase == -1:
                bounds.phase = len(self.options) if self.options[0] else 0
            self.copy(bounds)
        else:
            super(BoundsList, self)._add(
                min_bound=min_bound, max_bound=max_bound, option_type=Bounds, **extra_arguments
            )

    def __getitem__(self, item) -> Bounds:
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


class QAndQDotBounds(Bounds):
    """
    Specialized Bounds that reads a model to automatically extract q and qdot bounds
    """

    def __init__(
        self,
        biorbd_model,
        dof_mappings: Union[BiMapping, BiMappingList] = None,
    ):
        """
        Parameters
        ----------
        biorbd_model: biorbd.Model
            A reference to the model
        dof_mappings: BiMappingList
            The mapping of q and qdot (if only q, then qdot = q)
        """
        if dof_mappings is None:
            dof_mappings = {}

        if biorbd_model.nbQuat() > 0:
            if "q" in dof_mappings and "qdot" not in dof_mappings:
                raise RuntimeError(
                    "It is not possible to provide a q_mapping but not a qdot_mapping if the model have quaternion"
                )
            elif "q" not in dof_mappings and "qdot" in dof_mappings:
                raise RuntimeError(
                    "It is not possible to provide a qdot_mapping but not a q_mapping if the model have quaternion"
                )

        if "q" not in dof_mappings:
            dof_mappings["q"] = BiMapping(range(biorbd_model.nbQ()), range(biorbd_model.nbQ()))

        if "qdot" not in dof_mappings:
            if biorbd_model.nbQuat() > 0:
                dof_mappings["qdot"] = BiMapping(range(biorbd_model.nbQdot()), range(biorbd_model.nbQdot()))
            else:
                dof_mappings["qdot"] = dof_mappings["q"]

        q_ranges = []
        qdot_ranges = []
        for i in range(biorbd_model.nbSegment()):
            segment = biorbd_model.segment(i)
            q_ranges += [q_range for q_range in segment.QRanges()]
            qdot_ranges += [qdot_range for qdot_range in segment.QDotRanges()]

        x_min = [q_ranges[i].min() for i in dof_mappings["q"].to_first.map_idx] + [
            qdot_ranges[i].min() for i in dof_mappings["qdot"].to_first.map_idx
        ]
        x_max = [q_ranges[i].max() for i in dof_mappings["q"].to_first.map_idx] + [
            qdot_ranges[i].max() for i in dof_mappings["qdot"].to_first.map_idx
        ]

        super(QAndQDotBounds, self).__init__(min_bound=x_min, max_bound=x_max)


class QAndQDotAndQDDotBounds(QAndQDotBounds):
    """
    Specialized Bounds that reads a model to automatically extract q, qdot and qddot bounds
    """

    def __init__(
        self,
        biorbd_model,
        dof_mappings: Union[BiMapping, BiMappingList] = None,
    ):
        """
        Parameters
        ----------
        biorbd_model: biorbd.Model
            A reference to the model
        dof_mappings: BiMappingList
            The mapping of q and qdot (if only q, then qdot = q)
        """

        super(QAndQDotAndQDDotBounds, self).__init__(biorbd_model=biorbd_model, dof_mappings=dof_mappings)

        if dof_mappings is None:
            dof_mappings = {}

        if "q" not in dof_mappings:
            dof_mappings["q"] = BiMapping(range(biorbd_model.nbQ()), range(biorbd_model.nbQ()))

        if "qdot" not in dof_mappings:
            if biorbd_model.nbQuat() > 0:
                dof_mappings["qdot"] = BiMapping(range(biorbd_model.nbQdot()), range(biorbd_model.nbQdot()))
            else:
                dof_mappings["qdot"] = dof_mappings["q"]

        if "qddot" not in dof_mappings:
            if biorbd_model.nbQuat() > 0:
                dof_mappings["qddot"] = BiMapping(range(biorbd_model.nbQddot()), range(biorbd_model.nbQddot()))
            else:
                dof_mappings["qddot"] = dof_mappings["qdot"]

        qddot_ranges = []
        for i in range(biorbd_model.nbSegment()):
            segment = biorbd_model.segment(i)
            qddot_ranges += [qddot_range for qddot_range in segment.QDDotRanges()]

        x_min = [qddot_ranges[i].min() for i in dof_mappings["qddot"].to_first.map_idx]
        x_max = [qddot_ranges[i].max() for i in dof_mappings["qddot"].to_first.map_idx]

        self.concatenate(Bounds(x_min, x_max))


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
    __setitem__(self, _slice: Union[slice, list, tuple], value: Union[np.ndarray, list, float])
        Allows to set from square brackets
    """

    def __init__(
        self,
        initial_guess: Union[np.ndarray, list, tuple, float, Callable] = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        initial_guess: Union[np.ndarray, list, tuple, float, Callable]
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
            self.init = PathCondition(initial_guess, interpolation=interpolation, **parameters)

        super(InitialGuess, self).__init__(**parameters)
        self.type = interpolation

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

    def scale(self, scaling: float):
        """
        Scaling an InitialGuess

        Parameters
        ----------
        scaling: float
            The scaling factor
        """
        self.init /= scaling
        return

    def __bool__(self) -> bool:
        """
        Get if the InitialGuess is empty

        Returns
        -------
        If the InitialGuess is empty
        """

        return len(self.init) > 0

    @property
    def shape(self) -> int:
        """
        Get the size of the InitialGuess

        Returns
        -------
        The size of the InitialGuess
        """

        return self.init.shape

    def __setitem__(self, _slice: Union[slice, list, tuple], value: Union[np.ndarray, list, float]):
        """
        Allows to set from square brackets

        Parameters
        ----------
        _slice: Union[slice, list, tuple]
            The slice where to put the data
        value: Union[np.ndarray, float]
            The value to set
        """

        self.init[_slice] = value


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
        initial_guess: Union[np.ndarray, list, tuple, float, Callable, PathCondition, InitialGuess] = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        bounds: Union[Bounds, BoundsList, QAndQDotBounds] = None,
        noise_magnitude: Union[list, int, float, np.ndarray] = 1,
        n_shooting: int = None,
        bound_push: Union[list, int, float] = 0.1,
        seed: int = None,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        initial_guess: Union[np.ndarray, list, tuple, float, Callable, PathCondition]
            The initial guess
        init_interpolation: InterpolationType
            The type of interpolation of the initial guess
        bounds: Union[Bounds, BoundsList, QAndQDotBounds]
            The bounds
        noise_magnitude: Union[list, int, float]
            The magnitude of the noised that must be applied between 0 and 1 (0 = no noise, 1 = gaussian noise with a
            standard deviation of the size of the range defined between the bounds
        n_elements: int
            Number of elements (first dim)
        n_shooting: int
            Number of nodes (second dim)
        bound_push: Union[list, int, float]
            The absolute minimal distance between the bound and the noised initial guess (if the originally generated
            initial guess is outside the bound-bound_push, this node is attributed the value bound-bound_push)
        parameters: dict
            Any extra parameters that is associated to the path condition
        """

        if n_shooting is None:
            raise RuntimeError("n_shooting must be specified to generate noised initial guess")
        self.n_shooting = n_shooting + 1

        if bounds is None:
            raise RuntimeError("'bounds' must be specified to generate noised initial guess")

        # test if initial_guess is a np.array, tuple or list
        if isinstance(initial_guess, (np.ndarray, tuple, list)):
            self.init = InitialGuess(initial_guess, interpolation=interpolation, **parameters)

        if isinstance(initial_guess, InitialGuess):
            interpolation = initial_guess.type
        if interpolation == InterpolationType.ALL_POINTS:
            bounds.n_shooting = initial_guess.shape[1]

        self.bounds = bounds
        self.n_elements = self.bounds.min.shape[0]
        self.bounds.check_and_adjust_dimensions(self.n_elements, n_shooting)
        self.bound_push = bound_push

        self.seed = seed

        self._check_noise_magnitude(noise_magnitude)
        self.noise = None

        self._create_noise_matrix(initial_guess=initial_guess, interpolation=interpolation, **parameters)

        super(NoisedInitialGuess, self).__init__(
            initial_guess=self.noised_initial_guess,
            interpolation=interpolation
            if interpolation == InterpolationType.ALL_POINTS  # interpolation should always be done at each data point
            else InterpolationType.EACH_FRAME,
            **parameters,
        )

    def _check_noise_magnitude(self, noise_magnitude: Union[list, int, float, np.ndarray]):
        """
        Check noise magnitude type and shape

        Parameters
        ----------
        noise_magnitude: Union[list, int, float, np.ndarray]
            The magnitude of the noised that must be applied between 0 and 1 (0 = no noise, 1 = gaussian noise with a
            standard deviation of the size of the range defined between the bounds
        """

        if isinstance(noise_magnitude, (int, float)):
            noise_magnitude = (noise_magnitude,)

        if isinstance(noise_magnitude, (list, tuple)):
            noise_magnitude = np.array(noise_magnitude)

        if len(noise_magnitude.shape) == 1:
            noise_magnitude = noise_magnitude[:, np.newaxis]

        if noise_magnitude.shape[0] == 1:
            noise_magnitude = np.repeat(noise_magnitude, self.n_elements, axis=0)

        if noise_magnitude.shape[0] != 1 and noise_magnitude.shape[0] != self.n_elements:
            raise ValueError("noise_magnitude must be a float or list of float of the size of states or controls")

        self.noise_magnitude = noise_magnitude

    def _create_noise_matrix(
        self,
        initial_guess: Union[np.ndarray, list, tuple, float, Callable, PathCondition, InitialGuess] = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **parameters: Any,
    ):
        """
        Create the matrix of the initial guess + noise evaluated at each node
        """

        if isinstance(initial_guess, InitialGuess):
            tp = initial_guess
        else:
            tp = InitialGuess(initial_guess, interpolation=interpolation, **parameters)
        if tp.type == InterpolationType.EACH_FRAME:
            n_shooting = self.n_shooting - 1
        elif tp.type == InterpolationType.ALL_POINTS:
            n_shooting = tp.shape[1] - 1
        else:
            n_shooting = self.n_shooting

        ns = n_shooting + 1 if interpolation == InterpolationType.ALL_POINTS else self.n_shooting
        bounds_min_matrix = np.zeros((self.n_elements, ns))
        bounds_max_matrix = np.zeros((self.n_elements, ns))
        self.bounds.min.n_shooting = ns
        self.bounds.max.n_shooting = ns
        for shooting_point in range(ns):
            if shooting_point == ns - 1:
                bounds_min_matrix[:, shooting_point] = self.bounds.min.evaluate_at(shooting_point + 1)
                bounds_max_matrix[:, shooting_point] = self.bounds.max.evaluate_at(shooting_point + 1)
            else:
                bounds_min_matrix[:, shooting_point] = self.bounds.min.evaluate_at(shooting_point)
                bounds_max_matrix[:, shooting_point] = self.bounds.max.evaluate_at(shooting_point)

        if self.seed:
            np.random.seed(self.seed)

        self.noise = (
            (np.random.random((self.n_elements, ns)) * 2 - 1)  # random noise
            * self.noise_magnitude  # magnitude of the noise within the range defined by the bounds
            * (bounds_max_matrix - bounds_min_matrix)  # scale the noise to the range of bounds
        )

        # building the noised initial guess
        if initial_guess is None:
            initial_guess_matrix = (bounds_min_matrix + bounds_max_matrix) / 2
        else:
            tp.check_and_adjust_dimensions(self.n_elements, n_shooting)
            initial_guess_matrix = np.zeros((self.n_elements, ns))
            for shooting_point in range(ns):
                initial_guess_matrix[:, shooting_point] = tp.init.evaluate_at(shooting_point)

        init_instance = InitialGuess(
            initial_guess_matrix,
            interpolation=interpolation
            if interpolation == InterpolationType.ALL_POINTS
            else InterpolationType.EACH_FRAME,
        )

        self.noised_initial_guess = init_instance.init + self.noise
        for shooting_point in range(ns):
            too_small_index = np.where(
                self.noised_initial_guess[:, shooting_point] < bounds_min_matrix[:, shooting_point]
            )
            too_big_index = np.where(
                self.noised_initial_guess[:, shooting_point] > bounds_max_matrix[:, shooting_point]
            )
            self.noised_initial_guess[too_small_index, shooting_point] = (
                bounds_min_matrix[too_small_index, shooting_point] + self.bound_push
            )
            self.noised_initial_guess[too_big_index, shooting_point] = (
                bounds_max_matrix[too_big_index, shooting_point] - self.bound_push
            )


class InitialGuessList(UniquePerPhaseOptionList):
    """
    A list of InitialGuess if more than one is required

    Methods
    -------
    add(self, initial_guess: Union[PathCondition, np.ndarray, list, tuple], **extra_arguments)
        Add a new initial guess to the list
    print(self)
        Print the InitialGuessList to the console
    """

    def add(self, initial_guess: Union[InitialGuess, np.ndarray, list, tuple], **extra_arguments: Any):
        """
        Add a new initial guess to the list

        Parameters
        ----------
        initial_guess: Union[InitialGuess, np.ndarray, list, tuple]
            The initial guess to add
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if isinstance(initial_guess, InitialGuess):
            self.copy(initial_guess)
        else:
            super(InitialGuessList, self)._add(initial_guess=initial_guess, option_type=InitialGuess, **extra_arguments)

    def print(self):
        """
        Print the InitialGuessList to the console
        """
        raise NotImplementedError("Printing of InitialGuessList is not ready yet")
