import numpy as np
from scipy.interpolate import interp1d

from ..misc.enums import InterpolationType
from ..misc.mapping import BidirectionalMapping, Mapping
from ..misc.options_lists import UniquePerPhaseOptionList, OptionGeneric


class PathCondition(np.ndarray):
    """Sets path constraints"""

    def __new__(cls, input_array, t=None, interpolation=InterpolationType.CONSTANT, slice_list=None, **extra_params):
        """
        Interpolates path conditions with the chosen interpolation type.
        :param input_array: Array of path conditions (initial guess). (list)
        :param interpolation: Type of interpolation. (Instance of InterpolationType)
        (InterpolationType.CONSTANT, InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT, InterpolationType.LINEAR_CONTINUOUS
        or InterpolationType.EACH_FRAME)
        :return: obj -> Objective. (?)
        """
        # Check and reinterpret input
        if interpolation == InterpolationType.CUSTOM:
            if not callable(input_array):
                raise TypeError("The input when using InterpolationType.CUSTOM should be a callable function")
            custom_function = input_array
            input_array = np.array(())
        else:
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
                    f"Invalid number of column for InterpolationType.LINEAR_CONTINUOUS (ncols = {input_array.shape[1]}), "
                    f"the expected number of column is 2"
                )
        elif interpolation == InterpolationType.EACH_FRAME:
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
        obj = np.asarray(input_array).view(cls)

        # Additional information (do not forget to update __reduce__ and __setstate__)
        obj.nb_shooting = None
        obj.type = interpolation
        obj.t = t
        obj.extra_params = extra_params
        obj.slice_list = slice_list
        if interpolation == InterpolationType.CUSTOM:
            obj.custom_function = custom_function

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.nb_shooting = getattr(obj, "nb_shooting", None)
        self.type = getattr(obj, "type", None)

    def __reduce__(self):
        pickled_state = super(PathCondition, self).__reduce__()
        new_state = pickled_state[2] + (self.nb_shooting, self.type, self.t, self.extra_params, self.slice_list)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.nb_shooting = state[-5]
        self.type = state[-4]
        self.t = state[-3]
        self.extra_params = state[-2]
        self.slice_list = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(PathCondition, self).__setstate__(state[0:-5])

    def check_and_adjust_dimensions(self, nb_elements, nb_shooting, element_type):
        """
        Raises errors on the dimensions of the elements to be interpolated.
        :param nb_elements: Number of elements to be interpolated. (integer)
        :param nb_shooting: Number of shooting points. (integer)
        :param element_type: Type of interpolation. (instance of InterpolationType class)
        """
        if (
            self.type == InterpolationType.CONSTANT
            or self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            or self.type == InterpolationType.LINEAR
            or self.type == InterpolationType.SPLINE
            or self.type == InterpolationType.CUSTOM
        ):
            self.nb_shooting = nb_shooting
        elif self.type == InterpolationType.EACH_FRAME:
            self.nb_shooting = nb_shooting + 1
        else:
            if self.nb_shooting != nb_shooting:
                raise RuntimeError(
                    f"Invalid number of shooting ({self.nb_shooting}), the expected number is {nb_shooting}"
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
        if val_size != nb_elements:
            raise RuntimeError(f"Invalid number of {element_type} ({val_size}), the expected size is {nb_elements}")

        if self.type == InterpolationType.EACH_FRAME:
            if self.shape[1] != self.nb_shooting:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.EACH_FRAME (ncols = {self.shape[1]}), "
                    f"the expected number of column is {self.nb_shooting}"
                )

    def evaluate_at(self, shooting_point):
        """
        Discriminates first and last nodes and evaluates self in function of the interpolation type.
        :param shooting_point: Number of shooting points. (integer)
        """
        if self.nb_shooting is None:
            raise RuntimeError(f"check_and_adjust_dimensions must be called at least once before evaluating at")

        if self.type == InterpolationType.CONSTANT:
            return self[:, 0]
        elif self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if shooting_point == 0:
                return self[:, 0]
            elif shooting_point == self.nb_shooting:
                return self[:, 2]
            elif shooting_point > self.nb_shooting:
                raise RuntimeError("shooting point too high")
            else:
                return self[:, 1]
        elif self.type == InterpolationType.LINEAR:
            return self[:, 0] + (self[:, 1] - self[:, 0]) * shooting_point / self.nb_shooting
        elif self.type == InterpolationType.EACH_FRAME:
            return self[:, shooting_point]
        elif self.type == InterpolationType.SPLINE:
            spline = interp1d(self.t, self)
            return spline(shooting_point / self.nb_shooting * (self.t[-1] - self.t[0]))
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


class BoundsOption(OptionGeneric):
    def __init__(self, bounds, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT, **params):
        if not isinstance(bounds, Bounds):
            bounds = Bounds(min_bound=bounds[0], max_bound=bounds[1], interpolation=interpolation, **params)

        super(BoundsOption, self).__init__(**params)
        self.bounds = bounds
        self.min = self.bounds.min
        self.max = self.bounds.max

    def __getitem__(self, slice):
        return self.min[slice], self.max[slice]

    def __setitem__(self, slice, value):
        self.bounds[slice] = value

    @property
    def shape(self):
        return self.bounds.shape


class BoundsList(UniquePerPhaseOptionList):
    def add(
        self,
        bounds,
        **extra_arguments,
    ):
        if isinstance(bounds, BoundsOption):
            self.copy(bounds)
        else:
            super(BoundsList, self)._add(bounds=bounds, option_type=BoundsOption, **extra_arguments)

    def __getitem__(self, item):
        return super(BoundsList, self).__getitem__(item).bounds

    def __next__(self):
        return super(BoundsList, self).__next__().bounds


class Bounds:
    """
    Organizes bounds of states("X"), controls("U") and "V".
    """

    def __init__(
        self,
        min_bound=(),
        max_bound=(),
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        slice_list=None,
        **parameters,
    ):
        """
        Initializes bound conditions.
        :param min_bound: Minimal bounds. (list of size number of nodes x number of states or controls ?)
        :param max_bound: Maximal bounds. (list of size number of nodes x number of states or controls ?)
        :param  interpolation: Interpolation type. (Instance of InterpolationType class)
        """
        if isinstance(min_bound, PathCondition):
            self.min = min_bound
        else:
            self.min = PathCondition(min_bound, interpolation=interpolation, slice_list=slice_list, **parameters)

        if isinstance(max_bound, PathCondition):
            self.max = max_bound
        else:
            self.max = PathCondition(max_bound, interpolation=interpolation, slice_list=slice_list, **parameters)

        self.type = interpolation
        self.t = None
        self.extra_params = self.min.extra_params
        self.nb_shooting = self.min.nb_shooting

    def check_and_adjust_dimensions(self, nb_elements, nb_shooting):
        """
        Detects if bounds are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same bounds than intermediates nodes.
        :param nb_elements: Length of each list. (integer)
        :param nb_shooting: Number of shooting nodes. (integer)
        """
        self.min.check_and_adjust_dimensions(nb_elements, nb_shooting, "Bound min")
        self.max.check_and_adjust_dimensions(nb_elements, nb_shooting, "Bound max")
        self.t = self.min.t
        self.nb_shooting = self.min.nb_shooting

    def concatenate(self, other):
        """
        Concatenates minimal and maximal bounds.
        :param other: Bounds to concatenate. (Instance of Bounds class)
        """
        self.min = PathCondition(np.concatenate((self.min, other.min)), interpolation=self.min.type)
        self.max = PathCondition(np.concatenate((self.max, other.max)), interpolation=self.max.type)

        self.type = self.min.type
        self.t = self.min.t
        self.extra_params = self.min.extra_params
        self.nb_shooting = self.min.nb_shooting

    def __getitem__(self, slice_list):
        if isinstance(slice_list, slice):
            t = self.min.t
            param = self.extra_params
            interpolation = self.type
            if interpolation == InterpolationType.CUSTOM:
                min_bound = self.min.custom_function
                max_bound = self.max.custom_function
            else:
                min_bound = np.array(self.min[slice_list.start : slice_list.stop : slice_list.step])
                max_bound = np.array(self.max[slice_list.start : slice_list.stop : slice_list.step])
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
                "Invalid input for slicing bounds. It should be like [a:b] or [a:b:c] with a the start index, "
                "b the stop index and c the step for slicing."
            )

    def __setitem__(self, slice, value):
        self.min[slice] = value
        self.max[slice] = value

    def __bool__(self):
        return len(self.min) > 0

    @property
    def shape(self):
        return self.min.shape


class QAndQDotBounds(Bounds):
    """Sets bounds on states which are [generalized coordinates positions, generalized coordinates velocities]"""

    def __init__(self, biorbd_model, all_generalized_mapping=None, q_mapping=None, q_dot_mapping=None):
        """
        Initializes and fills up the bounds on the states which are
        [generalized coordinates positions, generalized coordinates velocities].
        Takes in account the mapping of the states.
        all_generalized_mapping can not be used at the same time as q_mapping or q_dot_mapping.
        :param biorbd_model: Model biorbd. (biorbd.Model('path_to_model'))
        :param all_generalized_mapping: Mapping of the states. (Instance of Mapping class)
        :param q_mapping: Mapping of the generalized coordinates positions. (Instance of Mapping class)
        :param q_dot_mapping: Mapping of the generalized coordinates velocities. (Instance of Mapping class)
        """
        if all_generalized_mapping is not None:
            if q_mapping is not None or q_dot_mapping is not None:
                raise RuntimeError("all_generalized_mapping and a specified mapping cannot be used along side")
            q_mapping = all_generalized_mapping
            q_dot_mapping = all_generalized_mapping

        if not q_mapping:
            q_mapping = BidirectionalMapping(Mapping(range(biorbd_model.nbQ())), Mapping(range(biorbd_model.nbQ())))
        if not q_dot_mapping:
            q_dot_mapping = BidirectionalMapping(
                Mapping(range(biorbd_model.nbQdot())), Mapping(range(biorbd_model.nbQdot()))
            )

        QRanges = []
        QDotRanges = []
        for i in range(biorbd_model.nbSegment()):
            segment = biorbd_model.segment(i)
            QRanges += [q_range for q_range in segment.QRanges()]
            QDotRanges += [qdot_range for qdot_range in segment.QDotRanges()]

        x_min = [QRanges[i].min() for i in q_mapping.reduce.map_idx] + [
            QDotRanges[i].min() for i in q_dot_mapping.reduce.map_idx
        ]
        x_max = [QRanges[i].max() for i in q_mapping.reduce.map_idx] + [
            QDotRanges[i].max() for i in q_dot_mapping.reduce.map_idx
        ]

        super(QAndQDotBounds, self).__init__(min_bound=x_min, max_bound=x_max)


class InitialGuessOption(OptionGeneric):
    def __init__(self, initial_guess, interpolation=InterpolationType.CONSTANT, **params):
        if not isinstance(initial_guess, InitialGuess):
            initial_guess = InitialGuess(initial_guess, interpolation=interpolation, **params)

        super(InitialGuessOption, self).__init__(**params)
        self.initial_guess = initial_guess

    @property
    def shape(self):
        return self.initial_guess.shape


class InitialGuessList(UniquePerPhaseOptionList):
    def add(self, initial_guess, **extra_arguments):
        if isinstance(initial_guess, InitialGuessOption):
            self.copy(initial_guess)
        else:
            super(InitialGuessList, self)._add(
                initial_guess=initial_guess, option_type=InitialGuessOption, **extra_arguments
            )

    def __getitem__(self, item):
        return super(InitialGuessList, self).__getitem__(item).initial_guess

    def __next__(self):
        return super(InitialGuessList, self).__next__().initial_guess


class InitialGuess:
    def __init__(self, initial_guess=(), interpolation=InterpolationType.CONSTANT, **parameters):
        """
        Sets initial guesses.
        :param initial_guess: Initial guesses. (list)
        :param interpolation: Interpolation type. (Instance of InterpolationType class)
        """
        if isinstance(initial_guess, PathCondition):
            self.init = initial_guess
        else:
            self.init = PathCondition(initial_guess, interpolation=interpolation, **parameters)

    def check_and_adjust_dimensions(self, nb_elements, nb_shooting):
        """
        Detects if initial values are not given, in that case "0" is given for all degrees of freedom.
        Detects if initial values are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same  values than intermediates nodes.
        :param nb_elements: Number of elements to interpolate. (integer)
        :param nb_shooting: Number of shooting points. (integer)
        """
        self.init.check_and_adjust_dimensions(nb_elements, nb_shooting, "InitialGuess")

    def concatenate(self, other):
        """
        Concatenates initial guesses.
        :param other: Initial guesses to concatenate. (?)
        """
        self.init = PathCondition(
            np.concatenate((self.init, other.init)),
            interpolation=self.init.type,
        )

    def __bool__(self):
        return len(self.init) > 0

    @property
    def shape(self):
        return self.init.shape
