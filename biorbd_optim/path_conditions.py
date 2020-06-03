import numpy as np

from .mapping import BidirectionalMapping, Mapping
from .enums import InterpolationType


class PathCondition(np.ndarray):
    """Sets path constraints"""

    def __new__(cls, input_array, interpolation_type=InterpolationType.CONSTANT):
        """
        Interpolates path conditions with the chosen interpolation type.
        :param input_array: Array of path conditions (initial guess). (list)
        :param interpolation_type: Type of interpolation. (Instance of InterpolationType)
        (InterpolationType.CONSTANT, InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT, InterpolationType.LINEAR
        or InterpolationType.EACH_FRAME)
        :return: obj -> Objective. (?)
        """
        # Check and reinterpret input
        input_array = np.asarray(input_array, dtype=float)
        if len(input_array.shape) == 0:
            input_array = input_array[np.newaxis, np.newaxis]
        if interpolation_type == InterpolationType.CONSTANT:
            if len(input_array.shape) == 1:
                input_array = input_array[:, np.newaxis]
            if input_array.shape[1] != 1:
                raise RuntimeError("Value for InterpolationType.CONSTANT must have exactly one column")

        elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if len(input_array.shape) == 1:
                input_array = input_array[:, np.newaxis]
            if input_array.shape[1] != 1 and input_array.shape[1] != 3:
                raise RuntimeError("Value for InterpolationType.CONSTANT must have exactly one or three columns")
            if input_array.shape[1] == 1:
                input_array = np.repeat(input_array, 3, axis=1)
        elif interpolation_type == InterpolationType.LINEAR:
            if input_array.shape[1] != 2:
                raise RuntimeError("Value for InterpolationType.LINEAR must have exactly two columns")
        elif interpolation_type == InterpolationType.EACH_FRAME:
            if input_array.shape[1] < 2:
                raise RuntimeError("Value for InterpolationType.EACH_FRAME must exactly match the number of points")
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")
        obj = np.asarray(input_array).view(cls)

        # Additional information
        obj.nb_shooting = None
        obj.type = interpolation_type

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.nb_shooting = getattr(obj, "nb_shooting", None)
        self.type = getattr(obj, "type", None)

    def __reduce__(self):
        pickled_state = super(PathCondition, self).__reduce__()
        new_state = pickled_state[2] + (self.nb_shooting, self.type)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.nb_shooting = state[-2]
        self.type = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(PathCondition, self).__setstate__(state[0:-2])

    def check_and_adjust_dimensions(self, nb_elements, nb_shooting, condition_type):
        """
        Raises errors on the dimensions of the elements to be interpolated.
        :param nb_elements: Number of elements to be interpolated. (integer)
        :param nb_shooting: Number of shooting points. (integer)
        :param condition_type: Type of interpolation. (instance of InterpolationType class)
        """
        if (
            self.type == InterpolationType.CONSTANT
            or self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            or self.type == InterpolationType.LINEAR
        ):
            self.nb_shooting = nb_shooting
        elif self.type == InterpolationType.EACH_FRAME:
            self.nb_shooting = nb_shooting + 1
        else:
            if self.nb_shooting != nb_shooting:
                raise RuntimeError(
                    f"Invalid number of shooting ({self.nb_shooting}), the expected number is {nb_shooting}"
                )

        if self.shape[0] != nb_elements:
            raise RuntimeError(
                f"Invalid number of {condition_type} ({self.shape[0] }), the expected size is {nb_elements}"
            )

        if self.type == InterpolationType.CONSTANT:
            if self.shape[1] != 1:
                raise RuntimeError(
                    f"Invalid number of {condition_type} for InterpolationType.CONSTANT (ncols = {self.shape[1]}), "
                    f"the expected number of column is 1"
                )
        elif self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if self.shape[1] != 3:
                raise RuntimeError(
                    f"Invalid number of {condition_type} for InterpolationType.CONSTANT (ncols = {self.shape[1]}), "
                    f"the expected number of column is 3"
                )
        elif self.type == InterpolationType.LINEAR:
            if self.shape[1] != 2:
                raise RuntimeError(
                    f"Invalid number of {condition_type} for InterpolationType.LINEAR (ncols = {self.shape[1]}), "
                    f"the expected number of column is 2"
                )
        elif self.type == InterpolationType.EACH_FRAME:
            if self.shape[1] != self.nb_shooting:
                raise RuntimeError(
                    f"Invalid number of {condition_type} for InterpolationType.LINEAR (ncols = {self.shape[1]}), "
                    f"the expected number of column is {self.nb_shooting}"
                )
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")

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
            else:
                return self[:, 1]
        elif self.type == InterpolationType.LINEAR:
            return self[:, 0] + (self[:, 1] - self[:, 0]) * shooting_point / self.nb_shooting
        elif self.type == InterpolationType.EACH_FRAME:
            return self[:, shooting_point]
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")


class Bounds:
    """
    Organizes bounds of states("X"), controls("U") and "V".
    """

    def __init__(
        self,
        min_bound=(),
        max_bound=(),
        interpolation_type=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        **parameters,
    ):
        """
        Initializes bound conditions.
        :param min_bound: Minimal bounds. (list of size number of nodes x number of states or controls ?)
        :param max_bound: Maximal bounds. (list of size number of nodes x number of states or controls ?)
        :param  interpolation_type: Interpolation type. (Instance of InterpolationType class)
        """
        if isinstance(min_bound, PathCondition):
            self.min = min_bound
        else:
            self.min = PathCondition(min_bound, interpolation_type=interpolation_type, **parameters)

        if isinstance(max_bound, PathCondition):
            self.max = max_bound
        else:
            self.max = PathCondition(max_bound, interpolation_type=interpolation_type, **parameters)

    def check_and_adjust_dimensions(self, nb_elements, nb_shooting):
        """
        Detects if bounds are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same bounds than intermediates nodes.
        :param nb_elements: Length of each list. (integer)
        :param nb_shooting: Number of shooting nodes. (integer)
        """
        self.min.check_and_adjust_dimensions(nb_elements, nb_shooting, "Bound min")
        self.max.check_and_adjust_dimensions(nb_elements, nb_shooting, "Bound max")

    def concatenate(self, other):
        """
        Concatenates minimal and maximal bounds.
        :param other: Bounds to concatenate. (Instance of Bounds class)
        """
        self.min = PathCondition(np.concatenate((self.min, other.min)), interpolation_type=self.min.type)
        self.max = PathCondition(np.concatenate((self.max, other.max)), interpolation_type=self.max.type)


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


class InitialConditions:
    def __init__(self, initial_guess=(), interpolation_type=InterpolationType.CONSTANT, **parameters):
        """
        Sets initial guesses.
        :param initial_guess: Initial guesses. (list)
        :param interpolation_type: Interpolation type. (Instance of InterpolationType class)
        """
        if isinstance(initial_guess, PathCondition):
            self.init = initial_guess
        else:
            self.init = PathCondition(initial_guess, interpolation_type=interpolation_type, **parameters)

    def check_and_adjust_dimensions(self, nb_elements, nb_shooting):
        """
        Detects if initial values are not given, in that case "0" is given for all degrees of freedom.
        Detects if initial values are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same  values than intermediates nodes.
        :param nb_elements: Number of elements to interpolate. (integer)
        :param nb_shooting: Number of shooting points. (integer)
        """
        self.init.check_and_adjust_dimensions(nb_elements, nb_shooting, "InitialConditions")

    def concatenate(self, other):
        """
        Concatenates initial guesses.
        :param other: Initial guesses to concatenate. (?)
        """
        self.init = PathCondition(np.concatenate((self.init, other.init)), interpolation_type=self.init.type,)
