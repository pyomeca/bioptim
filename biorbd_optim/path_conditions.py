import numpy as np

from .mapping import BidirectionalMapping, Mapping
from .enums import InterpolationType


class PathCondition:
    def __init__(self, val, nb_nodes=0, interpolation_type=InterpolationType.CONSTANT):
        if nb_nodes == 0 and interpolation_type != InterpolationType.CONSTANT:
            raise RuntimeError("nb_nodes must be defined for interpolation_type != InterpolationType.CONSTANT")
        self.nb_nodes = nb_nodes
        self.type = interpolation_type

        val = np.array(val)
        if len(val) == 1:
            val = val[:, np.newaxis]
        self.value = np.ndarray((val.shape[0], nb_nodes))
        self.value[:, 0] = val[:, 0]
        if interpolation_type == InterpolationType.LINEAR:
            self.value[:, -1] = val[:, -1]

    def check_dimensions(self, nb_elements, condition_type):
        if self.value.shape[0] != nb_elements:
            raise RuntimeError(f"Invalid number of {condition_type} ({self.value.shape[1] }), the expected size is {str(nb_elements)}")

        if self.type == InterpolationType.CONSTANT and self.value.shape[1] != 1:
            raise RuntimeError(f"Invalid number of {condition_type} for InterpolationType.CONSTANT (ncols = {self.value.shape[1]}), the expected number of column is 1")
        elif self.type == InterpolationType.LINEAR and self.value.shape[1] != 2:
            raise RuntimeError(f"Invalid number of {condition_type} for InterpolationType.LINEAR (ncols = {self.value.shape[1]}), the expected number of column is 2")
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")

    def expand(self, other):
        self.value += other.value


class Bounds:
    """
    Organizes bounds of states("X"), controls("U") and "V".
    """

    def __init__(self, min_bound=(), max_bound=(), interpolation_type=InterpolationType.CONSTANT):
        if isinstance(min_bound, PathCondition):
            self.min = min_bound
        else:
            self.min = PathCondition(min_bound, interpolation_type)

        if isinstance(max_bound, PathCondition):
            self.max = max_bound
        else:
            self.max = PathCondition(max_bound, interpolation_type)

    def check_dimensions(self, nb_elements):
        """
        Detects if bounds are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same bounds than intermediates nodes.
        :param nb_elements: Length of each list.
        """
        self.min.check_dimensions(nb_elements, "Bound min")
        self.max.check_dimensions(nb_elements, "Bound max")

    def expand(self, other):
        self.min.expand(other.min)
        self.max.expand(other.max)


class QAndQDotBounds(Bounds):
    def __init__(self, biorbd_model, all_generalized_mapping=None, q_mapping=None, q_dot_mapping=None):
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
    def __init__(self, initial_guess=(), interpolation_type=InterpolationType.CONSTANT):
        if isinstance(initial_guess, PathCondition):
            self.init = initial_guess
        else:
            self.init = PathCondition(initial_guess, interpolation_type)

    def check_dimensions(self, nb_elements):
        """
        Detects if initial values are not given, in that case "0" is given for all degrees of freedom.
        Detects if initial values are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same  values than intermediates nodes.
        """
        if len(self.init) == 0:
            self.init = [0] * nb_elements
            self.initial_type = Initialization.CONSTANT

        if self.initial_type == Initialization.CONSTANT:
            if len(self.first_node_init) == 0:
                self.first_node_init = self.init
            if len(self.last_node_init) == 0:
                self.last_node_init = self.init
        elif self.initial_type == Initialization.LINEAR:
            if len(self.first_node_init) == 0:
                self.first_node_init = self.init[0::2]
            if len(self.last_node_init) == 0:
                self.last_node_init = self.init[1::2]

        if self.initial_type == Initialization.LINEAR:
            self.regulation_private(self.init, 2 * nb_elements, "Init")
        elif self.initial_type == Initialization.CONSTANT:
            self.regulation_private(self.init, nb_elements, "Init")

        # if self.initial_type == Initialization.LINEAR:
        #     self.regulation_private(self.first_node_init,
        #                             2*nb_elements, "First node init")
        #     self.regulation_private(self.last_node_init,
        #                             2*nb_elements, "Last node init")
        if self.initial_type == Initialization.CONSTANT:
            self.regulation_private(self.first_node_init,
                                    nb_elements, "First node init")
            self.regulation_private(self.last_node_init,
                                    nb_elements, "Last node init")

    def expand(self, other):
        self.init += other.init

    def get_init(self, initial, type = Initialization.CONSTANT, facteur = 1):
        if type == Initialization.CONSTANT:
            return initial
        elif type == Initialization.LINEAR:
            return [(initial.init[1 + 2*i] - initial.init[2*i]) * facteur +
                    initial.init[2*i] for i in range(len(initial.init)//2)
                    ]
