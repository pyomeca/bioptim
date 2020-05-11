from .mapping import BidirectionalMapping, Mapping
from .enums import Initialization


class PathCondition:
    """
    Parent class of Bounds and InitialConditions.
    Uses only for methods overloading.
    """

    @staticmethod
    def regulation(var, nb_elements):
        pass

    @staticmethod
    def regulation_private(var, nb_elements, type):
        if len(var) != nb_elements:
            raise RuntimeError(f"Invalid number of {type} ({str(len(var))}), the expected size is {str(nb_elements)}")

    def expand(self, other):
        pass


class Bounds(PathCondition):
    """
    Organizes bounds of states("X"), controls("U") and "V".
    """

    def __init__(self, min_bound=(), max_bound=()):
        """
        There are 3 groups of nodes :
        1. First node
        2. Intermediates (= all nodes except first and last nodes)
        3. Last node
        Each group have 2 lists of bounds : one of minimum and one of maximum values.

        For X and Y bounds, lists have the number of degree of freedom elements.
        For V bounds, lists have number of degree of freedom elements * number of shooting points.
        """
        self.min = list(min_bound)
        self.first_node_min = list(min_bound)
        self.last_node_min = list(min_bound)

        self.max = list(max_bound)
        self.first_node_max = list(max_bound)
        self.last_node_max = list(max_bound)

    def regulation(self, nb_elements):
        """
        Detects if bounds are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same bounds than intermediates nodes.
        :param nb_elements: Length of each list.
        """
        self.regulation_private(self.min, nb_elements, "Bound min")
        self.regulation_private(self.max, nb_elements, "Bound max")

        if len(self.first_node_min) == 0:
            self.first_node_min = self.min
        if len(self.last_node_min) == 0:
            self.last_node_min = self.min

        if len(self.first_node_max) == 0:
            self.first_node_max = self.max
        if len(self.last_node_max) == 0:
            self.last_node_max = self.max

        self.regulation_private(self.first_node_min, nb_elements, "Bound first node min")
        self.regulation_private(self.first_node_max, nb_elements, "Bound first node max")
        self.regulation_private(self.last_node_min, nb_elements, "Bound last node min")
        self.regulation_private(self.last_node_max, nb_elements, "Bound last node max")

    def expand(self, other):
        self.min += other.min
        self.first_node_min += other.first_node_min
        self.last_node_min += other.last_node_min

        self.max += other.max
        self.first_node_max += other.first_node_max
        self.last_node_max += other.last_node_max


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


class InitialConditions(PathCondition):
    def __init__(self, initial_guess=(), initial_type = Initialization.CONSTANT):
        """
        Organises initial values (for solver)
        There are 3 groups of nodes :
        1. First node
        2. Intermediates (= all nodes without first and last nodes)
        3. Last node
        Each group have a list of initial values.
        """

        # TODO: Add the capability to initialize using initial and final frame that linearly complete between
        self.first_node_init = list(initial_guess)
        self.init = list(initial_guess)
        self.last_node_init = list(initial_guess)
        self.initial_type = initial_type

    def regulation(self, nb_elements):
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
        self.first_node_init += other.first_node_init
        self.last_node_init += other.last_node_init

    def get_init(self, initial, type = Initialization.CONSTANT, facteur = 1):
        if type == Initialization.CONSTANT:
            return initial
        elif type == Initialization.LINEAR:
            return [(initial.init[1 + 2*i] - initial.init[2*i]) * facteur +
                    initial.init[2*i] for i in range(len(initial.init)//2)
                    ]
