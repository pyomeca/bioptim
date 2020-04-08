from .mapping import Mapping


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
            raise RuntimeError(f"Invalid number of {type}")

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

        self.regulation_private(
            self.first_node_min, nb_elements, "Bound first node min"
        )
        self.regulation_private(
            self.first_node_max, nb_elements, "Bound first node max"
        )
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
    def __init__(self,  biorbd_model, dof_mapping=None):
        if not dof_mapping:
            dof_mapping = Mapping(range(biorbd_model.nbQ()), range(biorbd_model.nbQ()))

        QRanges = []
        QDotRanges = []
        for i in range(biorbd_model.nbSegment()):
            segment = biorbd_model.segment(i)
            QRanges += [q_range for q_range in segment.QRanges()]
            QDotRanges += [qdot_range for qdot_range in segment.QDotRanges()]

        x_min = [QRanges[i].min() for i in dof_mapping.reduce_idx] + [
            QDotRanges[i].min() for i in dof_mapping.reduce_idx
        ]
        x_max = [QRanges[i].max() for i in dof_mapping.reduce_idx] + [
            QDotRanges[i].max() for i in dof_mapping.reduce_idx
        ]

        super(QAndQDotBounds, self).__init__(min_bound=x_min, max_bound=x_max)


class InitialConditions(PathCondition):
    def __init__(self, initial_guess=()):
        """
        Organises initial values (for solver)
        There are 3 groups of nodes :
        1. First node
        2. Intermediates (= all nodes without first and last nodes)
        3. Last node
        Each group have a list of initial values.
        """
        self.first_node_init = list(initial_guess)
        self.init = list(initial_guess)
        self.last_node_init = list(initial_guess)

    def regulation(self, nb_elements):
        """
        Detects if initial values are not given, in that case "0" is given for all degrees of freedom.
        Detects if initial values are not correct (wrong size of list: different than degrees of freedom).
        Detects if first or last nodes are not complete, in that case they have same  values than intermediates nodes.
        """
        if len(self.init) == 0:
            self.init = [0] * nb_elements
        self.regulation_private(self.init, nb_elements, "Init")

        if len(self.first_node_init) == 0:
            self.first_node_init = self.init
        if len(self.last_node_init) == 0:
            self.last_node_init = self.init

        self.regulation_private(self.first_node_init, nb_elements, "First node init")
        self.regulation_private(self.last_node_init, nb_elements, "Last node init")

    def expand(self, other):
        self.init += other.init
        self.first_node_init += other.first_node_init
        self.last_node_init += other.last_node_init
