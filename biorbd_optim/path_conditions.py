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


class Bounds(PathCondition):
    """
    Organizes bounds of states("X"), controls("U") and "V".
    """

    def __init__(self):
        """
        There are 3 groups of nodes :
        1. First node
        2. Intermediates (= all nodes except first and last nodes)
        3. Last node
        Each group have 2 lists of bounds : one of minimum and one of maximum values.

        For X and Y bounds, lists have the number of degree of freedom elements.
        For V bounds, lists have number of degree of freedom elements * number of shooting points.
        """
        self.min = []
        self.first_node_min = []
        self.last_node_min = []

        self.max = []
        self.first_node_max = []
        self.last_node_max = []

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


class InitialConditions(PathCondition):
    def __init__(self):
        """
        Organises initial values (for solver)
        There are 3 groups of nodes :
        1. First node
        2. Intermediates (= all nodes without first and last nodes)
        3. Last node
        Each group have a list of initial values.
        """
        self.first_node_init = []
        self.init = []
        self.last_node_init = []

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
