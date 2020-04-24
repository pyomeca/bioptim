import enum


class Axe(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


class OdeSolver(enum.Enum):
    """
    Four models to solve.
    RK is pretty much good balance.
    """

    COLLOCATION = 0
    RK = 1
    CVODES = 2
    NO_SOLVER = 3


class Instant(enum.Enum):
    """
    Five groups of nodes.
    START: first node only.
    MID: middle node only.
    INTERMEDIATES: all nodes except first and last.
    END: last node only.
    ALL: obvious.
    """

    START = "start"
    MID = "mid"
    INTERMEDIATES = "intermediates"
    END = "end"
    ALL = "all"
