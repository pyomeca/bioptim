from enum import Enum, IntEnum


class Axe(IntEnum):
    X = 0
    Y = 1
    Z = 2


class OdeSolver(Enum):
    """
    Four models to solve.
    RK is pretty much good balance.
    """

    COLLOCATION = 0
    RK = 1
    CVODES = 2
    NO_SOLVER = 3


class Instant(Enum):
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

class Initialization(Enum):
    """
    Two kind of initialization of your problem value (State X or Control U)
    CONSTANT: constant initialization. Give the list of your unic value.
    LINEAR: linear initialization. Give the list of the first and last value.
    By default, the Initialization will be CONSTANT.
    """

    CONSTANT = 0
    LINEAR = 1
