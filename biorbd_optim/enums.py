from enum import Enum, IntEnum


class Axe(IntEnum):
    X = 0
    Y = 1
    Z = 2


class OdeSolver(Enum):
    """
    Integration methods.
    (RK is pretty much good balance)
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


class InterpolationType(Enum):
    """
    Type of interpolation.
    CONSTANT: Constant value.
    CONSTANT: Constant value except for the first and last nodes.
    LINEAR: Linear.
    EACH_FRAME: Values defined for each node (no interpolation).
    """

    CONSTANT = 0
    CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT = 1
    LINEAR = 2
    EACH_FRAME = 3
    SPLINE = 4
    CUSTOM = 5


class PlotType(Enum):
    """
    Type of plot.
    PLOT: plot nodes and linear interpolation between them.
    INTEGRATED: plot nodes and integrate between them.
    STEP: stair plot.
    """

    PLOT = 0
    INTEGRATED = 1
    STEP = 2
