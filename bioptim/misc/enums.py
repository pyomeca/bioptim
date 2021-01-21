from enum import Enum, IntEnum


class Axis(IntEnum):
    """
    Selection of valid axis (X, Y or Z)
    """

    X = 0
    Y = 1
    Z = 2


class Solver(Enum):
    """
    Selection of valid nonlinear solvers
    The goto value IPOPT
    """

    IPOPT = "Ipopt"
    ACADOS = "ACADOS"
    NONE = None


class OdeSolver(Enum):
    """
    Selection of valid integrator
    The goto value is RK4
    """

    RK4 = 0  # Runge-Kutta of the 4th order
    RK8 = 1  # Runge-Kutta of the 8th order
    IRK = 2  # Implicit runge-Kutta
    CVODES = 3
    NO_SOLVER = 4


class Node(Enum):
    """
    Selection of valid node
    """

    START = "start"  # The first node of the phase
    MID = "mid"  # The middle node of the phase
    INTERMEDIATES = "intermediates"  # All the nodes but the first and last
    END = "end"  # The last node of the phase
    ALL = "all"  # All the nodes
    DEFAULT = "default"


class InterpolationType(Enum):
    """
    Selection of valid type of interpolation
    """

    CONSTANT = 0  # All values are set (time independent)
    CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT = 1  # All values are set, with the first and last defined to another one
    LINEAR = 2  # Linear interpolation between first and last
    EACH_FRAME = 3  # Each values are provided by the user
    SPLINE = 4  # Cubic spline interpolation
    CUSTOM = 5  # Interpolation via a used-defined custom function


class PlotType(Enum):
    """
    Selection of valid plots
    """

    PLOT = 0  # Linking between points
    INTEGRATED = 1  # Linking between interpolated points
    STEP = 2  # Step plot


class ControlType(Enum):
    """
    Selection of valid controls
    The goto value is CONSTANT
    """

    CONSTANT = 1  # Constant over the integration step (=1 column)
    LINEAR_CONTINUOUS = 2  # Linear interpolation between integration steps (=2 columns)
    NONE = 0  # Undeclared control type
