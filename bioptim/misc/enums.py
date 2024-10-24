from enum import Enum, IntEnum, auto
import platform


class PhaseDynamics(Enum):
    SHARED_DURING_THE_PHASE = "shared_during_the_phase"
    ONE_PER_NODE = "one_per_node"


class Axis(IntEnum):
    """
    Selection of valid axis (X, Y or Z)
    """

    X = 0
    Y = 1
    Z = 2


class SolverType(Enum):
    """
    Selection of valid nonlinear solvers
    The goto value IPOPT
    """

    IPOPT = "Ipopt"
    ACADOS = "ACADOS"
    SQP = "SqpMethod"
    NONE = None


class Node(Enum):
    """
    Selection of valid node
    """

    START = "start"  # The first node of the phase
    MID = "mid"  # The middle node of the phase
    INTERMEDIATES = "intermediates"  # All the nodes but the first and last
    PENULTIMATE = "penultimate"  # The second to last node of the phase
    END = "end"  # The last node of the phase
    ALL = "all"  # All the nodes
    ALL_SHOOTING = "all_shooting"  # All the shooting nodes
    TRANSITION = "transition"  # The last node of a phase and the first node of the next phase
    MULTINODES = "multinodes"  # Constraint an arbitrary number of node to be equal
    DEFAULT = "default"


class InterpolationType(Enum):
    """
    Selection of valid type of interpolation
    """

    CONSTANT = 0  # All values are set (time independent)
    CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT = 1  # All values are set, with the first and last defined to another one
    LINEAR = 2  # Linear interpolation between first and last
    EACH_FRAME = 3  # Each value is provided by the user
    ALL_POINTS = 4  # If in direct collocation, it is at all collocation points, otherwise it acts as EACH_FRAME
    SPLINE = 5  # Cubic spline interpolation
    CUSTOM = 6  # Interpolation via a used-defined custom function


class Shooting(Enum):
    """
    The type of integration
    MULTIPLE resets the state at each node
    SINGLE resets the state at each phase
    SINGLE_CONTINUOUS never resets the state
    """

    MULTIPLE = "Multiple"
    SINGLE = "Single"
    SINGLE_DISCONTINUOUS_PHASE = "Single discontinuous phase"


class CostType(Enum):
    """
    The type of cost
    """

    OBJECTIVES = "Objectives"
    CONSTRAINTS = "Constraints"
    ALL = "All"


class PlotType(Enum):
    """
    Selection of valid plots
    """

    PLOT = 0  # Linking between points
    INTEGRATED = 1  # Linking between interpolated points
    STEP = 2  # Step plot
    POINT = 3  # Point plot


class OnlineOptim(Enum):
    """
    The type of callback

    Attributes
    ----------
    NONE: No online plotting
    DEFAULT: Default online plotting (MULTIPROCESS on Linux, MULTIPROCESS_SERVER on Windows and NONE on MacOS)
    MULTIPROCESS: Multiprocess online plotting
    SERVER: Server online plotting
    MULTIPROCESS_SERVER: Multiprocess server online plotting
    """

    DEFAULT = auto()
    MULTIPROCESS = auto()
    SERVER = auto()
    MULTIPROCESS_SERVER = auto()

    def get_default(self):
        if self != OnlineOptim.DEFAULT:
            return self

        if platform.system() == "Linux":
            return OnlineOptim.MULTIPROCESS
        elif platform.system() == "Windows":
            return OnlineOptim.MULTIPROCESS_SERVER
        else:
            return None


class ControlType(Enum):
    """
    Selection of valid controls
    The goto value is CONSTANT
    """

    NONE = 0  # Undeclared control type
    CONSTANT = 1  # Constant over the integration step, the last node is a NaN (=1 column)
    LINEAR_CONTINUOUS = 2  # Linear interpolation between integration steps (=2 columns)
    CONSTANT_WITH_LAST_NODE = 3  # Constant over the integration step, the last node exists (=1 columns)


class VariableType(Enum):
    """
    Selection of valid variable types
    """

    STATES = "states"
    CONTROLS = "controls"
    STATES_DOT = "states_dot"
    ALGEBRAIC_STATES = "algebraic_states"


class SolutionIntegrator(Enum):
    """
    Selection of integrator to use integrate function
    """

    OCP = "OCP"
    SCIPY_RK23 = "RK23"
    SCIPY_RK45 = "RK45"
    SCIPY_DOP853 = "DOP853"
    SCIPY_BDF = "BDF"
    SCIPY_LSODA = "LSODA"


class PenaltyType(Enum):  # it's more of a "Category" than "Type"
    """
    Selection of penalty types
    """

    USER = "user"
    INTERNAL = "internal"


class ConstraintType(Enum):
    """
    Selection of constraint types
    """

    IMPLICIT = "implicit"


class QuadratureRule(Enum):
    """
    Selection of quadrature rule to approximate integrals.
    """

    DEFAULT = "default"
    RECTANGLE_LEFT = "rectangle_left"
    RECTANGLE_RIGHT = "rectangle_right"
    MIDPOINT = "midpoint"
    APPROXIMATE_TRAPEZOIDAL = "approximate_trapezoidal"
    TRAPEZOIDAL = "trapezoidal"


class SoftContactDynamics(Enum):
    ODE = "ode"
    CONSTRAINT = "constraint"


class DefectType(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    NOT_APPLICABLE = "not_applicable"


class MagnitudeType(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class MultiCyclicCycleSolutions(Enum):
    """
    Selection of extra solution for multi cyclic receding horizon optimization
    """

    NONE = "none"
    FIRST_CYCLES = "first_cycles"
    ALL_CYCLES = "all_cycles"


class ReferenceFrame(Enum):
    """
    Selection of reference frame
    """

    GLOBAL = "global"
    LOCAL = "local"


class ExternalForcesType(Enum):
    """
    Selection of external forces type
    """

    LINEAR_FORCE = "linear_force"
    TORQUE = "torque"
