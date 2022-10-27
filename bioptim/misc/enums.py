from enum import Enum, IntEnum


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
    SQP_METHOD = "SQP"
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
    DEFAULT = "default"


class InterpolationType(Enum):
    """
    Selection of valid type of interpolation
    """

    CONSTANT = 0  # All values are set (time independent)
    CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT = 1  # All values are set, with the first and last defined to another one
    LINEAR = 2  # Linear interpolation between first and last
    EACH_FRAME = 3  # Each value is provided by the user
    ALL_POINTS = 4  # Values at all collocation points are provided by the user
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


class ControlType(Enum):
    """
    Selection of valid controls
    The goto value is CONSTANT
    """

    CONSTANT = 1  # Constant over the integration step (=1 column)
    LINEAR_CONTINUOUS = 2  # Linear interpolation between integration steps (=2 columns)
    NONE = 0  # Undeclared control type


class VariableType(Enum):
    """
    Selection of valid variable types
    """

    STATES = "states"
    CONTROLS = "controls"


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


class IntegralApproximation(Enum):
    """
    Selection of integral approximation
    """

    DEFAULT = "default"
    RECTANGLE = "rectangle"
    TRAPEZOIDAL = "trapezoidal"
    TRUE_TRAPEZOIDAL = "true_trapezoidal"


class SoftContactDynamics(Enum):
    ODE = "ode"
    CONSTRAINT = "constraint"


class RigidBodyDynamics(Enum):
    ODE = "ode"
    DAE_INVERSE_DYNAMICS = "dae_inverse_dynamics"
    DAE_FORWARD_DYNAMICS = "dae_forward_dynamics"
    DAE_INVERSE_DYNAMICS_JERK = "dae_inverse_dynamics_jerk"
    DAE_FORWARD_DYNAMICS_JERK = "dae_forward_dynamics_jerk"


class DefectType(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    NOT_APPLICABLE = "not_applicable"
