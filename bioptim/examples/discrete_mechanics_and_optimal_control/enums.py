from enum import Enum


class VariationalIntegratorType(Enum):
    """
    The different variational integrator types
    """

    DISCRETE_EULER_LAGRANGE = "discrete_euler_lagrange"
    CONSTRAINED_DISCRETE_EULER_LAGRANGE = "constrained_discrete_euler_lagrange"
    FORCED_DISCRETE_EULER_LAGRANGE = "forced_discrete_euler_lagrange"
    FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE = "forced_constrained_discrete_euler_lagrange"


class InitialGuessApproximation(Enum):
    """
    The different initial guess approximations available for the Newton's method
    """

    LAGRANGIAN = "lagrangian"
    CURRENT = "current"
    EXPLICIT_EULER = "explicit_euler"
    SEMI_IMPLICIT_EULER = "semi_implicit_euler"
    CUSTOM = "custom"
