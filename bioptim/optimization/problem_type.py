import numpy as np


class SocpType:
    """
    Selection of the type of optimization problem to be solved.
    """

    def __init__(self):
        pass

    class TRAPEZOIDAL_EXPLICIT:
        """
        The class used to declare a stochastic optimal control problem with trapezoidal integration and the variables
        defined explicitly.
        Attributes
        ----------
        with_cholesky: bool
            If True, the Cholesky decomposition is used to reduce the number of optimization variables
        """

        def __init__(self, with_cholesky: bool = False):
            self.with_cholesky = with_cholesky

    class TRAPEZOIDAL_IMPLICIT:
        """
        The class used to declare a stochastic optimal control problem with trapezoidal integration and the variables
        defined implicitly.
        Attributes
        ----------
        with_cholesky: bool
            If True, the Cholesky decomposition is used to reduce the number of optimization variables
        """

        def __init__(self, with_cholesky: bool = False):
            self.with_cholesky = with_cholesky

    class COLLOCATION:
        """
        The class used to declare a stochastic optimal control problem taking advantage of the collocation integration.
        Attributes
        ----------
        polynomial_degree: int
            The order of the polynomial to use during the collocation integration
        method: str
            The method to use during the collocation integration
        """

        def __init__(
            self,
            polynomial_degree: int = 4,
            method: str = "legendre",
            auto_initialization: bool = False,
            initial_cov: np.ndarray = None,
        ):
            """
            Parameters
            ----------
            polynomial_degree: int
                The order of the polynomial to use during the collocation integration
            method: str
                The method to use during the collocation integration (Legendre or Radau)
            auto_initialization: bool
                If True, the initial guess of the states and controls are automatically initialized
            initial_cov: np.ndarray
                The initial covariance matrix of the states and controls
            """
            self.polynomial_degree = polynomial_degree
            self.method = method
            self.auto_initialization = auto_initialization
            if auto_initialization and initial_cov is None:
                raise RuntimeError(
                    "To initialize automatically the values of the stochastic variables, you need to provide the value of the covariance matrix at the first node (initial_cov)."
                )
            self.initial_cov = initial_cov
