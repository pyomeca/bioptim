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
        with_cholesky: bool  # TODO: This does not work for now
            If True, the Cholesky decomposition is used to reduce the number of optimization variables
        polynomial_degree: int
            The order of the polynomial to use during the collocation integration
        method: str
            The method to use during the collocation integration
        """

        def __init__(self, with_cholesky: bool = False, polynomial_degree: int = 4, method: str = "legendre"):
            self.with_cholesky = with_cholesky
            self.polynomial_degree = polynomial_degree
            self.method = method
