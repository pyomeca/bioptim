from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..misc.enums import SolverType


@dataclass
class GenericSolver(ABC):
    """
    Abstract class for Solver Options

    Methods
    -------
    set_convergence_tolerance(self,tol: float):
        Set some convergence tolerance
    set_constraint_tolerance(self, tol: float):
        Set some constraint tolerance
    """

    type: SolverType = SolverType.NONE

    @abstractmethod
    def set_convergence_tolerance(self, tol: float):
        """
        This function set the convergence tolerance

        Parameters
        ----------
        tol: float
            Global converge tolerance value
        """

    @abstractmethod
    def set_constraint_tolerance(self, tol: float):
        """
        This function set the constraint tolerance.

        Parameters
        ----------
        tol: float
            Global constraint tolerance value
        """

    @abstractmethod
    def set_maximum_iterations(self, num: int):
        """
        This function set the number of maximal iterations.

        Parameters
        ----------
        num: int
            Number of iterations
        """

    @abstractmethod
    def as_dict(self, solver) -> dict:
        """
        This function return the dict options to launch the optimization

        Parameters
        ----------
        solver: SolverInterface
            Ipopt ou Acados interface
        """

    @abstractmethod
    def set_print_level(self, num: int):
        """
        This function set Output verbosity level.

        Parameters
        ----------
        num: int
            print_level
        """
