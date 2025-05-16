from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..misc.enums import SolverType

from ..misc.parameters_types import (
    Int,
    Float,
    AnyDict,
)


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
    def set_convergence_tolerance(self, tol: Float) -> None:
        """
        This function set the convergence tolerance

        Parameters
        ----------
        tol: float
            Global converge tolerance value
        """

    @abstractmethod
    def set_constraint_tolerance(self, tol: Float) -> None:
        """
        This function set the constraint tolerance.

        Parameters
        ----------
        tol: float
            Global constraint tolerance value
        """

    @abstractmethod
    def set_maximum_iterations(self, num: Int) -> None:
        """
        This function set the number of maximal iterations.

        Parameters
        ----------
        num: int
            Number of iterations
        """

    @abstractmethod
    def as_dict(self, solver) -> AnyDict:
        """
        This function return the dict options to launch the optimization

        Parameters
        ----------
        solver: SolverInterface
            Ipopt ou Acados interface
        """

    @abstractmethod
    def set_print_level(self, num: Int) -> None:
        """
        This function set Output verbosity level.

        Parameters
        ----------
        num: int
            print_level
        """
