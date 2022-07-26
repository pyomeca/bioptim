from typing import Callable
from enum import Enum


class Fcn:
    """
    A function presented to the user.
    """

    def __init__(self, func: Callable):
        """
        Initialize an Fcn.

        Parameters
        ----------
        func: Callable
            the function to be presented
        """
        self.func = func

    def __call__(self, *args, **kwargs):  # makes enum members callable
        """
        Call the function.

        Returns
        -------
        the function call
        """
        return self.func(*args, **kwargs)


class FcnEnum(Fcn, Enum):  # an enum of Fcn precisely, helps with validation later
    @staticmethod
    def get_type():
        """
        Returns the type of the member.
        """
        raise NotImplementedError("Not implemented by enum.")

    @staticmethod
    def get_fcn_types():
        """
        Returns the types of the enum.
        """
        raise NotImplementedError("Not implemented by enum.")
