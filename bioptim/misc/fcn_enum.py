from abc import abstractmethod
from enum import Enum


class FcnEnum(Enum):
    def __call__(self, *args, **kwargs):
        """
        Call the member.
        """
        return self.value[0](*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_type():
        """
        Returns the type of the member.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_fcn_types():
        """
        Returns the types of the enum.
        """
        pass
