from enum import Enum


class FcnEnum(Enum):  # an enum of Fcn precisely, helps with validation later
    def __call__(self, *args, **kwargs):
        """
        Call the member.
        """
        return self.value[0](*args, **kwargs)

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
