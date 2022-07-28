from enum import Enum


class FcnEnum(Enum):
    def __call__(self, *args, **kwargs):
        """
        Call the member.
        """
        if args and kwargs:
            return self.value[0](*args, **kwargs)
        elif args and not kwargs:
            return self.value[0](*args)
        elif not args and kwargs:
            return self.value[0](**kwargs)
        elif not args and not kwargs:
            return self.value[0]()

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
