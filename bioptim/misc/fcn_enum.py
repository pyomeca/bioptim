from enum import Enum


class Fcn:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class FcnEnum(Fcn, Enum):  # an enum of Fcn precisely
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

    # TODO: maybe add get_type here too

    @staticmethod
    def get_type():
        raise NotImplementedError("Not implemented by enum.")

    @staticmethod
    def get_fcn_type():
        raise NotImplementedError("Not implemented by enum.")
