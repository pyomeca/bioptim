class strstaticproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner) -> str:
        return self.func()


class intstaticproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner) -> int:
        return self.func()
