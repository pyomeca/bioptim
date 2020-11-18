class OptionGeneric:
    def __init__(self, phase=-1, option_index=-1, name=None, type=None, target=None, **params):
        self.phase = phase
        self.option_index = option_index

        self.name = name
        self.type = type

        self.params = params
        self.target = target
        self.sliced_target = None  # This one is the sliced node from the target. This is what is actually tracked

    @property
    def shape(self):
        raise RuntimeError("OptionGeneric is Abstract")


class OptionList:
    def __init__(self):
        self.options = [
            [],
        ]

    def __len__(self):
        if self.options == [[]]:
            return 0
        else:
            return len(self.options)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1]

    def __getitem__(self, item):
        return self.options[item]

    def _add(self, option_type=OptionGeneric, phase=0, option_index=-1, **extra_arguments):
        option_index = self.__prepare_option_list(phase, option_index)
        self.options[phase][option_index] = option_type(phase=phase, option_index=option_index, **extra_arguments)

    def copy(self, option):
        self.__prepare_option_list(option.phase, option.option_index)
        self.options[option.phase][option.option_index] = option

    def __prepare_option_list(self, phase, option_index):
        for i in range(len(self.options), phase + 1):
            self.options.append([])
        if option_index == -1:
            for i, opt in enumerate(self.options[phase]):
                if not opt:
                    option_index = i
                    break
            else:
                option_index = len(self.options[phase])
        for i in range(len(self.options[phase]), option_index + 1):
            self.options[phase].append(None)
        return option_index

    def __bool__(self):
        return len(self) > 0

    def print(self):
        # TODO: Print all elements in the console
        pass


class UniquePerPhaseOptionList(OptionList):
    def _add(self, phase=-1, **extra_arguments):
        if phase == -1:
            phase = len(self)
        super(UniquePerPhaseOptionList, self)._add(phase=phase, option_index=0, **extra_arguments)

    def __getitem__(self, item):
        return super(UniquePerPhaseOptionList, self).__getitem__(item)[0]

    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1][0]
