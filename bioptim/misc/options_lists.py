class OptionGeneric:
    def __init__(self, phase=-1, list_index=-1, name=None, type=None, **params):
        self.phase = phase
        self.list_index = list_index

        self.name = name
        self.type = type

        self.params = params

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

    def _add(self, option_type=OptionGeneric, phase=0, list_index=-1, **extra_arguments):
        list_index = self.__prepare_option_list(phase, list_index)
        self.options[phase][list_index] = option_type(phase=phase, list_index=list_index, **extra_arguments)

    def copy(self, option):
        self.__prepare_option_list(option.phase, option.list_index)
        self.options[option.phase][option.list_index] = option

    def __prepare_option_list(self, phase, list_index):
        for i in range(len(self.options), phase + 1):
            self.options.append([])
        if list_index == -1:
            for i, opt in enumerate(self.options[phase]):
                if not opt:
                    list_index = i
                    break
            else:
                list_index = len(self.options[phase])
        for i in range(len(self.options[phase]), list_index + 1):
            self.options[phase].append(None)
        return list_index

    def __bool__(self):
        return len(self) > 0

    def print(self):
        # TODO: Print all elements in the console
        pass


class UniquePerPhaseOptionList(OptionList):
    def _add(self, phase=-1, **extra_arguments):
        if phase == -1:
            phase = len(self)
        super(UniquePerPhaseOptionList, self)._add(phase=phase, list_index=0, **extra_arguments)

    def __getitem__(self, item):
        return super(UniquePerPhaseOptionList, self).__getitem__(item)[0]

    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1][0]
