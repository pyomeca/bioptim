class OptionGeneric:
    def __init__(self, phase=-1, idx=-1, name=None, type=None, **params):
        self.phase = phase
        self.idx = idx

        self.name = name
        self.type = type

        self.params = params


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

    def _add(self, option_type=OptionGeneric, phase=0, idx=-1, **extra_arguments):
        idx = self.__prepare_option_list(phase, idx)
        self.options[phase][idx] = option_type(phase=phase, idx=idx, **extra_arguments)

    def copy(self, option):
        self.__prepare_option_list(option.phase, option.idx)
        self.options[option.phase][option.idx] = option

    def __prepare_option_list(self, phase, idx):
        for i in range(len(self.options), phase + 1):
            self.options.append([])
        if idx == -1:
            for i, opt in enumerate(self.options):
                if not opt:
                    idx = i
                    break
            else:
                idx = len(self.options[phase])
        for i in range(len(self.options[phase]), idx + 1):
            self.options[phase].append(None)
        return idx

    def print(self):
        # TODO: Print all elements in the console
        pass


class UniquePerPhaseOptionList(OptionList):
    def _add(self, phase=-1, **extra_arguments):
        if phase == -1:
            phase = len(self)
        super(UniquePerPhaseOptionList, self)._add(phase=phase, idx=0, **extra_arguments)

    def __getitem__(self, item):
        return super(UniquePerPhaseOptionList, self).__getitem__(item)[0]

    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1][0]
