from typing import Any, Callable


class OptionGeneric:
    """
    A placeholder for an option (abstract)

    Attributes
    ----------
    phase: int
        The phase the option is associated with
    list_index: int
        The index of the option if is it in a list
    name: str
        The name of the option
    type: Any
        The type of the option
    params: dict
        Any extra parameters that did not fall in any category

    Methods
    -------
    shape(self)
        Get the size of the OptionGeneric (abstract)
    """

    def __init__(self, phase: int = -1, list_index: int = -1, name: str = None, type: Any = None, **params):
        """
        Parameters
        ----------
        phase: int
            The phase the option is associated with
        list_index: int
            The index of the option if is it in a list
        name: str
            The name of the option
        type: Any
            The type of the option
        params: dict
            Any extra parameters that did not fall in any category
        """

        self.phase = phase
        self.list_index = list_index

        self.name = name
        self.type = type

        self.params = params

    @property
    def shape(self):
        """
        Get the size of the OptionGeneric (abstract)

        Returns
        -------
        The size of the OptionGeneric
        """

        raise RuntimeError("OptionGeneric is Abstract")


class OptionList:
    """
    A list of OptionGeneric if more than one is required

    Attributes
    options: list[list[]]
        A list [phase] of list [OptionGeneric]

    Methods
    -------
    __len__(self)
        Allow for len(option) to be called
    __iter__(self)
        Allow for the list to be used in a for loop
    __next__(self):
        Get the next phase of the option list
    __getitem__(self, i) -> list
        Get the ith phase list of the option list
    _add(self, option_type: Callable = OptionGeneric, phase: int = 0, list_index: int = -1, **extra_arguments)
        Add a new option to the list
    copy(self, option: OptionGeneric)
        Deepcopy of an option in the list
    __prepare_option_list(self, phase: int, list_index: int) -> int
        Reshape the option according to the requested phase and index
    __bool__(self) -> bool
        Check if the list is empty
    print(self)
        Print the option to the console
    """

    def __init__(self):
        """"""
        self.options = [
            [],
        ]

    def __len__(self):
        """
        Allow for len(option) to be called

        Returns
        -------
        The len of the list of OptionGeneric
        """

        if self.options == [[]]:  # Special case which would return 1 even though it is empty
            return 0
        else:
            return len(self.options)

    def __iter__(self):
        """
        Allow for the list to be used in a for loop

        Returns
        -------
        A reference to self
        """

        self._iter_idx = 0
        return self

    def __next__(self):
        """
        Get the next phase of the option list

        Returns
        -------
        The next phase of the option list
        """

        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1]

    def __getitem__(self, i) -> list:
        """
        Get the ith phase list of the option list

        Parameters
        ----------
        i: int
            The index of the option to get

        Returns
        -------
        The ith phase list of the option list
        """

        return self.options[i]

    def _add(self, option_type: Callable = OptionGeneric, phase: int = 0, list_index: int = -1, **extra_arguments: Any):
        """
        Add a new option to the list

        Parameters
        ----------
        option_type: Callable
            The type of option
        phase: int
            The phase the option is associated with
        list_index: int
            The index of the option in the list. If list_index < 0, the option is appended at the end. If the list_index
            refers to a previously declare, the latter override the former
        extra_arguments: dict
            Any extra parameters that did not fall in any category
        """

        list_index = self.__prepare_option_list(phase, list_index)
        self.options[phase][list_index] = option_type(phase=phase, list_index=list_index, **extra_arguments)

    def copy(self, option: OptionGeneric):
        """
        Deepcopy of an option in the list

        Parameters
        ----------
        option: OptionGeneric
            The option to copy
        """

        self.__prepare_option_list(option.phase, option.list_index)
        self.options[option.phase][option.list_index] = option

    def __prepare_option_list(self, phase: int, list_index: int) -> int:
        """
        Reshape the option according to the requested phase and index

        Parameters
        ----------
        phase: int
            The phase index to add the option to
        list_index: int
            The index of the option in a specific phase. If the value is -1, the option is appended. If the list_index
            is out of range, the option list is filled with None up to the required index

        Returns
        -------
        The list_index that may have been modify (if -1)
        """

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

    def __bool__(self) -> bool:
        """
        Check if the list is empty

        Returns
        -------
        If the list is empty
        """

        return len(self) > 0

    def print(self):
        """
        Print the option to the console
        """
        # TODO: Print all elements in the console
        raise NotImplementedError("Printing of options is not ready yet")


class UniquePerPhaseOptionList(OptionList):
    """
    OptionList that does not allow for more than one element per phase

    Methods
    -------
    _add(self, phase: int = -1, **extra_arguments)
        Add a new option to the list
    copy(self, option: OptionGeneric)
        Deepcopy of an option in the list
    __getitem__(self, i_phase) -> Any
        Get the ith option of the list
    __next__(self) -> int
        Get the next option of the list
    print(self):
        Print the UniquePerPhaseOptionList to the console
    """

    def _add(self, phase: int = -1, **extra_arguments: Any):
        """
        Add a new option to the list

        Parameters
        ----------
        phase: int
            The phase the option is associated with
        extra_arguments: dict
            Everything but phase which is passed to OptionList, with list_index set to 0
        """

        if phase == -1:
            phase = len(self)
        super(UniquePerPhaseOptionList, self)._add(phase=phase, list_index=0, **extra_arguments)

    def copy(self, option: OptionGeneric):
        """
        Deepcopy of an option in the list

        Parameters
        ----------
        """

        if option.phase == -1:
            option.phase = len(self)
        super(UniquePerPhaseOptionList, self).copy(option)

    def __getitem__(self, i_phase) -> Any:
        """
        Get the ith option of the list

        Parameters
        ----------
        i_phase: int
            The index of the option to get

        Returns
        -------
        The ith option of the list
        """

        return super(UniquePerPhaseOptionList, self).__getitem__(i_phase)[0]

    def __next__(self) -> int:
        """
        Get the next option of the list

        Returns
        -------
        The next option of the list
        """
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1][0]

    def print(self):
        """
        Print the UniquePerPhaseOptionList to the console
        """
        raise NotImplementedError("Printing of UniquePerPhaseOptionList is not ready yet")


class UniquePerProblemOptionList(OptionList):
    """
    OptionList that cannot change throughout phases (e.g., parameters)

    Methods
    -------
    _add(self, phase: int = -1, **extra_arguments)
        Add a new option to the list
    copy(self, option: OptionGeneric)
        Deepcopy of an option in the list
    __getitem__(self, i_phase) -> Any
        Get the ith option of the list
    __next__(self) -> int
        Get the next option of the list
    print(self):
        Print the UniquePerPhaseOptionList to the console
    """

    def _add(self, list_index: int = -1, **extra_arguments: Any):
        """
        Add a new option to the list

        Parameters
        ----------
        phase: int
            The phase the option is associated with
        extra_arguments: dict
            Everything but phase which is passed to OptionList, with list_index set to 0
        """

        super(UniquePerProblemOptionList, self)._add(phase=0, list_index=list_index, **extra_arguments)

    def copy(self, option: OptionGeneric):
        """
        Deepcopy of an option in the list

        Parameters
        ----------
        """

        if option.phase == -1:
            option.phase = len(self)
        super(UniquePerProblemOptionList, self).copy(option)

    def __getitem__(self, index) -> Any:
        """
        Get the ith option of the list

        Parameters
        ----------
        index: int
            The index of the option to get

        Returns
        -------
        The ith option of the list
        """

        return super(UniquePerProblemOptionList, self).__getitem__(0)[index]

    def __next__(self) -> int:
        """
        Get the next option of the list

        Returns
        -------
        The next option of the list
        """
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[0][self._iter_idx - 1]

    def __len__(self):
        return len(self.options[0])

    def print(self):
        """
        Print the UniquePerProblemOptionList to the console
        """
        raise NotImplementedError("Printing of UniquePerProblemOptionList is not ready yet")
