from typing import Callable

import numpy as np
from casadi import MX, SX, vertcat

from ..misc.mapping import BiMapping
from ..misc.options import OptionGeneric, OptionDict


class VariableScaling(OptionGeneric):
    def __init__(self, key: str, scaling: np.ndarray | list = None, **kwargs):
        """
        Parameters
        ----------
        scaling: np.ndarray | list
            The scaling of the variables
        """

        super(VariableScaling, self).__init__()

        if isinstance(scaling, list):
            scaling = np.array(scaling)
        elif not (isinstance(scaling, np.ndarray) or isinstance(scaling, VariableScaling)):
            raise RuntimeError(f"Scaling must be a list or a numpy array, not {type(scaling)}")

        self.scaling = scaling
        self.key = key

    def scaling(self):
        """
        Get the scaling

        Returns
        -------
        The scaling
        """

        return self.scaling["all"]

    def to_vector(self, n_repeat: int):
        """
        Repeat the scaling to match the variables vector format
        """

        n_elements = self.scaling.shape[0]

        scaling_vector = np.zeros((n_repeat * n_elements, 1))
        for i in range(n_repeat):
            scaling_vector[i * n_elements : (i + 1) * n_elements] = np.reshape(self.scaling, (n_elements, 1))

        return scaling_vector

    def to_array(self, n_elements: int, n_shooting: int):
        """
        Repeate the scaling to match the variables array format
        """

        scaling_array = np.zeros((n_elements, n_shooting))
        for i in range(n_shooting):
            scaling_array[:, i] = self.scaling

        return scaling_array


class VariableScalingList(OptionDict):
    """
    A list of VariableScaling if more than one is required

    Methods
    -------
    add(self, scaling: np.ndarray | list = None)
        Add a new variable scaling to the list
    __getitem__(self, item) -> Bounds
        Get the ith option of the list
    print(self)
        Print the VariableScalingList to the console
    """

    def __init__(self):
        super(VariableScalingList, self).__init__()

    def add(
        self,
        key: str = None,
        scaling: np.ndarray | list | VariableScaling = None,
        phase: int = -1,
    ):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        key: str
            The name of the variable to apply the scaling to
        scaling
            The value of the scaling for the variable
        phase
            The phase to apply the scaling to
        """

        if isinstance(scaling, VariableScaling):
            self.add(key=scaling.key, scaling=scaling.scaling, phase=phase)
        else:
            if scaling is None:
                raise ValueError("Scaling cannot be None")

            for i, elt in enumerate(scaling):
                if elt <= 0:
                    raise RuntimeError(
                        f"Scaling factors must be strictly greater than zero. {i}th element {elt} is not > 0."
                    )
            super(VariableScalingList, self)._add(key=key, phase=phase, scaling=scaling, option_type=VariableScaling)

    def __getitem__(self, item) -> VariableScaling | dict:
        """
        Get the ith option of the list

        Parameters
        ----------
        item: int
            The index of the option to get

        Returns
        -------
        The ith option of the list
        """

        return super(VariableScalingList, self).__getitem__(item)

    def print(self):
        """
        Print the VariableScalingList to the console
        """

        raise NotImplementedError("Printing of VariableScalingList is not ready yet")

    @staticmethod
    def scaling_fill_phases(ocp, x_scaling, xdot_scaling, u_scaling, x_init, u_init):
        """
        Fill the scaling with ones if not defined, in that case, the dimensions of the scaling are chosen to match the number of intial guesses.
        """
        x_scaling_out = VariableScalingList()
        xdot_scaling_out = VariableScalingList()
        u_scaling_out = VariableScalingList()

        for phase in range(ocp.n_phases):
            if "all" not in x_scaling.keys():
                nx = x_init[phase].shape[0]
                if len(x_scaling.keys()) > 0:
                    x_scaling_all = np.array([])
                    for key in x_scaling.keys():
                        if len(x_scaling) > 1:
                            x_scaling_all = np.concatenate((x_scaling_all, x_scaling[phase][key].scaling))
                        else:
                            x_scaling_all = np.concatenate((x_scaling_all, x_scaling[key].scaling))
                else:
                    x_scaling_all = np.ones((nx,))

            else:
                if len(x_scaling) > 1:
                    x_scaling_all = x_scaling[phase]["all"].scaling
                else:
                    x_scaling_all = x_scaling["all"].scaling

            if len(x_scaling.keys()) > 0:
                for key in x_scaling.keys():
                    if len(x_scaling) > 0:
                        x_scaling_out.add(key, scaling=x_scaling[phase][key].scaling, phase=phase)
                    else:
                        x_scaling_out.add(key, scaling=x_scaling[key].scaling, phase=0)

            if "all" not in xdot_scaling.keys():
                nx = x_init[phase].shape[0]
                if len(xdot_scaling.keys()) > 0:
                    xdot_scaling_all = np.array([])
                    for key in xdot_scaling.keys():
                        if len(xdot_scaling) > 1:
                            xdot_scaling_all = np.concatenate((xdot_scaling_all, xdot_scaling[phase][key].scaling))
                        else:
                            xdot_scaling_all = np.concatenate((xdot_scaling_all, xdot_scaling[key].scaling))
                else:
                    xdot_scaling_all = np.ones((nx,))
            else:
                if len(xdot_scaling) > 1:
                    xdot_scaling_all = xdot_scaling[phase]["all"].scaling
                else:
                    xdot_scaling_all = xdot_scaling["all"].scaling

            if len(xdot_scaling.keys()) > 0:
                for key in xdot_scaling.keys():
                    if len(xdot_scaling) > 0:
                        xdot_scaling_out.add(key, scaling=xdot_scaling[phase][key].scaling, phase=phase)
                    else:
                        xdot_scaling_out.add(key, scaling=xdot_scaling[key].scaling, phase=0)

            if "all" not in u_scaling.keys():
                nu = u_init[phase].shape[0]
                if len(u_scaling.keys()) > 0:
                    u_scaling_all = np.array([])
                    for key in u_scaling.keys():
                        if len(u_scaling) > 1:
                            u_scaling_all = np.concatenate((u_scaling_all, u_scaling[phase][key].scaling))
                        else:
                            u_scaling_all = np.concatenate((u_scaling_all, u_scaling[key].scaling))
                else:
                    u_scaling_all = np.ones((nu,))
            else:
                if len(u_scaling) > 1:
                    u_scaling_all = u_scaling[phase]["all"].scaling
                else:
                    u_scaling_all = u_scaling["all"].scaling

            if len(u_scaling.keys()) > 0:
                for key in u_scaling.keys():
                    if len(u_scaling) > 0:
                        u_scaling_out.add(key, scaling=u_scaling[phase][key].scaling, phase=phase)
                    else:
                        u_scaling_out.add(key, scaling=u_scaling[key].scaling, phase=0)

            x_scaling_out.add(key="all", scaling=x_scaling_all, phase=phase)
            xdot_scaling_out.add(key="all", scaling=xdot_scaling_all, phase=phase)
            u_scaling_out.add(key="all", scaling=u_scaling_all, phase=phase)

        return x_scaling_out, xdot_scaling_out, u_scaling_out


class OptimizationVariable:
    """
    An optimization variable and the indices to find this variable in its state or control vector

    Attributes
    ----------
    name: str
        The name of the variable
    mx: MX
        The MX variable associated with this variable
    index: range
        The indices to find this variable
    mapping: BiMapping
        The mapping of the MX
    parent_list: OptimizationVariableList
        The parent that added this entry

    Methods
    -------
    __len__(self)
        The len of the MX reduced
    cx_start(self)
        The CX of the variable (starting point)
    cx_end(self)
        The CX of the variable (ending point)
    """

    def __init__(
        self,
        name: str,
        mx: MX,
        cx_start: list | None,
        index: [range, list],
        mapping: BiMapping = None,
        parent_list=None,
    ):
        """
        Parameters
        ----------
        name: str
            The name of the variable
        mx: MX
            The MX variable associated with this variable
        index: [range, list]
            The indices to find this variable
        parent_list: OptimizationVariableList
            The list the OptimizationVariable is in
        """
        self.name: str = name
        self.mx: MX = mx
        self.original_cx: list = cx_start
        self.index: [range, list] = index
        self.mapping: BiMapping = mapping
        self.parent_list: OptimizationVariableList = parent_list
        self._current_cx_to_get = lambda: self.cx_start

    def __len__(self):
        """
        The len of the MX reduced

        Returns
        -------
        The number of element (correspond to the nrows of the MX)
        """
        return len(self.index)

    @property
    def cx(self):
        return self._current_cx_to_get()

    @property
    def cx_start(self):
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_start[self.index, :]  # TODO: ADD [self.index, :]

    @property
    def cx_end(self):
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_end[self.index, :]  # TODO: ADD [self.index, :]


class OptimizationVariableList:
    """
    A list of OptimizationVariable

    Attributes
    ----------
    elements: list
        Each of the variable separated
    _cx_start: MX | SX
        The symbolic MX or SX of the list (starting point)
    _cx_end: MX | SX
        The symbolic MX or SX of the list (ending point)
    mx_reduced: MX
        The reduced MX to the size of _cx
    cx_constructor: Callable
        The casadi symbolic type used for the optimization (MX or SX)

    Methods
    -------
    __getitem__(self, item: int | str)
        Get a specific variable in the list, whether by name or by index
    append(self, name: str, cx: list, mx: MX, bimapping: BiMapping)
        Add a new variable to the list
    cx(self)
        The cx of all elements together (starting point)
    cx_end(self)
        The cx of all elements together (ending point)
    mx(self)
        The MX of all variable concatenated together
    shape(self)
        The size of the CX
    __len__(self)
        The number of variables in the list
    """

    def __init__(self, cx_constructor):
        self.elements: list = []
        self.fake_elements: list = []
        self._cx_start: MX | SX | np.ndarray = np.array([])
        self._cx_end: MX | SX | np.ndarray = np.array([])
        self._cx_intermediates: list = []
        self.mx_reduced: MX = MX.sym("var", 0, 0)
        self.cx_constructor = cx_constructor
        self._current_cx_to_get = lambda: self.cx_start

    def __getitem__(self, item: int | str | list | range):
        """
        Get a specific variable in the list, whether by name or by index

        Parameters
        ----------
        item: int | str
            The index or name of the element to return

        Returns
        -------
        The specific variable in the list
        """

        if isinstance(item, int):
            return self.elements[item]
        elif isinstance(item, str):
            if item == "all":
                index = []
                for elt in self.elements:
                    index.extend(list(elt.index))
                return OptimizationVariable("all", self.mx, self.cx_start, index, None, self)

            for elt in self.elements:
                if item == elt.name:
                    return elt
            for elt in self.fake_elements:
                if item == elt.name:
                    return elt
            raise KeyError(f"{item} is not in the list")
        elif isinstance(item, (list, tuple)) or isinstance(item, range):
            mx = vertcat([elt.mx for elt in self.elements if elt.name in item])
            index = []
            for elt in self.elements:
                if elt.name in item:
                    index.extend(list(elt.index))
            return OptimizationVariable("some", mx, None, index)
        else:
            raise ValueError("OptimizationVariableList can be sliced with int, list, range or str only")

    def __setitem__(self, key, value: OptimizationVariable):
        self.elements.append(value)

    def set_current_cx_to_get_index(self, ocp, index: int):
        """
        Set the value of current_cx_to_get to corresponding index (cx_start for 0, cx_mid for 1, cx_end for 2) if
        ocp.assume_phase_dynamics. Otherwise, it is always cx_start

        Parameters
        ----------
        index
            The index to set the current cx to
        """

        if not ocp.assume_phase_dynamics:
            self._current_cx_to_get = lambda: self.cx_start
            return

        if index == 0:
            self._current_cx_to_get = lambda: self.cx_start

        elif index == 1:
            # TODO Change that to cx_mid Benjamin
            self._current_cx_to_get = lambda: self.cx_end

        elif index == 2:
            self._current_cx_to_get = lambda: self.cx_end

        else:
            raise ValueError(
                "Valid values for setting the cx is 0, 1 or 2. If you reach this error message, you probably tried to "
                "add more penalties than available. You can try to split them into more penalties or use "
                "assume_phase_dynamics=False.")

    def append_fake(self, name: str, index: MX | SX | list, mx: MX, bimapping: BiMapping):
        """
        Add a new variable to the fake list which add something without changing the size of the normal elements

        Parameters
        ----------
        name: str
            The name of the variable
        index: MX | SX
            The SX or MX variable associated with this variable. Is interpreted as index if is_fake is true
        mx: MX
            The MX variable associated with this variable
        bimapping: BiMapping
            The Mapping of the MX against CX
        """

        self.fake_elements.append(OptimizationVariable(name, mx, None, index, bimapping, self))

    def append(self, name: str, cx: list, mx: MX, bimapping: BiMapping):
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: list
            The list of SX or MX variable associated with this variable
        mx: MX
            The MX variable associated with this variable
        bimapping: BiMapping
            The Mapping of the MX against CX
        """

        index = range(self._cx_start.shape[0], self._cx_start.shape[0] + cx[0].shape[0])
        self._cx_start = vertcat(self._cx_start, cx[0])
        self._cx_end = vertcat(self._cx_end, cx[-1])

        for i, c in enumerate(cx[1:-1]):
            if i >= len(self._cx_intermediates):
                self._cx_intermediates.append(c)
            else:
                self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)

        self.mx_reduced = vertcat(self.mx_reduced, MX.sym("var", cx[0].shape[0]))
        self.elements.append(OptimizationVariable(name, mx, cx, index, bimapping, self))

    def append_from_scaled(
        self,
        name: str,
        cx: list,
        scaled_optimization_variable: "OptimizationVariableList",
    ):
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: list
            The list of SX or MX variable associated with this variable
        scaled_optimization_variable: OptimizationVariable
            The scaled optimization variable associated with this variable
        """

        self._cx_start = vertcat(self._cx_start, cx[0])
        self._cx_end = vertcat(self._cx_end, cx[-1])

        for i, c in enumerate(cx[1:-1]):
            if i >= len(self._cx_intermediates):
                self._cx_intermediates.append(c)
            else:
                self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)

        self.mx_reduced = scaled_optimization_variable.mx_reduced
        var = scaled_optimization_variable[name]
        self.elements.append(OptimizationVariable(name, var.mx, cx, var.index, var.mapping, self))

    @property
    def cx(self):
        """
        Return the current cx to get. This is to get the user an easy accessor to the proper cx to get. Indeed, in the
        backend, we may need to redirect the user toward cx_start, cx_mid or cx_end, but they may not know which to use.
        CX is expected to return the proper cx to the user.

        Returns
        -------
        Tne cx related to the current state
        """
        return self._current_cx_to_get()

    @property
    def cx_start(self):
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        return self.cx_constructor([] if self.shape == 0 else self._cx_start[:, 0])

    @property
    def cx_end(self):
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        return self.cx_constructor([] if self.shape == 0 else self._cx_end[:, 0])

    @property
    def cx_intermediates_list(self):
        """
        The cx of all elements together (starting point)
        """

        return self._cx_intermediates

    @property
    def mx(self):
        """
        Returns
        -------
        The MX of all variable concatenated together
        """

        return vertcat(*[elt.mx for elt in self.elements])

    def __contains__(self, item: str):
        """
        If the item of name item is in the list
        """

        for elt in self.elements:
            if item == elt.name:
                return True
        for elt in self.fake_elements:
            if item == elt.name:
                return True
        else:
            return False

    def keys(self):
        """
        All the names of the elements in the list
        """

        return [elt for elt in self]

    @property
    def shape(self):
        """
        The size of the CX
        """

        return self._cx_end.shape[0]

    def __len__(self):
        """
        The number of variables in the list
        """

        return len(self.elements)

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
        return self[self._iter_idx - 1].name


class OptimizationVariableContainer:
    def __init__(self, assume_phase_dynamics: bool):
        """
        This is merely a declaration function, it is mandatory to call initialize_from_shooting to get valid structures

        Parameters
        ----------
        assume_phase_dynamics
            If the dynamics is the same for all the phase (effectively always setting _node_index to 0 even though the
            user sets it to something else)
        """
        self.cx_constructor = None
        self._unscaled: list[OptimizationVariableList, ...] = []
        self._scaled: list[OptimizationVariableList, ...] = []
        self._node_index = 0  # TODO: [0] to [node_index]
        self.assume_phase_dynamic = assume_phase_dynamics

    @property
    def node_index(self):
        return self._node_index

    @node_index.setter
    def node_index(self, value):
        if not self.assume_phase_dynamic:
            self._node_index = value

    def initialize_from_shooting(self, n_shooting: int, cx: Callable):
        """
        Initialize the Container so the dimensions are up to number of shooting points of the program

        Parameters
        ----------
        n_shooting
            The number of shooting points
        cx
            The type (MX or SX) of the variable
        """

        if not isinstance(cx, Callable):
            raise ValueError("This entry should be either casadi.MX or casadi.SX")

        for node_index in range(n_shooting):
            self.cx_constructor = cx
            self._scaled.append(OptimizationVariableList(cx))
            self._unscaled.append(OptimizationVariableList(cx))

    def __getitem__(self, item: int | str):
        if isinstance(item, int):
            raise ValueError("To get a specific node, please set the node_index property then call the desired method.")
        elif isinstance(item, str):
            return self.unscaled[item]
        else:
            raise ValueError("item should be a node index or the name of the variable")

    @property
    def unscaled(self):
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._unscaled[self.node_index]

    @property
    def scaled(self):
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._scaled[self.node_index]

    def keys(self):
        return self.unscaled.keys()

    @property
    def shape(self):
        return self.unscaled.shape

    @property
    def mx(self):
        return self.unscaled.mx

    @property
    def mx_reduced(self):
        return self.unscaled.mx_reduced

    @property
    def cx_start(self):
        return self.unscaled.cx_start

    @property
    def cx_intermediates_list(self):
        return self.unscaled.cx_intermediates_list

    @property
    def cx_end(self):
        return self.unscaled.cx_end

    def append(
        self,
        name: str,
        cx: list,
        cx_scaled: list,
        mx: MX,
        mapping: BiMapping,
        node_index: int,
    ):
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: list
            The list of unscaled SX or MX variable associated with this variable
        cx_scaled: list
            The list of scaled SX or MX variable associated with this variable
        mx: MX
            The symbolic variable associated to this variable
        mapping
            The mapping to apply to the unscaled variable
        node_index
            The index of the node for the scaled variable
        """
        self._scaled[node_index].append(name, cx_scaled, mx, mapping)
        self._unscaled[node_index].append_from_scaled(name, cx, self._scaled[node_index])

    def __contains__(self, item: str):
        """
        If the item of name item is in the list
        """
        if item == "scaled" or item == "unscaled":
            return True

        return item in self.unscaled

    def __len__(self):
        """
        The number of variables in the list
        """

        return len(self.unscaled)

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
        return self.unscaled[self._iter_idx - 1].name
