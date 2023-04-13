from typing import Callable

import numpy as np
from casadi import MX, SX, vertcat

from ..misc.mapping import BiMapping
from ..misc.options import OptionGeneric, OptionDict
from ..misc.enums import CXStep


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
        min_bound: PathCondition | np.ndarray | list | tuple
            The minimum path condition. If min_bound if defined, then max_bound must be so and bound should be None
        max_bound: PathCondition | np.ndarray | list | tuple
            The maximum path condition. If max_bound if defined, then min_bound must be so and bound should be None
        bounds: Bounds
            Copy a Bounds. If bounds is defined, min_bound and max_bound should be None
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if isinstance(scaling, VariableScaling):
            self.add(key=scaling.key, scaling=scaling.scaling, phase=phase)
        else:
            for i, elt in enumerate(scaling):
                if elt <= 0:
                    raise RuntimeError(
                        f"Scaling factors must be strictly greater than zero. {i}th element {elt} is not > 0."
                    )
            super(VariableScalingList, self)._add(key=key, phase=phase, scaling=scaling, option_type=VariableScaling)

    def __getitem__(self, item) -> VariableScaling:
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

    def __len__(self):
        """
        The len of the MX reduced

        Returns
        -------
        The number of element (correspond to the nrows of the MX)
        """
        return len(self.index)

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
        return self.parent_list.cx_start

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
        return self.parent_list.cx_end


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

    def __init__(self):

        self.elements: list = []
        self.fake_elements: list = []
        self._cx_start: MX | SX | np.ndarray = np.array([])
        self._cx_end: MX | SX | np.ndarray = np.array([])
        self._cx_intermediates: list = []
        self.mx_reduced: MX = MX.sym("var", 0, 0)
        self.cx_constructor = None

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
                return OptimizationVariable("all", self.mx, self.cx_start, index, None)

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

    def get_cx(self, key, cx_type):
        if key == "all":
            return self.cx_start if cx_type == CXStep.CX_START else self.cx_end
        else:
            return self[key].cx_start if cx_type == CXStep.CX_START else self[key].cx_end

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
    def cx_start(self):
        """
        The cx of all elements together
        """

        return self.cx_constructor([]) if self.shape == 0 else self._cx_start[:, 0]

    @property
    def cx_end(self):
        """
        The cx of all elements together
        """

        return self.cx_constructor([]) if self.shape == 0 else self._cx_end[:, 0]

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
    def __init__(self, variable_scaled=None, variables_unscaled=None, casadi_symbolic_callable: Callable = None):
        self.cx_constructor = casadi_symbolic_callable
        if variable_scaled is None:
            variable_scaled = OptimizationVariableList()
            variable_scaled.cx_constructor = self.cx_constructor
        if variables_unscaled is None:
            variables_unscaled = OptimizationVariableList()
            variable_scaled.cx_constructor = self.cx_constructor
        self.optimization_variable = {"scaled": variable_scaled, "unscaled": variables_unscaled}

    def __getitem__(self, item: int | str | list | range):
        if isinstance(item, str) and (item == "unscaled" or item == "scaled"):
            return self.optimization_variable[item]
        else:
            return self.optimization_variable["unscaled"][item]

    def __setitem__(self, item: int | str | list | range, value: OptimizationVariableList | np.ndarray):
        if isinstance(item, str) and (item == "unscaled" or item == "scaled"):
            self.optimization_variable[item] = value
        else:
            self.optimization_variable["unscaled"][item] = value

    def _set_cx_constructor(self, cx_constructor: Callable = None):
        "Sets the constructors with MX or SX, to be called later, as MX.zeros(...)"
        if cx_constructor is None and not isinstance(cx_constructor, Callable):
            raise ValueError("This entry should be either casadi.MX or casadi.SX")
        self.cx_constructor = cx_constructor
        self.optimization_variable["scaled"].cx_constructor = self.cx_constructor
        self.optimization_variable["unscaled"].cx_constructor = self.cx_constructor

    @staticmethod
    def _set_states_and_controls(n_shooting: int, cx: MX) -> list:
        out = [OptimizationVariableContainer() for _ in range(n_shooting)]
        for node_index in range(n_shooting):
            out[node_index]._set_cx_constructor(cx)
        return out

    def keys(self):
        return self.optimization_variable["unscaled"].keys()

    @property
    def shape(self):
        return self.optimization_variable["unscaled"].shape

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

        return self.optimization_variable["unscaled"].append(name, cx, mx, bimapping)

    def append_from_scaled(
        self,
        name: str,
        cx: list,
        scaled_optimization_variable: OptimizationVariableList,
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

        return self.optimization_variable["unscaled"].append_from_scaled(name, cx, scaled_optimization_variable)

    def __contains__(self, item: str):
        """
        If the item of name item is in the list
        """
        if item == "scaled" or item == "unscaled":
            return True

        return item in self.optimization_variable["unscaled"]

    def get_cx(self, key, cx_type):
        return self.optimization_variable["unscaled"].get_cx(key, cx_type)

    @property
    def cx_start(self):
        """
        The cx of all elements together (starting point)
        """

        return self.optimization_variable["unscaled"].cx_start

    @property
    def cx_end(self):
        """
        The cx of all elements together (starting point)
        """

        return self.optimization_variable["unscaled"].cx_end


    """"autre a coder"""

    @property
    def cx_intermediates_list(self):
        """
        The cx of all elements together (starting point)
        """

        return self.optimization_variable["unscaled"].cx_intermediates_list

    @property
    def mx(self):
        return self.optimization_variable["unscaled"].mx

    @property
    def mx_reduced(self):
        return self.optimization_variable["unscaled"].mx_reduced

    def __len__(self):
        """
        The number of variables in the list
        """

        return len(self.optimization_variable["unscaled"])

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
        return self.optimization_variable["unscaled"][self._iter_idx - 1].name
