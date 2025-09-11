import numpy as np
from typing import Callable, Iterator, Tuple, Any


Key = Tuple[Any, ...]  # (phase, var_type, node) or ("global", "time"/"parameters")
KeySize = Tuple[Key, int]
GeneratorType = Callable[
    [], tuple[Iterator[KeySize], int]
]  # Function that returns an iterator yielding (key, size, horizontal_size)

from casadi import MX, SX, vertcat, DM

_CASADI_TYPES = (MX, SX)

from ..misc.enums import ControlType
from ..misc.parameters_types import CX


def _keys_variable_major(ocp) -> Iterator[KeySize]:
    """
    VARIABLE-MAJOR ORDERING (HISTORICAL in Bioptim):
        Group variables by type first, then by time.

    This creates a vector where all states come first, then all controls,
    then all algebraic states:

    Structure: [t, x₀,x₁,x₂,..., u₀,u₁,u₂,..., a₀,a₁,a₂,..., params]

    For each variable type (states, controls, algebraics):
        For each phase:
            For each time node:
                - Variables of current type at this node
    Yields:
    tuple: ((phase, var_type, node), vertical_size, horizontal_size)
        - phase: which phase this variable belongs to
        - var_type: "states", "controls", or "algebraic_states"
        - node: time node index
        - vertical_size: number of variables of this type at this node
        - horizontal_size: number of columns (direct collocation states of Lagrange polynomials)
    """

    for var in ("states", "controls", "algebraic_states"):
        for p, nlp in enumerate(ocp.nlp):
            method_map = {
                "states": nlp.n_states_decision_steps,
                "controls": lambda node: 1,
                # "controls": nlp.n_controls_steps,
                "algebraic_states": nlp.n_algebraic_states_decision_steps,
            }

            for node in range(nlp.ns):
                attr = getattr(nlp, var)
                yield (p, var, node), _len_of(attr.shape), method_map[var](node)

            if var == "states":
                # last node for states is the final state
                yield (p, var, nlp.ns), _len_of(attr.shape), method_map["states"](nlp.ns)

            if var == "algebraic_states":
                nlp.algebraic_states.node_index = nlp.ns
                n_cols = nlp.n_algebraic_states_decision_steps(nlp.ns)
                yield (p, var, nlp.ns), _len_of(attr.shape), n_cols

            if var == "controls" and nlp.control_type.has_a_final_node:
                # last node for controls is the final control
                yield (p, var, nlp.ns), _len_of(attr.shape), method_map["controls"](nlp.ns)


def _keys_time_major(ocp) -> Iterator[KeySize]:
    """
    TIME-MAJOR ORDERING: Group variables by time node first, then by variable type.

    This creates a vector where all variables at time node 0 come first,
    then all variables at time node 1, and so on:

    Structure: [t, x₀,u₀,a₀, x₁,u₁,a₁, x₂,u₂,a₂, ..., params]

    For each phase:
        For each time node:
            - States at this node
            - Controls at this node
            - Algebraic states at this node

    Yields:
        tuple: ((phase, var_type, node), vertical_size, horizontal_size)
            Same format as time-major, but different ordering
    """
    for p, nlp in enumerate(ocp.nlp):
        for node in range(nlp.ns):
            yield (p, "states", node), _len_of(nlp.states.shape), nlp.n_states_decision_steps(node)
            yield (p, "controls", node), _len_of(nlp.controls.shape), 1

            nlp.algebraic_states.node_index = node
            n_cols = nlp.n_algebraic_states_decision_steps(node)
            yield (p, "algebraic_states", node), _len_of(nlp.algebraic_states.shape), n_cols

        yield (p, "states", nlp.ns), _len_of(nlp.states.shape), nlp.n_states_decision_steps(nlp.ns)

        if nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
            yield (p, "controls", nlp.ns), _len_of(nlp.controls.shape), 1

        nlp.algebraic_states.node_index = node
        n_cols = nlp.n_algebraic_states_decision_steps(node)
        yield (p, "algebraic_states", nlp.ns), _len_of(nlp.algebraic_states.shape), n_cols


ORDERING_STRATEGIES: dict[str, GeneratorType] = {
    "time-major": _keys_time_major,
    "variable-major": _keys_variable_major,
}


class VectorLayout:
    """
    Manages the layout of optimization variables in a flat vector.

    In optimal control problems, we have various types of variables:
    - States (x): system state at each time node
    - Controls (u): control inputs at each time node
    - Algebraic states (a): algebraic variables at each time node
    - Parameters: global optimization parameters
    - Time: global time parameter

    This class determines how these variables are arranged ("stacked") in
    a single flat vector that gets passed to the optimizer.

    The ordering strategy can significantly impact solver performance depending
    on how the solver accesses and processes the variables.

    Built-in orderings:
    - "time-major": Group by time node first [x₀,u₀,a₀, x₁,u₁,a₁, ...]
    - "variable-major": Group by variable type first [x₀,x₁,..., u₀,u₁,..., a₀,a₁,...]

    Custom orderings can be registered
    """

    def __init__(self, ocp, ordering: str | Callable = "variable-major"):
        self.ocp = ocp
        self.ordering = ordering
        self.generator = self._pick_generator()
        self.index_map = self._build_index_map()  # maps (phase, var_type, node, key) -> slice

    def stack(self, time, states, controls, algebraics, parameters, query_function: Callable):
        """
        Given time, states, controls, algebraics, and parameters,
        stack them into a flat vector.
        """

        values = [query_function(time, states, controls, algebraics, parameters, key) for key in self.index_map]

        if _CASADI_TYPES and isinstance(values[0], _CASADI_TYPES):
            return vertcat(*values)
        else:
            return np.vstack(values)

    def unstack(self, vec):
        """
        Given flat vector, return dict with same structure as index_map.
        Works for NumPy arrays and CasADi DM.
        """
        result = {}
        for i, (key, (sl, n_cols)) in enumerate(self.index_map.items()):

            vec_sliced = vec[sl].toarray() if isinstance(vec[sl], DM) else vec[sl]
            result[key] = vec_sliced

            v_size = sl.stop - sl.start
            if v_size != 0:
                result[key] = result[key].reshape((v_size // n_cols, -1), order="F")

        return result

    def unstack_to_lists(self, vec):
        """
        Convert a flat vector into a nested list structure:
        [phases][var_type_index][node] = ndarray
        var_type_index: 0=states, 1=controls, 2=algebraics, 3=parameters (global)
        """
        # First, get the dict of all slices using the same method
        unstacked = self.unstack(vec)

        result_states = []
        result_controls = []
        result_algebraics = []

        for p, nlp in enumerate(self.ocp.nlp):
            phase_states = []
            phase_controls = []
            phase_algebraics = []

            for node in range(nlp.n_states_nodes):
                phase_states.append(unstacked[(p, "states", node)])
            for node in range(nlp.n_controls_nodes):
                phase_controls.append(unstacked[(p, "controls", node)])
            for node in range(nlp.n_states_nodes):
                phase_algebraics.append(unstacked[(p, "algebraic_states", node)])

            result_states.append(phase_states)
            result_controls.append(phase_controls)
            result_algebraics.append(phase_algebraics)

        # Parameters are global
        params = unstacked[("global", "parameters")]

        return result_states, result_controls, params, result_algebraics

    def unstack_to_dicts(self, vec):
        """
        Convert a flat vector into a nested dictionary structure with variable names.
        Structure:
            (
                [phase][var_name][node] = ndarray   # states
                [phase][var_name][node] = ndarray   # controls
                parameters_dict[var_name] = ndarray
                [phase][var_name][node] = ndarray   # algebraics
            )
        """
        list_states, list_controls, params, list_algebraics = self.unstack_to_lists(vec)

        data_states, data_controls, data_algebraics = [], [], []
        data_parameters = {}

        for p, nlp in enumerate(self.ocp.nlp):

            data_states += _extract_states_dict(list_states[p], nlp)
            data_controls += _extract_controls_dict(list_controls[p], nlp)
            data_algebraics += _extract_algebraics_dict(list_algebraics[p], nlp)

        for key in self.ocp.parameters.keys():
            data_parameters[key] = [params[self.ocp.parameters[key].index]]

        return data_states, data_controls, [data_parameters], data_algebraics

    def _build_index_map(self):
        idx_map = {}
        offset = 0

        # Time parameter always comes first
        offset = self._add_block(idx_map, ("global", "time"), offset, self.ocp.dt_parameter.shape, 1)

        # use the generator to append blocks (no branching here)
        for key, v_size, h_size in self.generator(self.ocp):
            offset = self._add_block(idx_map, key, offset, v_size, h_size)

        # Parameters always last
        offset = self._add_block(idx_map, ("global", "parameters"), offset, self.ocp.parameters.shape, 1)

        self.total_size = offset
        return idx_map

    def _pick_generator(self):
        """Select the generator function based on the ordering attribute."""
        if callable(self.ordering):
            return self.ordering
        try:
            return ORDERING_STRATEGIES[self.ordering]
        except KeyError:
            raise ValueError(f"Unknown ordering mode: {self.ordering}")

    def _add_block(self, idx_map: dict, key: Key, slice_start: int, size_like, horizontal_size) -> int:
        """Create a slice for key at current offset and return new offset."""
        length = _len_of(size_like)
        slice_end = slice_start + length * horizontal_size
        idx_map[key] = slice(slice_start, slice_end), horizontal_size
        return slice_end

    @staticmethod
    def query_function(
        time: CX, states: list[list[CX]], controls: list[list[CX]], algebraics: list[list[CX]], parameters: CX, key
    ) -> CX:
        """
        Query function to retrieve values from the OCP based on the key.
        This is a placeholder and should be replaced with actual logic
        to retrieve values from the OCP.
        """
        if key == ("global", "time"):
            return time
        elif key == ("global", "parameters"):
            return parameters
        else:
            phase, var_type, node = key
            if var_type == "states":
                return states[phase][node].reshape((-1, 1))
            elif var_type == "controls":
                return controls[phase][node]
            elif var_type == "algebraic_states":
                return algebraics[phase][node].reshape((-1, 1))
            else:
                raise ValueError(f"Unknown key: {key}")


def _len_of(shape_like) -> int:
    """Return a single integer length from shape-like (int or tuple)."""
    try:
        # shape_like might be an int or something with .shape
        if isinstance(shape_like, int):
            return int(shape_like)
        # if it's a numpy shape tuple or an object with a .shape attribute:
        if hasattr(shape_like, "shape"):
            shp = shape_like.shape
            # if shape is a tuple, compute product; if it's a scalar shape, turn into int
            if isinstance(shp, tuple):
                return int(np.prod(shp))
            return int(shp)
        # if it's a tuple itself:
        if hasattr(shape_like, "__iter__"):
            return int(np.prod(shape_like))
        return int(shape_like)
    except Exception:
        # fallback: try int conversion
        return int(shape_like)


def _extract_attr_dict(list_data, nlp, attr) -> list[dict[str, list[np.ndarray]]]:
    """Extract attribute (states, controls, algebraic_states) into dictionary format for one phase."""
    extracted_attr = getattr(nlp, attr)
    keys = extracted_attr.keys()
    var_dict = {key: [None] * len(list_data) for key in keys}
    for node, arr in enumerate(list_data):
        for key in keys:
            # key_index = extracted_attr[key].index
            key_index = extracted_attr.key_index(key)
            var_dict[key][node] = arr[key_index, :]
    return [var_dict]


def _extract_states_dict(list_data, nlp):
    """Extract states into dictionary format for one phase."""
    return _extract_attr_dict(list_data, nlp, "states")


def _extract_controls_dict(list_data, nlp):
    """Extract controls into dictionary format for one phase."""
    return _extract_attr_dict(list_data, nlp, "controls")


def _extract_algebraics_dict(list_data, nlp):
    """Extract algebraic states into dictionary format for one phase."""
    return _extract_attr_dict(list_data, nlp, "algebraic_states")
