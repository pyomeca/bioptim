import numpy as np
from typing import Callable, Iterator, Tuple, Any


Key = Tuple[Any, ...]  # (phase, var_type, node) or ("global", "time"/"parameters")
KeySize = Tuple[Key, int]
GeneratorType = Callable[
    [], tuple[Iterator[KeySize], int]
]  # Function that returns an iterator yielding (key, size, horizontal_size)

import casadi as ca

_CASADI_TYPES = (ca.MX, ca.SX)

from ..misc.enums import ControlType


class VectorLayout:
    def __init__(self, ocp, ordering: str = "variable-major"):
        self.ocp = ocp
        self.ordering = ordering
        # strategy name -> generator function that yields (key, size)
        self._ordering_strategies: dict[str, GeneratorType] = {
            "time-major": self._keys_time_major,
            "variable-major": self._keys_variable_major,
        }
        self.index_map = self._build_index_map()  # maps (phase, var_type, node, key) -> slice

    # -----------------------
    # helpers
    # -----------------------
    def _len_of(self, shape_like) -> int:
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

    def _add_block(self, idx_map: dict, key: Key, slice_start: int, size_like, horizontal_size) -> int:
        """Create a slice for key at current offset and return new offset."""
        length = self._len_of(size_like)
        slice_end = slice_start + length * horizontal_size
        idx_map[key] = slice(slice_start, slice_end), horizontal_size
        return slice_end

    # -----------------------
    # ordering key generators
    # -----------------------
    def _keys_time_major(self) -> Iterator[KeySize]:
        """
        Stage-wise stacking: for each phase, for each node: states, controls, algebraic_states
        All the variables are stacked this way:
            [ t, x0, u0, a0, x1, u1, a1, ..., params ]
        yields: ((phase, var_type, node), size)
        """
        for p, nlp in enumerate(self.ocp.nlp):
            for node in range(nlp.ns):
                yield (p, "states", node), self._len_of(nlp.states.shape), nlp.n_states_decision_steps(node)
                yield (p, "controls", node), self._len_of(nlp.controls.shape), nlp.n_controls_steps(node)

                nlp.algebraic_states.node_index = node
                n_cols = nlp.n_algebraic_states_decision_steps(node)
                yield (p, "algebraic_states", node), self._len_of(nlp.algebraic_states.shape), n_cols

            yield (p, "states", nlp.ns), self._len_of(nlp.states.shape), nlp.n_states_decision_steps(nlp.ns)

            if self.ocp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                yield (p, "controls", nlp.ns), self._len_of(nlp.controls.shape), nlp.n_controls_steps(nlp.ns)

            nlp.algebraic_states.node_index = node
            n_cols = nlp.n_algebraic_states_decision_steps(node)
            yield (p, "algebraic_states", nlp.ns), self._len_of(nlp.algebraic_states.shape), n_cols

    def _keys_variable_major(self) -> Iterator[KeySize]:
        """
        All states for all nodes first, then all controls, then all algebraic_states.
        All the variables are stacked this way:
            [ t, states, controls, algebraic_states, params ]
        yields: ((phase, var_type, node), size)

        """

        for var in ("states", "controls", "algebraic_states"):
            for p, nlp in enumerate(self.ocp.nlp):
                method_map = {
                    "states": nlp.n_states_decision_steps,
                    "controls": nlp.n_controls_steps,
                    "algebraic_states": nlp.n_algebraic_states_decision_steps,
                }

                for node in range(nlp.ns):
                    attr = getattr(nlp, var)
                    yield (p, var, node), self._len_of(attr.shape), method_map[var](node)

                if var == "states":
                    # last node for states is the final state
                    yield (p, var, nlp.ns), self._len_of(attr.shape), method_map["states"](nlp.ns)

                if var == "algebraic_states":
                    nlp.algebraic_states.node_index = nlp.ns
                    n_cols = nlp.n_algebraic_states_decision_steps(nlp.ns)
                    yield (p, var, nlp.ns), self._len_of(attr.shape), n_cols

                if var == "controls" and nlp.control_type in (
                    ControlType.LINEAR_CONTINUOUS,
                    ControlType.CONSTANT_WITH_LAST_NODE,
                ):
                    # last node for controls is the final control
                    yield (p, var, nlp.ns), self._len_of(attr.shape), method_map["controls"](nlp.ns)

    def _build_index_map(self):
        idx_map = {}
        offset = 0

        # Time parameter always comes first
        offset = self._add_block(idx_map, ("global", "time"), offset, self.ocp.dt_parameter.shape, 1)

        # pick generator by ordering; raising ValueError if unknown
        try:
            generator = self._ordering_strategies[self.ordering]
        except KeyError:
            raise ValueError(f"Unknown ordering mode: {self.ordering}")

        # use the generator to append blocks (no branching here)
        for key, v_size, h_size in generator():
            offset = self._add_block(idx_map, key, offset, v_size, h_size)

        # Parameters always last
        offset = self._add_block(idx_map, ("global", "parameters"), offset, self.ocp.parameters.shape, 1)

        self.total_size = offset
        return idx_map

    # -----------------------
    # public API
    # -----------------------
    def register_ordering(self, name: str, generator: GeneratorType) -> None:
        """Register a custom ordering generator. generator() must yield (key, size)."""
        self._ordering_strategies[name] = generator

    def serialize(self, ocp_values):
        """
        Given a dict of values keyed by index_map keys, build a flat vector.
        If values are CasADi symbols, returns a vertcat.
        If values are NumPy arrays, returns a stacked array.
        """
        values = [ocp_values[key] for key in self.index_map]

        # best debug found
        for i, (v, (key, val)) in enumerate(zip(values, self.index_map.items())):
            print(i, v.shape, key, val)

        if _CASADI_TYPES and isinstance(values[0], _CASADI_TYPES):
            return ca.vertcat(*values)
        else:
            return np.vstack(values)

    def deserialize(self, vec):
        """
        Given flat vector, return dict with same structure as index_map.
        Works for NumPy arrays and CasADi objects.
        """
        result = {}
        for i, (key, (sl, n_cols)) in enumerate(self.index_map.items()):
            print(i, "key", key)
            result[key] = vec[sl].toarray()

            v_size = sl.stop - sl.start
            if v_size != 0:
                result[key] = result[key].reshape((v_size // n_cols, -1), order="F")

        return result

    def deserialize_to_lists(self, vec):
        """
        Convert a flat vector into a nested list structure:
        [phases][var_type_index][node] = ndarray
        var_type_index: 0=states, 1=controls, 2=algebraics, 3=parameters (global)
        """
        # First, get the dict of all slices using the same method
        deserialized = self.deserialize(vec)

        result_states = []
        result_controls = []
        result_algebraics = []

        for p, nlp in enumerate(self.ocp.nlp):
            phase_states = []
            phase_controls = []
            phase_algebraics = []

            for node in range(nlp.n_states_nodes):
                phase_states.append(deserialized[(p, "states", node)])
            for node in range(nlp.n_controls_nodes):
                phase_controls.append(deserialized[(p, "controls", node)])
            for node in range(nlp.n_states_nodes):
                phase_algebraics.append(deserialized[(p, "algebraic_states", node)])

            result_states.append(phase_states)
            result_controls.append(phase_controls)
            result_algebraics.append(phase_algebraics)

        # Parameters are global
        params = deserialized[("global", "parameters")]

        return result_states, result_controls, params, result_algebraics

    def deserialize_to_dicts(self, vec):
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
        # Step 1: get list-based structure
        list_states, list_controls, params, list_algebraics = self.deserialize_to_lists(vec)

        data_states, data_controls, data_algebraics = [], [], []
        data_parameters = {}

        # Define variable type mapping: (list_structure, accessor, index_getter, output_list)
        var_types = [
            (list_states, lambda nlp: nlp.states.keys(), lambda nlp, k: nlp.states[k].index, data_states),
            (list_controls, lambda nlp: nlp.controls.keys(), lambda nlp, k: nlp.controls.key_index(k), data_controls),
            (
                list_algebraics,
                lambda nlp: nlp.algebraic_states.keys(),
                lambda nlp, k: nlp.algebraic_states[k].index,
                data_algebraics,
            ),
        ]

        # Step 2: loop over phases once
        for p, nlp in enumerate(self.ocp.nlp):
            for list_struct, key_getter, idx_getter, out_list in var_types:
                var_dict = {key: [None] * len(list_struct[p]) for key in key_getter(nlp)}
                for node, arr in enumerate(list_struct[p]):
                    for key in key_getter(nlp):
                        var_dict[key][node] = arr[idx_getter(nlp, key), :]
                out_list.append(var_dict)

        # Step 3: parameters (global)
        for key in self.ocp.parameters.keys():
            data_parameters[key] = params[self.ocp.parameters[key].index]

        return data_states, data_controls, [data_parameters], data_algebraics


# if __name__ == "__main__":
#
#     def my_custom_order():
#         # yield global time already handled, this just yields other keys
#         # e.g., alternate states and controls across all phases, then algebraics
#         for p, nlp in enumerate(ocp.nlp):
#             for node in range(nlp.n_nodes):
#                 yield (p, "states", node), nlp.states.shape
#                 yield (p, "controls", node), nlp.controls.shape
#         for p, nlp in enumerate(ocp.nlp):
#             for node in range(nlp.n_nodes):
#                 yield (p, "algebraic_states", node), nlp.algebraic_states.shape
#
#     layout = VectorLayout(ocp, ordering="time-major")
#     layout.register_ordering("my-order", my_custom_order)
#     layout.ordering = "my-order"
#     layout.index_map = layout._build_index_map()
