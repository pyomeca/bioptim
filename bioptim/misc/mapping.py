from typing import Union

import numpy as np
from casadi import MX, SX, DM

from .options import OptionDict, OptionGeneric
from .enums import Node


class Mapping(OptionGeneric):
    """
    Mapping of index set to a different index set

    Example of use:
        - to_map = Mapping([0, 1, 1, 3, -1, 1], [3])
        - obj = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        - mapped_obj = to_map.map(obj)
    Expected result:
        - mapped_obj == np.array([0.1, 0.2, 0.2, -0.4, 0, 0.1])

    Attributes
    ----------
    map_idx: list[int]
        The actual index list that links to the other set, an negative value links to a numerical 0
    oppose: list[int]
        Index to multiply by -1

    Methods
    -------
    map(self, obj: list) -> list
        Apply the mapping to an obj
    len(self) -> int
        Get the len of the mapping
    """

    def __init__(
        self,
        map_idx: Union[list, tuple, range, np.ndarray],
        oppose: Union[int, list, tuple, range, np.ndarray] = None,
        **params
    ):
        """
        Parameters
        ----------
        map_idx: Union[list, tuple, range]
            The actual index list that links to the other set
        oppose: Union[list, tuple, range]
            Index to multiply by -1
        """
        super(Mapping, self).__init__(**params)
        self.map_idx = map_idx
        self.oppose = [1] * len(self.map_idx)
        if oppose is not None:
            if isinstance(oppose, int):
                oppose = [oppose]
            for i in oppose:
                self.oppose[i] = -1

    def map(self, obj: Union[tuple, list, np.ndarray, MX, SX, DM]) -> Union[np.ndarray, MX, SX, DM]:
        """
        Apply the mapping to an matrix object. The rows are mapped while the columns are preserved as is

        Parameters
        ----------
        obj: Union[np.ndarray, MX, SX, DM]
            The matrix to map

        Returns
        -------
        The list mapped
        """
        # Declare a zero filled object
        if isinstance(obj, (tuple, list)):
            obj = np.array(obj)

        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 1:
                obj = obj[:, np.newaxis]
            mapped_obj = np.zeros((len(self.map_idx), obj.shape[1]))
        elif isinstance(obj, (MX, SX, DM)):
            mapped_obj = type(obj).zeros(len(self.map_idx), obj.shape[1])
        else:
            raise RuntimeError("map must be applied on np.ndarray, MX or SX")

        # Fill the positive values
        index_plus_in_origin = []
        index_plus_in_new = []
        index_minus_in_origin = []
        index_minus_in_new = []
        for i, v in enumerate(self.map_idx):
            if v is not None and self.oppose[i] > 0:
                index_plus_in_origin.append(v)
                index_plus_in_new.append(i)
            elif v is not None and self.oppose[i] < 0:
                index_minus_in_origin.append(v)
                index_minus_in_new.append(i)
        mapped_obj[index_plus_in_new, :] = obj[index_plus_in_origin, :]  # Fill the non zeros values
        mapped_obj[index_minus_in_new, :] = -obj[index_minus_in_origin, :]  # Fill the non zeros values

        return mapped_obj

    def __len__(self) -> int:
        """
        Get the len of the mapping

        Returns
        -------
        The len of the mapping
        """

        return len(self.map_idx)


class BiMapping(OptionGeneric):
    """
    Mapping of two index sets between each other

    Attributes
    ----------
    to_second: Mapping
        The mapping that links the first variable to the second
    to_first: Mapping
        The mapping that links the second variable to the first
    """

    def __init__(
        self,
        to_second: Union[Mapping, int, list, tuple, range],
        to_first: Union[Mapping, int, list, tuple, range],
        oppose_to_second: Union[Mapping, int, list, tuple, range] = None,
        oppose_to_first: Union[Mapping, int, list, tuple, range] = None,
        **params
    ):
        """
        Parameters
        ----------
        to_second: Union[Mapping, list[int], tuple[int], range]
            The mapping that links the first index set to the second
        to_first: Union[Mapping, list[int], tuple[int], range]
            The mapping that links the second index set to the first
        oppose_to_second: Union[list, tuple, range]
            Index to multiply by -1 of the to_second mapping
        oppose_to_first: Union[list, tuple, range]
            Index to multiply by -1 of the to_first mapping
        """
        super(BiMapping, self).__init__(**params)

        if isinstance(to_second, (list, tuple, range)):
            to_second = Mapping(map_idx=to_second, oppose=oppose_to_second)
        if isinstance(to_first, (list, tuple, range)):
            to_first = Mapping(map_idx=to_first, oppose=oppose_to_first)

        if not isinstance(to_second, Mapping):
            raise RuntimeError("to_second must be a Mapping class")
        if not isinstance(to_first, Mapping):
            raise RuntimeError("to_first must be a Mapping class")

        self.to_second = to_second
        self.to_first = to_first


class BiMappingList(OptionDict):
    def __init__(self):
        super(BiMappingList, self).__init__()

    def add(
        self,
        name: str,
        to_second: Union[Mapping, int, list, tuple, range] = None,
        to_first: Union[Mapping, int, list, tuple, range] = None,
        oppose_to_second: Union[Mapping, int, list, tuple, range] = None,
        oppose_to_first: Union[Mapping, int, list, tuple, range] = None,
        bimapping: BiMapping = None,
        phase: int = -1,
    ):

        """
        Add a new BiMapping to the list

        Parameters
        name: str
            The name of the new BiMapping
        to_second: Mapping
            The mapping that links the first variable to the second
        to_first: Mapping
            The mapping that links the second variable to the first
        oppose_to_second: Union[list, tuple, range]
            Index to multiply by -1 of the to_second mapping
        oppose_to_first: Union[list, tuple, range]
            Index to multiply by -1 of the to_first mapping
        bimapping: BiMapping
            The BiMapping to copy
        """

        if isinstance(bimapping, BiMapping):
            if to_second is not None or to_first is not None:
                raise ValueError("BiMappingList should either be a to_second/to_first or an actual BiMapping")
            self.add(
                name,
                phase=phase,
                to_second=bimapping.to_second,
                to_first=bimapping.to_first,
                oppose_to_second=oppose_to_second,
                oppose_to_first=oppose_to_first,
            )

        else:
            if to_second is None or to_first is None:
                raise ValueError("BiMappingList should either be a to_second/to_first or an actual BiMapping")
            super(BiMappingList, self)._add(
                key=name,
                phase=phase,
                option_type=BiMapping,
                to_second=to_second,
                to_first=to_first,
                oppose_to_second=oppose_to_second,
                oppose_to_first=oppose_to_first,
            )

    def variable_mapping_fill_phases(self, n_phases):
        for mappings in self.options:
            for key in mappings:
                if mappings[key].automatic_multiple_phase:
                    for i_phase in range(n_phases):
                        if i_phase == 0:
                            mappings[key].phase = 0
                        else:
                            self.add(name=key, bimapping=mappings[key], phase=i_phase)
        return self

    def __getitem__(self, item) -> Union[dict, BiMapping]:
        return super(BiMappingList, self).__getitem__(item)

    def __contains__(self, item):
        return item in self.options[0].keys()


class NodeMapping(OptionGeneric):
    """
    Mapping of two node sets between each other

    Attributes
    ----------
    to_second: Mapping
        The mapping that links the first variable to the second
    to_first: Mapping
        The mapping that links the second variable to the first
    """

    def __init__(
        self,
        phase_pre: int = None,
        phase_post: int = None,
        nodes_pre: Union[Node, int, list, tuple, range] = None,
        nodes_post: Union[Node, int, list, tuple, range] = None,
        oppose_to_second: Union[Node, int, list, tuple, range] = None,
        oppose_to_first: Union[Node, int, list, tuple, range] = None,
        **params
    ):

        """
        Parameters
        ----------
        phase_pre: int
            The number of the first phase to map
        phase_post: int
            The number of the second phase to map
        nodes_pre: Union[Node, int, list, tuple, range]
            The indices of the nodes to map in first phase
        nodes_post: Union[Node, int, list, tuple, range]
            The indices of the nodes to map in second phase
        oppose_to_second: Union[list, tuple, range]
            Index to multiply by -1 of the to_second mapping
        oppose_to_first: Union[list, tuple, range]
            Index to multiply by -1 of the to_first mapping
        """
        super(NodeMapping, self).__init__(**params)

        if not isinstance(phase_pre, int):
            raise RuntimeError("phase_pre must be an int (the number of the fisrt phase to map)")
        if not isinstance(phase_post, int):
            raise RuntimeError("phase_post must be an int (the number of the second phase to map)")
        if not isinstance(nodes_pre, (Node, int, list, tuple, range)):
            raise RuntimeError("nodes_pre must be a Node class, an int, a list, a tuple or a range")
        if not isinstance(nodes_post, (Node, int, list, tuple, range)):
            raise RuntimeError("nodes_post must be a Node class, an int, a list, a tuple or a range")
        if oppose_to_second is not None:
            if not isinstance(oppose_to_second, (Node, int, list, tuple, range)):
                raise RuntimeError("oppose_to_second must be a Node class, an int, a list, a tuple or a range")
        if oppose_to_first is not None:
            if not isinstance(oppose_to_first, (Node, int, list, tuple, range)):
                raise RuntimeError("oppose_to_first must be a Node class, an int, a list, a tuple or a range")

        self.phase_pre = phase_pre
        self.phase_post = phase_post
        self.nodes_pre = nodes_pre
        self.nodes_post = nodes_post
        self.oppose_to_second = oppose_to_second
        self.oppose_to_first = oppose_to_first

class NodeMappingList(OptionDict):
    def __init__(self):
        super(NodeMappingList, self).__init__()

    def add(
        self,
        name: str,
        phase_pre: int = None,
        phase_post: int = None,
        nodes_pre: Union[Node, int, list, tuple, range] = None,
        nodes_post: Union[Node, int, list, tuple, range] = None,
        oppose_to_second: Union[Node, int, list, tuple, range] = None,
        oppose_to_first: Union[Node, int, list, tuple, range] = None,
    ):
        """
        Add a new NodeMapping to the list

        Parameters
        name: str
            The name of the new BiMapping
        to_second: Mapping
            The mapping that links the first variable to the second
        to_first: Mapping
            The mapping that links the second variable to the first
        oppose_to_second: Union[list, tuple, range]
            Index to multiply by -1 of the to_second mapping
        oppose_to_first: Union[list, tuple, range]
            Index to multiply by -1 of the to_first mapping
        bimapping: BiMapping
            The BiMapping to copy
        """

        if phase_pre is None or phase_post is None or nodes_pre is None or nodes_post is None:
            raise ValueError("NodeMappingList should contain phase_pre, phase_post, node_pre and node_post.")

        super(NodeMappingList, self)._add(
            key=name,
            phase=phase_pre,
            option_type=NodeMapping,
            phase_pre=phase_pre,
            phase_post=phase_post,
            nodes_pre=nodes_pre,
            nodes_post=nodes_post,
            oppose_to_second=oppose_to_second,
            oppose_to_first=oppose_to_first,
        )

    def get_variable_from_phase_idx(self, ocp, NLP, states_names, controls_names, states_dot_names):
        # ocp.nlp[ocp.nlp.use_states_dot_from_phase_idx].states_dot
        use_states_from_phase_idx = []
        use_states_dot_from_phase_idx = []
        use_controls_from_phase_idx = []
        for i in range(len(ocp.nlp)):
            for key in self.keys():
                if key in states_names[self[key].phase_post] and i == self[key].phase_post:
                    use_states_from_phase_idx += [self[key].phase_pre]
                else:
                    use_states_from_phase_idx += [i]
                if key in states_dot_names[self[key].phase_post] and i == self[key].phase_post:
                    use_states_dot_from_phase_idx += [self[key].phase_pre]
                else:
                    use_states_dot_from_phase_idx += [i]
                if key in controls_names[self[key].phase_post] and i == self[key].phase_post:
                    use_controls_from_phase_idx += [self[key].phase_pre]
                else:
                    use_controls_from_phase_idx += [i]

        NLP.add(ocp, "use_states_from_phase_idx", use_states_from_phase_idx, False)
        NLP.add(ocp, "use_states_dot_from_phase_idx", use_states_dot_from_phase_idx, False)
        NLP.add(ocp, "use_controls_from_phase_idx", use_controls_from_phase_idx, False)
        return

    def __getitem__(self, item) -> Union[dict, BiMapping]:
        return super(NodeMappingList, self).__getitem__(item)

    def __contains__(self, item):
        return item in self.options[0].keys()
