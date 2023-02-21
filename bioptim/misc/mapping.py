import numpy as np
from casadi import MX, SX, DM

from .options import OptionDict, OptionGeneric
from .enums import Node


class Mapping(OptionGeneric):
    """
    Mapping of index set to a different index set

    Example of use:
        - to_map = Mapping([0, 1, 1, 3, None, 1], [3])
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
        map_idx: list | tuple | range | np.ndarray,
        oppose: int | list | tuple | range | np.ndarray = None,
        **params
    ):
        """
        Parameters
        ----------
        map_idx: list | tuple | range | np.ndarray
            The actual index list that links to the other set
        oppose: int | list | tuple | range | np.ndarray
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

    def map(self, obj: tuple | list | np.ndarray | MX | SX | DM) -> np.ndarray | MX | SX | DM:
        """
        Apply the mapping to an matrix object. The rows are mapped while the columns are preserved as is

        Parameters
        ----------
        obj: tuple | list | np.ndarray | MX | SX | DM
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
        to_second: Mapping | int | list | tuple | range,
        to_first: Mapping | int | list | tuple | range,
        oppose_to_second: Mapping | int | list | tuple | range = None,
        oppose_to_first: Mapping | int | list | tuple | range = None,
        **params
    ):
        """
        Parameters
        ----------
        to_second: Mapping | int | list | tuple | range
            The mapping that links the first index set to the second
        to_first: Mapping | int | list | tuple | range
            The mapping that links the second index set to the first
        oppose_to_second: Mapping | int | list | tuple | range
            Index to multiply by -1 of the to_second mapping
        oppose_to_first: Mapping | int | list | tuple | range
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


class SelectionMapping(BiMapping):
    """

    Mapping of two index sets according to the indexes that are kept at the end

    Attributes
    ----------
    to_second: Mapping
        The mapping that links the first variable to the second
    to_first: Mapping
        The mapping that links the second variable to the first

    """

    def __init__(
        self,
        nb_dof: int = None,
        list_kept_dof: list[int] = None,
        dependant_dof: list = None,

        **params
    ):
        """
        Parameters :

        nb_dof: int
            The number of dof in the model
        list_kept_dof: list
            The list of indexes of degrees of liberties that are to be kept at the end of optimisation
        dependant_dof : list of list
            The list of dependancies of degrees of liberty, each list contains the indexes of two degrees where the first
            degree depends on the second and a -1 if it needs to be opposed
        """

        # verify dependant dof : impossible multiple dependancies
        master = []
        dependant = []

        if nb_dof is not None:
            if not isinstance(nb_dof, int):
                raise ValueError("nb_dof should be an 'int'")
        if list_kept_dof is not None :
            if not isinstance(list_kept_dof,list) :
                raise ValueError('list_kept_dof should be a list')

        if dependant_dof is not None:
            if not isinstance(dependant_dof, list):
                raise ValueError('dependant_dof should be a list')
            if not isinstance(dependant_dof[0], list) :
                dependant_dof = [dependant_dof]
            for dependancy in dependant_dof:
                if len(dependancy) < 2:
                    raise ValueError("Dependant_dof must contain tuple or list of minimum size 2  ")
                if len(dependancy) > 3:
                    raise ValueError("Each list of dependant-dof must contain 3 values max")
                if len(dependancy) == 3:
                    if dependancy[2] != -1:
                        raise ValueError("Can't multiply indexes by else than -1 ")
                master.append(dependancy[1])
                dependant.append(dependancy[0])

            for i in range(len(dependant_dof)):
                if master[i] in dependant:
                    raise ValueError("Dependancies cant depend on others")

        if len(list_kept_dof) > nb_dof:
            raise ValueError('list_kept_dof must not contain more dofs than nb_dof')

        self.nb_dof = nb_dof
        self.list_kept_dof = list_kept_dof
        self.dependant_dof = dependant_dof

        index_dof = np.array([i for i in range(1, nb_dof + 1)])
        index_dof = index_dof.reshape(nb_dof, 1)

        selection_matrix = np.zeros((nb_dof, nb_dof))
        for dof in list_kept_dof:  # simple case
            selection_matrix[dof][dof] = 1
        if dependant_dof is not None:
            for dependancy in dependant_dof:
                selection_matrix[dependancy[0]][dependancy[1]] = 1
                if len(dependancy) ==3:
                    selection_matrix[dependancy[0]][dependancy[1]]*= -1

        first = selection_matrix @ index_dof
        matrix = [None for i in range(len(first))]
        oppose =[]
        for i in range(len(first)):
            if first[i] != 0 and first[i] > 0:
                matrix[i] = int(first[i] - 1)
            if first[i] < 0:
                oppose.append(i)
                matrix[i] = int(abs(first[i])-1)


        def build_to_second(x, u):
            for i in range(len(x)):
                for j in range(len(u)):
                    if x[i] == u[j]:
                        x[i] = j
                   # if x[i] is not None:
                    #    if x[i] < 0:
                        #    oppose.append(i)

            return x
        def build_vector_mapping(nb_dof, list_kept_dof):
            vector = [None for i in range(nb_dof)]
            for index_dof, dof in enumerate(list_kept_dof):
                if dof > nb_dof:
                    raise ValueError("index in list_kept_dof must be maximally equal to nb_dof")
                else:
                    vector[dof] = index_dof
            if dependant_dof is not None:
                for dependancy in dependant_dof:
                    vector[dependancy[0]] = vector[dependancy[1]]

            return vector
        to_second= build_to_second(matrix, list_kept_dof)
        #to_second_bis=build_vector_mapping(nb_dof=nb_dof, list_kept_dof=list_kept_dof)
        to_first = list_kept_dof
        self.to_second =to_second
        self.to_first = to_first

        super().__init__(
            to_second=to_second, to_first=to_first, oppose_to_second=oppose
        )


class BiMappingList(OptionDict):
    def __init__(self):
        super(BiMappingList, self).__init__()

    def add(
        self,
        name: str,
        to_second: Mapping | int | list | tuple | range = None,
        to_first: Mapping | int | list | tuple | range = None,
        oppose_to_second: Mapping | int | list | tuple | range = None,
        oppose_to_first: Mapping | int | list | tuple | range = None,
        bimapping: BiMapping = None,
        phase: int = -1,
    ):
        """
        Add a new BiMapping to the list

        Parameters
        name: str
            The name of the new BiMapping
        to_second: Mapping | int | list | tuple | range
            The mapping that links the first variable to the second
        to_first: Mapping | int | list | tuple | range
            The mapping that links the second variable to the first
        oppose_to_second: Mapping | int | list | tuple | range
            Index to multiply by -1 of the to_second mapping
        oppose_to_first: Mapping | int | list | tuple | range
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
                to_second=bimapping.to_second.map_idx,
                to_first=bimapping.to_first.map_idx,
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

    def __getitem__(self, item) -> dict | BiMapping:
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

    # TODO: should take care of Node individually instead of all the phase necessarily

    def __init__(
        self,
        map_states: bool = False,
        map_controls: bool = False,
        phase_pre: int = None,
        phase_post: int = None,
        **params
    ):
        """
        Parameters
        ----------
        phase_pre: int
            The number of the first phase to map
        phase_post: int
            The number of the second phase to map
        nodes_pre: Node | int | list | tuple | range
            The indices of the nodes to map in first phase
        nodes_post: Node | int | list | tuple | range
            The indices of the nodes to map in second phase
        """
        super(NodeMapping, self).__init__(**params)

        self.map_states = map_states
        self.map_controls = map_controls
        self.phase_pre = phase_pre
        self.phase_post = phase_post


class NodeMappingList(OptionDict):
    def __init__(self):
        super(NodeMappingList, self).__init__()

    def add(
        self,
        name: str,
        map_states: bool = False,
        map_controls: bool = False,
        phase_pre: int = None,
        phase_post: int = None,
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
        bimapping: BiMapping
            The BiMapping to copy
        """

        if map_states == False and map_controls == False:
            raise Warning(
                "You should use either map_states=True or map_controls=True. "
                "For now your node mapping has no effect."
            )

        if phase_pre is None or phase_post is None:
            raise ValueError("NodeMappingList should contain phase_pre and phase_post.")

        if phase_pre > phase_post:
            raise ValueError("Please provide a phase_pre index value smaller than the phase_post index value.")

        super(NodeMappingList, self)._add(
            key=name,
            map_states=map_states,
            map_controls=map_controls,
            phase=phase_pre,
            option_type=NodeMapping,
            phase_pre=phase_pre,
            phase_post=phase_post,
        )

    def get_variable_from_phase_idx(self, ocp):
        use_states_from_phase_idx = [i for i in range(ocp.n_phases)]
        use_states_dot_from_phase_idx = [i for i in range(ocp.n_phases)]
        use_controls_from_phase_idx = [i for i in range(ocp.n_phases)]

        for i in range(len(self)):
            for key in self[i].keys():
                if self[i][key].map_states:
                    use_states_from_phase_idx[self[i][key].phase_post] = self[i][key].phase_pre
                    use_states_dot_from_phase_idx[self[i][key].phase_post] = self[i][key].phase_pre
                if self[i][key].map_controls:
                    use_controls_from_phase_idx[self[i][key].phase_post] = self[i][key].phase_pre

        from ..optimization.non_linear_program import NonLinearProgram

        NonLinearProgram.add(ocp, "use_states_from_phase_idx", use_states_from_phase_idx, False)
        NonLinearProgram.add(ocp, "use_states_dot_from_phase_idx", use_states_dot_from_phase_idx, False)
        NonLinearProgram.add(ocp, "use_controls_from_phase_idx", use_controls_from_phase_idx, False)

        return use_states_from_phase_idx, use_states_dot_from_phase_idx, use_controls_from_phase_idx

    def __getitem__(self, item) -> dict | BiMapping:
        return super(NodeMappingList, self).__getitem__(item)

    def __contains__(self, item):
        return item in self.options[0].keys()
