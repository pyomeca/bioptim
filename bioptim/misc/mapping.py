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

        self.oppose_to_second = None
        self.oppose_to_first = None

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
        # Here `type` is used instead of `isinstance` because of the `SelectionMapping` inherits from `BiMapping`
        if type(bimapping) is BiMapping:
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
        # Here `type` is used instead of `isinstance` because of the `SelectionMapping` inherits from `BiMapping`
        elif type(bimapping) is SelectionMapping:
            if to_second is not None or to_first is not None:
                raise ValueError("BiMappingList should either be a to_second/to_first or an actual BiMapping")
            self.add(
                name,
                phase=phase,
                to_second=bimapping.to_second.map_idx,
                to_first=bimapping.to_first.map_idx,
                oppose_to_second=bimapping.oppose_to_second,
                oppose_to_first=None,
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


class Dependency:
    """
    Class that defines the dependency of two elements by their indices

    Attributes
    ----------
    dependent_index: int
        The index of the dependent variable
    reference_index: int
        The index of the variable on which relies the dependent variable
    factor : int
        The factor that multiplies the dependent element
    """

    def __init__(self, dependent_index: int = None, reference_index: int = None, factor: int = None):
        """
        Parameters
        ----------
        dependent_index : int
            The index of the element that depends on another
        reference_index : int
            The index of the element on which the afore-mentionned element relies
        factor : int
            The factor that multiplies the dependent element

        """
        if dependent_index is not None:
            if not isinstance(dependent_index, int):
                raise ValueError("dependent_index must be an int")
        if reference_index is not None:
            if not isinstance(reference_index, int):
                raise ValueError("referent_index must be an int")
        if factor is not None:
            if not isinstance(factor, int):
                raise ValueError("factor must be an int")
            if factor != 1 and factor != -1:
                raise ValueError("factor can only be -1 ")

        self.dependent_index = dependent_index
        self.reference_index = reference_index
        self.factor = factor


class SelectionMapping(BiMapping):
    """
    Mapping of two index sets according to the indexes that are independent

    Attributes
    ----------
    to_second: Mapping
        The mapping that links the first variable to the second
    to_first: Mapping
        The mapping that links the second variable to the first
    oppose_to_second : int | list
        Index to multiply by -1 of the to_second mapping
    nb_elements : int
        The number of elements, such as the number of dof in a model
    independent_indices : tuple
        The indices of the elements that are independent of others
    dependencies : Dependency class
        Contains the dependencies of the elements between them and the factor
    they are multiplied by if needed,only 1 or -1 is acceptable for now

    """

    def __init__(
        self,
        nb_elements: int = None,
        independent_indices: tuple[int, ...] = None,
        dependencies: tuple[Dependency, ...] = None,
        **params
    ):
        """
        Initializes the class SelectionMapping

        Parameters
        ----------
        nb_elements : int
            The number of elements, such as the number of dof in a model
        independent_indices : tuple
            The indices of the elements that are independent of others
        dependencies : Dependency class
            Contains the dependencies of the elements between them and the factor
        they are multiplied by if needed,only 1 or -1 is acceptable for now

        Methods
        -------
        _build_to_second(dependency_matrix: list, independent_indices: list) -> list
            build the second vector that defines the mapping
        """

        # verify dependant dof : impossible multiple dependancies
        master = []
        dependent = []

        if nb_elements is not None:
            if not isinstance(nb_elements, int):
                raise ValueError("nb_dof should be an 'int'")
        if independent_indices is not None:
            if not isinstance(independent_indices, tuple):
                raise ValueError("independent_indices should be a tuple")

        if dependencies is not None:
            if not isinstance(dependencies, tuple):
                raise ValueError("dependencies should be a tuple of Dependency class ")
            for dependency in dependencies:
                master.append(dependency.dependent_index)
                dependent.append(dependency.reference_index)

            for i in range(len(dependencies)):
                if master[i] in dependent:
                    raise ValueError("dependencies cant depend on others")

        if len(independent_indices) > nb_elements:
            raise ValueError("independent_indices must not contain more elements than nb_elements")

        self.nb_elements = nb_elements
        self.independent_indices = independent_indices
        self.dependencies = dependencies

        index_dof = np.array([i for i in range(1, nb_elements + 1)])
        index_dof = index_dof.reshape(nb_elements, 1)

        selection_matrix = np.zeros((nb_elements, nb_elements))
        for element in independent_indices:  # simple case
            if element not in master:
                selection_matrix[element][element] = 1
        if dependencies is not None:
            for dependency in dependencies:
                selection_matrix[dependency.dependent_index][dependency.reference_index] = 1
                if dependency.factor is not None:
                    selection_matrix[dependency.dependent_index][dependency.reference_index] *= dependency.factor

        first = selection_matrix @ index_dof
        dependency_matrix: list = [None for _ in range(len(first))]
        oppose = []
        for i in range(len(first)):
            if first[i] != 0 and first[i] > 0:
                dependency_matrix[i] = int(first[i] - 1)
            if first[i] < 0:
                oppose.append(i)
                dependency_matrix[i] = int(abs(first[i]) - 1)

        def _build_to_second(dependency_matrix: list, independent_indices: tuple):
            """
            Build the to_second vector used in BiMapping thanks to the dependency matrix of the elements in the system
            and the vector of independent indices

            Parameters
            ----------
            dependency_matrix : list
                the matrix of dependencies
            independent_indices : list
                the list of indexes of the independent elements in the system

            Returns
            ----------
            the to_second usual vector in BiMapping

            """
            for i in range(len(dependency_matrix)):
                for j in range(len(independent_indices)):
                    if dependency_matrix[i] == independent_indices[j]:
                        dependency_matrix[i] = j
            return dependency_matrix

        to_second = _build_to_second(dependency_matrix=dependency_matrix, independent_indices=independent_indices)
        to_first = list(independent_indices)
        self.to_second = to_second
        self.to_first = to_first
        self.oppose_to_second = oppose
        self.oppose_to_first = None

        super().__init__(to_second=to_second, to_first=to_first, oppose_to_second=oppose)


class NodeMapping(OptionGeneric):
    """
    Mapping of two node sets between each other
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

        if not map_states and not map_controls:
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
