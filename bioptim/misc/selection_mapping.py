import numpy as np
from bioptim import BiMapping


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
    """

    def __init__(
        self,
        nb_elements: int = None,
        independent_indices: tuple[int] = None,
        dependencies: tuple[Dependency, ...] = None,
        **params
    ):

        """
        Parameters
        ----------
        nb_elements : int
            the number of elements, such as the number of dof in a model
        independent_indices : tuple
            the indices of the elements that are independent of others
        dependencies : Dependency class
            contains the dependencies of the elements between them and the factor
        they are multiplied by if needed,only 1 or -1 is acceptable for now
        params

        Methods
        -------
        build_to_second(dependency_matrix: list, independent_indices: list) -> list
            build the to_second vector used in BiMapping
        """

        # verify dependant dof : impossible multiple dependancies
        master = []
        dependant = []

        if nb_elements is not None:
            if not isinstance(nb_elements, int):
                raise ValueError("nb_dof should be an 'int'")
        if independent_indices is not None:
            if not isinstance(independent_indices, tuple):
                raise ValueError("independent_indices should be a tuple")

        if dependencies is not None:
            if not isinstance(dependencies, tuple):
                raise ValueError("dependencies should be a tuple of Dependency class ")
            for dependancy in dependencies:
                master.append(dependancy.dependent_index)
                dependant.append(dependancy.reference_index)

            for i in range(len(dependencies)):
                if master[i] in dependant:
                    raise ValueError("Dependancies cant depend on others")

        if len(independent_indices) > nb_elements:
            raise ValueError("independent_indices must not contain more elements than nb_elements")

        self.nb_elements = nb_elements
        self.independent_indices = independent_indices
        self.dependencies = dependencies

        index_dof = np.array([i for i in range(1, nb_elements + 1)])
        index_dof = index_dof.reshape(nb_elements, 1)

        selection_matrix = np.zeros((nb_elements, nb_elements))
        for element in independent_indices:  # simple case
            selection_matrix[element][element] = 1
        if dependencies is not None:
            for dependancy in dependencies:
                selection_matrix[dependancy.dependent_index][dependancy.reference_index] = 1
                if dependancy.factor is not None:
                    selection_matrix[dependancy.dependent_index][dependancy.reference_index] *= dependancy.factor

        first = selection_matrix @ index_dof
        dependency_matrix = [None for i in range(len(first))]
        oppose = []
        for i in range(len(first)):
            if first[i] != 0 and first[i] > 0:
                dependency_matrix[i] = int(first[i] - 1)
            if first[i] < 0:
                oppose.append(i)
                dependency_matrix[i] = int(abs(first[i]) - 1)

        def build_to_second(dependency_matrix: list, independent_indices: list):
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

        to_second = build_to_second(dependency_matrix=dependency_matrix, independent_indices=independent_indices)
        to_first = independent_indices
        self.to_second = to_second
        self.to_first = to_first
        self.oppose_to_second = oppose
        self.oppose_to_first = None

        super().__init__(to_second=to_second, to_first=to_first, oppose_to_second=oppose)


class Dependancy:
    """
    Class that defines the dependency of two elements by their indices

    Attributes
    ----------
    dependent_index: int
        the index of the dependent variable
    reference_index: int
        the index of the variable on which relies the dependent variable
    factor : int
        the factor that multiplies the dependent element

    """

    def __init__(self, dependent_index: int = None, reference_index: int = None, factor: int = None):

        """
        Parameters
        ----------
        dependent_index : int
            the index of the element that depends on another
        reference_index : int
            the index of the element on which the afore-mentionned element relies
        factor : int
            the factor that multiplies the dependent element

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
