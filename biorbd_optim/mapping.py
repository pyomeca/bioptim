class Mapping:
    def __init__(self, expand_idx, reduce_idx, sign_to_oppose_for_expanded=()):
        """
        Creates a mapping to define the correspondence between two data models.
        Well suited to reduce the size of problems handled by the solver when they have symmetries.
        """
        self.expand_idx = expand_idx
        self.nb_expanded = len(self.expand_idx)
        self.reduce_idx = reduce_idx
        self.nb_reduced = len(self.reduce_idx)
        self.sign_to_oppose = sign_to_oppose_for_expanded

    def expand(self, obj):
        """
        Copies the elements of the input obj into an obj_expanded representing the extended list according to the mapping.
        Example :
       - L is an array or a casADi MX containing the elements [10, 20, 30, 40, 50, 60, 70, 80].
        - expand_idx is a list equal to [0, 1, 2, 1, 3, 2, 4, 5]
        expand(L) will return obj_expanded, an object of the same type as L with the elements : [10, 20, 30, 20, 40, 30, 50, 60].
        If expand_idx[i] < 0, then obj_expanded[i, :] = 0.
        Also inverts the sign of the elements whose index is filled in self.sign_to_oppose.
        """
        obj_expanded = obj[self.expand_idx, :]
        obj_expanded[[idx for idx, val in enumerate(self.expand_idx) if val < 0], :] = 0

        if self.sign_to_oppose != ():
            obj_expanded[self.sign_to_oppose, :] *= -1
        return obj_expanded

    def reduce(self, obj):
        """
        Copies the elements of the input obj into an obj_reduced representing the reduced list according to the mapping.
        Example :
        - L is an array or a casADi MX containing the elements [10, 20, 30, 20, 30, 40, 50, 60].
        - reduce_idx is a list equal to [0, 1, 2, 5, 6, 7]
        reduce(L) will return obj_reduced, an object of the same type as L with the elements : [10, 20, 30, 40, 50, 60]
        """
        obj_reduced = obj
        if self.sign_to_oppose != ():
            obj_reduced[self.sign_to_oppose, :] *= -1
        obj_reduced = obj_reduced[self.reduce_idx, :]
        return obj_reduced
