class Mapping:
    def __init__(self, expand_idx, reduce_idx, sign_to_oppose_for_expanded=()):
        """

        """
        self.expand_idx = expand_idx
        self.nb_expanded = len(self.expand_idx)
        self.reduce_idx = reduce_idx
        self.nb_reduced = len(self.reduce_idx)
        self.sign_to_oppose = sign_to_oppose_for_expanded

    def expand(self, obj):
        """
        Docstring à compléter, récupère des variables symétrisées qu'elle signe et renvoie non sym
        """

        obj_expanded = obj[self.expand_idx]
        if self.sign_to_oppose is not ():
            obj_expanded[self.sign_to_oppose] *= -1
        return obj_expanded

    def reduce(self, obj):
        """
        Docstring à compléter, récupère des variables non symétrisées qu'elle signe et renvoie sym
        """
        obj_reduced = obj
        obj_reduced[self.sign_to_oppose] *= -1
        obj_reduced = obj_reduced[self.reduce_idx]
        return obj_reduced
