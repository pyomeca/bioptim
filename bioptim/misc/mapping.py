class BidirectionalMapping:
    def __init__(self, expand_mapping, reducing_mapping):
        if not isinstance(expand_mapping, Mapping):
            raise RuntimeError("expand_mapping must be a Mapping class")
        if not isinstance(reducing_mapping, Mapping):
            raise RuntimeError("reducing_mapping must be a Mapping class")

        self.expand = expand_mapping
        self.reduce = reducing_mapping


class Mapping:
    def __init__(self, map_idx, sign_to_oppose=()):
        """
        Creates a mapping to define the correspondence between two data sets.
        Well suited to map the size of problems handled by the solver when they have symmetries.
        """
        self.map_idx = map_idx
        self.len = len(self.map_idx)
        self.sign_to_oppose = sign_to_oppose

    def map(self, obj):
        """
        Copies the elements of the input obj into a mapped_obj representing the map list according to the mapping.
        If map_idx[i] < 0, then mapped_obj[i, :] = 0.
        Also opposes the sign of the elements whose index is filled in self.sign_to_oppose.
        :param obj: A vector numpy array or CasADi MX of generalized coordinated, velocity, acceleration or torque

        Example of use:
            - to_map = Mapping([0, 1, 1, 3, -1, 1], [3])
            - obj = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            - mapped_obj = to_map.map(obj)
        Expected result:
            - mapped_obj == np.array([0.1, 0.2, 0.2, -0.4, 0, 0.1])
        """
        mapped_obj = obj[self.map_idx, :]
        mapped_obj[[idx for idx, val in enumerate(self.map_idx) if val < 0], :] = 0
        if self.sign_to_oppose != ():
            mapped_obj[self.sign_to_oppose, :] *= -1

        return mapped_obj
