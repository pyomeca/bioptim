from casadi import MX, Function

from ...misc.parameters_types import Int, StrTuple


class EuclideanConfigurationSpace:
    """
    Default description of the configuration space for models whose coordinates are Euclidean (#1037).

    It is valid when nq == nv, that is when q and qdot have the same size and q lives in R^n (no quaternion, no SO(2)
    coordinate). The number of coordinates is deduced from `name_dofs`, the tangent space is identical to the
    configuration space and `normalize_q` is the identity.

    Backends whose configuration space is not Euclidean (quaternions, SO(2), ...) must override `name_q`, `name_qdot`
    and `normalize_q` (and `nb_q`/`nb_qdot`). This mixin has no `__init__` so it is safe in cooperative inheritance.
    """

    @property
    def name_q(self) -> StrTuple:
        return tuple(self.name_dofs)

    @property
    def nb_q(self) -> Int:
        return len(self.name_q)

    @property
    def nb_qdot(self) -> Int:
        # Euclidean assumption, deliberately not defined from name_qdot to avoid infinite recursion
        return len(self.name_q)

    @property
    def name_qdot(self) -> StrTuple:
        if self.nb_q != self.nb_qdot or len(self.name_q) != self.nb_q:
            raise NotImplementedError(
                f"{type(self).__name__} has a non-Euclidean configuration space "
                f"(nb_q={self.nb_q}, nb_qdot={self.nb_qdot}): it must implement name_q and name_qdot"
            )
        return self.name_q

    def normalize_q(self) -> Function:
        """Identity Function q -> q (Euclidean configuration space), usable with MX and SX."""
        cached = self.__dict__.get("_euclidean_normalize_q")
        if cached is None:
            q = MX.sym("q", self.nb_q, 1)
            cached = Function("normalize_q", [q], [q], ["q"], ["q_normalized"])
            self.__dict__["_euclidean_normalize_q"] = cached
        return cached
