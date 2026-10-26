"""
Euclidean defaults for the description of the configuration space of a model (#1037).

A model lives in a configuration space Q (`nb_q` coordinates, labelled by `name_q`) whose tangent space TQ has
`nb_qdot` coordinates (labelled by `name_qdot`). This module provides `EuclideanConfigurationSpace`, a mixin giving
the members of the `BioModel` protocol that describe Q and TQ (`name_q`, `nb_q`, `nb_qdot`, `name_qdot` and
`normalize_q`) for models whose coordinates are Euclidean. It is inherited by `StateDynamics`, so that custom models
get these defaults without any extra code.
"""

from casadi import MX, Function

from ...misc.parameters_types import Int, StrTuple


class EuclideanConfigurationSpace:
    """
    Default description of the configuration space for models whose coordinates are Euclidean (#1037).

    It is valid when nq == nv, that is when q and qdot have the same size and q lives in R^n (no quaternion, no SO(2)
    coordinate). The number of coordinates is deduced from `name_dofs`, the tangent space is identical to the
    configuration space and `normalize_q` is the identity.

    Notes
    -----
    Backends whose configuration space is not Euclidean (quaternions, SO(2), ...) must override `name_q`, `name_qdot`,
    `normalize_q`, `nb_q` and `nb_qdot`. In a usable model the backend comes first in the method resolution order
    (e.g. `TorqueBiorbdModel(BiorbdModel, TorqueDynamics)`), so the members of the backend win over these defaults.
    This mixin has no `__init__` so it is safe in cooperative inheritance.
    """

    @property
    def name_q(self) -> StrTuple:
        """
        Get the label of each configuration coordinate.

        The default bridges to `name_dofs` and is kept until `name_dofs` is deprecated.

        Returns
        -------
        StrTuple
            One label per coordinate of q, that is `nb_q` labels (`tuple(name_dofs)` by default).
        """
        return tuple(self.name_dofs)

    @property
    def nb_q(self) -> Int:
        """
        Get the number of configuration coordinates.

        Returns
        -------
        Int
            The size of q, that is `len(name_q)`.
        """
        return len(self.name_q)

    @property
    def nb_qdot(self) -> Int:
        """
        Get the number of coordinates of the tangent space.

        It is `len(name_q)` under the Euclidean assumption. It is deliberately not derived from `name_qdot`, which
        itself reads `nb_qdot`, to avoid an infinite recursion.

        Returns
        -------
        Int
            The size of qdot (and of qddot, tau, ...), that is `len(name_q)`.
        """
        # Euclidean assumption, deliberately not defined from name_qdot to avoid infinite recursion
        return len(self.name_q)

    @property
    def name_qdot(self) -> StrTuple:
        """
        Get the label of each coordinate of the tangent space.

        These are also the labels of qddot and qdddot, and of the generalized forces tau, taudot and residual_tau.
        For a Euclidean space they are equal to `name_q`.

        Returns
        -------
        StrTuple
            One label per coordinate of qdot, that is `nb_qdot` labels (`name_q` by default).

        Raises
        ------
        NotImplementedError
            If `nb_q != nb_qdot` or `len(name_q) != nb_q`, i.e. the space is not Euclidean and the model must
            implement `name_q` and `name_qdot`.
        """
        if self.nb_q != self.nb_qdot or len(self.name_q) != self.nb_q:
            raise NotImplementedError(
                f"{type(self).__name__} has a non-Euclidean configuration space "
                f"(nb_q={self.nb_q}, nb_qdot={self.nb_qdot}): it must implement name_q and name_qdot"
            )
        return self.name_q

    def normalize_q(self) -> Function:
        """
        Get the projection of q onto the configuration space, which is the identity for a Euclidean space.

        The Function is built once per instance and stored in the instance `__dict__`. `cache_function` is not used
        because custom models do not have `_cached_functions`.

        Returns
        -------
        Function
            The identity Function q -> q (input "q", output "q_normalized") of size `nb_q`, usable with MX and SX.
        """
        cached = self.__dict__.get("_euclidean_normalize_q")
        if cached is None:
            q = MX.sym("q", self.nb_q, 1)
            cached = Function("normalize_q", [q], [q], ["q"], ["q_normalized"])
            self.__dict__["_euclidean_normalize_q"] = cached
        return cached
