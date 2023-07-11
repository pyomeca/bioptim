from typing import Any

import numpy as np

from .path_conditions import Bounds
from ..misc.enums import InterpolationType
from ..misc.fcn_enum import FcnEnum
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFunctions


class MultinodeConstraint(MultinodePenalty):
    def __init__(self, *args, min_bound: float = 0, max_bound: float = 0, **kwargs):
        if "weight" in kwargs and kwargs["weight"] is not None:
            raise ValueError(
                "MultinodeConstraints can't declare weight, use MultinodeObjective instead. If you were defining a "
                "custom function that uses 'weight' as parameter, please use another keyword."
            )

        super(MultinodeConstraint, self).__init__(MultinodeConstraintFcn, *args, **kwargs)

        self.weight = 0
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        super(MultinodeConstraint, self).add_or_replace_to_penalty_pool(ocp, nlp)
        self.set_bounds()

    def set_bounds(self):
        self.min_bound = np.array(self.min_bound) if isinstance(self.min_bound, (list, tuple)) else self.min_bound
        self.max_bound = np.array(self.max_bound) if isinstance(self.max_bound, (list, tuple)) else self.max_bound

        if self.bounds.shape[0] == 0:
            for i in self.rows:
                min_bound = (
                    self.min_bound[i]
                    if hasattr(self.min_bound, "__getitem__") and self.min_bound.shape[0] > 1
                    else self.min_bound
                )
                max_bound = (
                    self.max_bound[i]
                    if hasattr(self.max_bound, "__getitem__") and self.max_bound.shape[0] > 1
                    else self.max_bound
                )
                self.bounds.concatenate(Bounds(None, min_bound, max_bound, interpolation=InterpolationType.CONSTANT))
        elif self.bounds.shape[0] != len(self.rows):
            raise RuntimeError(f"bounds rows is {self.bounds.shape[0]} but should be {self.rows} or empty")

    def _get_pool_to_add_penalty(self, ocp, nlp):
        return nlp.g_internal if nlp else ocp.g_internal


class MultinodeConstraintList(MultinodePenaltyList):
    """
    A list of Multinode Constraint

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new MultinodePenalty to the list
    print(self)
        Print the MultinodeConstraintList to the console
    prepare_multinode_penalties(self, ocp) -> list
        Configure all the multinode penalties and put them in a list
    """

    def add(self, multinode_constraint: Any, **extra_arguments: Any):
        """
        Add a new MultinodePenalty to the list

        Parameters
        ----------
        multinode_constraint: Callable | MultinodeConstraintFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        super(MultinodeConstraintList, self).add(
            option_type=MultinodeConstraint,
            multinode_penalty=multinode_constraint,
            _multinode_penalty_fcn=MultinodeConstraintFcn,
            **extra_arguments,
        )


class MultinodeConstraintFunctions(MultinodePenaltyFunctions):
    pass


class MultinodeConstraintFcn(FcnEnum):
    """
    Selection of valid multinode constraint functions
    """

    STATES_EQUALITY = (MultinodeConstraintFunctions.Functions.states_equality,)
    CONTROLS_EQUALITY = (MultinodeConstraintFunctions.Functions.controls_equality,)
    CUSTOM = (MultinodeConstraintFunctions.Functions.custom,)
    COM_EQUALITY = (MultinodeConstraintFunctions.Functions.com_equality,)
    COM_VELOCITY_EQUALITY = (MultinodeConstraintFunctions.Functions.com_velocity_equality,)
    TIME_CONSTRAINT = (MultinodeConstraintFunctions.Functions.time_equality,)
    M_EQUALS_INVERSE_OF_DG_DZ = (MultinodeConstraintFunctions.Functions.m_equals_inverse_of_dg_dz,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodeConstraintFunctions
