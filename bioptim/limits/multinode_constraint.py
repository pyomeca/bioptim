from typing import Any

import numpy as np

from .path_conditions import Bounds
from ..misc.enums import InterpolationType, PenaltyType, ConstraintType
from ..misc.fcn_enum import FcnEnum
from .multinode_penalty import MultinodePenalty, MultinodePenaltyList, MultinodePenaltyFunctions

from ..misc.parameters_types import (
    Bool,
    Float,
)


class MultinodeConstraint(MultinodePenalty):
    def __init__(self, *args, min_bound: Float = 0, max_bound: Float = 0, is_stochastic: Bool = False, **kwargs):
        if "weight" in kwargs and kwargs["weight"] is not None:
            raise ValueError(
                "MultinodeConstraints can't declare weight, use MultinodeObjective instead. If you were defining a "
                "custom function that uses 'weight' as parameter, please use another keyword."
            )

        super(MultinodeConstraint, self).__init__(MultinodeConstraintFcn, *args, **kwargs)

        self.weight = 0
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.is_stochastic = is_stochastic
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

        if self.penalty_type == PenaltyType.INTERNAL:
            pool = nlp.g_internal if nlp else ocp.g_internal
        elif self.penalty_type == ConstraintType.IMPLICIT:
            pool = nlp.g_implicit if nlp else ocp.g_implicit
        elif self.penalty_type == PenaltyType.USER:
            pool = nlp.g if nlp else ocp.g
        else:
            raise ValueError(f"Invalid constraint type {self.penalty_type}.")

        return pool


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
    ALGEBRAIC_STATES_EQUALITY = (MultinodeConstraintFunctions.Functions.algebraic_states_equality,)
    ALGEBRAIC_STATES_CONTINUITY = (MultinodeConstraintFunctions.Functions.algebraic_states_continuity,)
    CUSTOM = (MultinodeConstraintFunctions.Functions.custom,)
    COM_EQUALITY = (MultinodeConstraintFunctions.Functions.com_equality,)
    COM_VELOCITY_EQUALITY = (MultinodeConstraintFunctions.Functions.com_velocity_equality,)
    TIME_CONSTRAINT = (MultinodeConstraintFunctions.Functions.time_equality,)
    TRACK_TOTAL_TIME = (MultinodeConstraintFunctions.Functions.track_total_time,)
    STOCHASTIC_HELPER_MATRIX_EXPLICIT = (MultinodeConstraintFunctions.Functions.stochastic_helper_matrix_explicit,)
    STOCHASTIC_HELPER_MATRIX_IMPLICIT = (MultinodeConstraintFunctions.Functions.stochastic_helper_matrix_implicit,)
    STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_IMPLICIT = (
        MultinodeConstraintFunctions.Functions.stochastic_covariance_matrix_continuity_implicit,
    )
    STOCHASTIC_DF_DW_IMPLICIT = (MultinodeConstraintFunctions.Functions.stochastic_df_dw_implicit,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodeConstraintFunctions
