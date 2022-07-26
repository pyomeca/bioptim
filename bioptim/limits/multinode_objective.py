from typing import Callable, Union, Any

from ..misc.fcn_enum import FcnEnum, Fcn
from ..misc.enums import Node
from ..misc.options import UniquePerPhaseOptionList
from .objective_functions import Objective, ObjectiveFcn, ObjectiveFunction
from .penalty_node import PenaltyNodeList
from .multinode_penalty import MultinodePenalty, MultinodePenaltyFunctions


# TODO: mirror multinode_constraint.py in here but for objectives
class MultinodeObjective(Objective, MultinodePenalty):
    """
    TODO: docstring
    """

    def __init__(
        self,
        phase_first_idx: int,
        phase_second_idx: int,
        first_node: Union[Node, int],
        second_node: Union[Node, int],
        multinode_objective: Union[Callable, Any] = None,
        custom_function: Callable = None,
        weight: float = 0,
        **params: Any,
    ):
        """
        Parameters
        ----------
        phase_first_idx: int
            The first index of the phase of concern
        params:
            Generic parameters for options
        """
        if custom_function and not callable(custom_function):
            raise RuntimeError("custom_function must be callable")

        if isinstance(multinode_objective, Fcn):
            if MultinodeObjective not in multinode_objective.get_fcn_types():
                raise RuntimeError(f"multinode_objective of type '{type(multinode_objective)}' not allowed")
        else:
            custom_function = multinode_objective
            multinode_objective = MultinodeObjectiveFcn.CUSTOM

        super(MultinodeObjective, self).__init__(
            multinode_penalty=multinode_objective,
            objective=multinode_objective,
            custom_function=custom_function,
            custom_type=ObjectiveFcn.Mayer,
            weight=weight,
            **params,
        )
        MultinodePenalty.__init__(
            self,
            phase_first_idx=phase_first_idx,
            phase_second_idx=phase_second_idx,
            first_node=first_node,
            second_node=second_node,
            multinode_penalty=multinode_objective,  # TODO: might not be necessary to store here.
            **params,
        )

    def _add_penalty_to_pool(self, all_pn: Union[PenaltyNodeList, list, tuple]):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp

        pool = nlp.J_internal if nlp else ocp.J_internal
        pool[self.list_index] = self

    def clear_penalty(self, ocp, nlp):
        pool_to_add_to = nlp.J_internal if nlp else ocp.J_internal

        if self.list_index < 0:
            for i, j in enumerate(pool_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                pool_to_add_to.append([])
                self.list_index = len(pool_to_add_to) - 1
        else:
            while self.list_index >= len(pool_to_add_to):
                pool_to_add_to.append([])
            pool_to_add_to[self.list_index] = []


class MultinodeObjectiveList(UniquePerPhaseOptionList):
    """
    A list of Multi Node Constraint

    Methods
    -------
    add(self, transition: Union[Callable, PhaseTransitionFcn], phase: int = -1, **extra_arguments)
        Add a new MultinodePenalty to the list
    print(self)
        Print the MultinodePenaltyList to the console
    prepare_multinode_penalty(self, ocp) -> list
        Configure all the multinode_objective and put them in a list
    """

    def add(self, multinode_objective: Any, **extra_arguments: Any):
        """
        Add a new MultinodeObjective to the list

        Parameters
        ----------
        multinode_objective: Union[Callable, MultinodePenaltyFcn]
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Penalty
        """

        if isinstance(multinode_objective, MultinodeObjective):
            self.copy(multinode_objective)
        else:
            super(MultinodeObjectiveList, self)._add(
                option_type=MultinodeObjective, multinode_objective=multinode_objective, phase=-1, **extra_arguments
            )

    def prepare_multinode_objectives(self, ocp) -> list:
        """
        Configure all the phase transitions and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of all the multi_node objectives prepared
        """
        full_phase_multinode_objective = []
        for mnc in self:

            if mnc.phase_first_idx >= ocp.n_phases or mnc.phase_second_idx >= ocp.n_phases:
                raise RuntimeError("Phase index of the multinode_objective is higher than the number of phases")
            if mnc.phase_first_idx < 0 or mnc.phase_second_idx < 0:
                raise RuntimeError("Phase index of the multinode_objective need to be positive")

            mnc.base = ObjectiveFunction.MayerFunction

            full_phase_multinode_objective.append(mnc)

        return full_phase_multinode_objective


class MultinodeObjectiveFcn(FcnEnum):
    """
    Selection of valid multinode penalty functions
    """

    EQUALITY = Fcn(MultinodePenaltyFunctions.Functions.equality)
    CUSTOM = Fcn(MultinodePenaltyFunctions.Functions.custom)
    COM_EQUALITY = Fcn(MultinodePenaltyFunctions.Functions.com_equality)
    COM_VELOCITY_EQUALITY = Fcn(MultinodePenaltyFunctions.Functions.com_velocity_equality)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodePenaltyFunctions

    @staticmethod
    def get_fcn_types():
        """
        Returns the types of the enum
        """
        return (MultinodeObjectiveFcn,) + ObjectiveFcn.Mayer.get_fcn_types()
