from typing import Callable, Any
from warnings import warn

from casadi import vertcat, MX

from .multinode_constraint import MultinodeConstraint
from .multinode_penalty import MultinodePenalty, MultinodePenaltyFunctions
from .path_conditions import Bounds
from .weight import Weight
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyController
from ..misc.enums import Node, PenaltyType, InterpolationType
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping
from ..misc.options import UniquePerPhaseOptionList


from ..misc.parameters_types import (
    IntOptional,
    Float,
    FloatOptional,
)


class PhaseTransition(MultinodePenalty):
    """
    A placeholder for a transition of state

    Attributes
    ----------
    min_bound: list
        The minimal bound of the phase transition
    max_bound: list
        The maximal bound of the phase transition
    bounds: Bounds
        The bounds (will be filled with min_bound/max_bound)
    weight: float
        The weight of the cost function
    quadratic: bool
        If the objective function is quadratic
    node: Node
        The kind of node
    dt: float
        The delta time
    node_idx: int
        The index of the node in nlp pre
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    """

    def __init__(
        self,
        phase_pre_idx: IntOptional = None,
        transition: Any | Callable = None,
        weight: FloatOptional = None,
        custom_function: Callable = None,
        min_bound: Float = 0,
        max_bound: Float = 0,
        **extra_parameters: Any,
    ):
        if not isinstance(transition, PhaseTransitionFcn):
            custom_function = transition
            transition = PhaseTransitionFcn.CUSTOM
        super(PhaseTransition, self).__init__(
            PhaseTransitionFcn,
            nodes_phase=(
                (-1, 0)
                if transition in [transition.CYCLIC, transition.COVARIANCE_CYCLIC]
                else (phase_pre_idx, phase_pre_idx + 1)
            ),
            nodes=(Node.END, Node.START),
            multinode_penalty=transition,
            custom_function=custom_function,
            **extra_parameters,
        )

        if isinstance(weight, Weight):
            self.weight = weight
        elif weight is not None:
            self.weight = Weight(weight)
        else:
            self.weight = Weight(0)
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds("phase_transition", interpolation=InterpolationType.CONSTANT)
        self.node = Node.TRANSITION
        self.quadratic = True
        self.is_transition = True

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        super(PhaseTransition, self).add_or_replace_to_penalty_pool(ocp, nlp)
        if not self.weight:
            self: MultinodeConstraint
            MultinodeConstraint.set_bounds(self)

    def _get_pool_to_add_penalty(self, ocp, nlp):
        if not self.weight:
            return nlp.g_internal if nlp else ocp.g_internal
        else:
            return nlp.J_internal if nlp else ocp.J_internal


class PhaseTransitionList(UniquePerPhaseOptionList):
    """
    A list of PhaseTransition

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new PhaseTransition to the list
    print(self)
        Print the PhaseTransitionList to the console
    prepare_phase_transitions(self, ocp) -> list
        Configure all the phase transitions and put them in a list
    """

    def add(self, transition: Any, **extra_arguments: Any):
        """
        Add a new PhaseTransition to the list

        Parameters
        ----------
        transition: Callable | PhaseTransitionFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(transition, PhaseTransitionFcn):
            extra_arguments["custom_function"] = transition
            transition = PhaseTransitionFcn.CUSTOM
        super(PhaseTransitionList, self)._add(
            option_type=PhaseTransition, transition=transition, phase=-1, **extra_arguments
        )

    def print(self):
        """
        Print the PhaseTransitionList to the console
        """
        raise NotImplementedError("Printing of PhaseTransitionList is not ready yet")


class PhaseTransitionFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """

    class Functions:
        """
        Implementation of all the phase transitions
        """

        @staticmethod
        def continuous(
            transition,
            controllers: list[PenaltyController, PenaltyController],
            states_mapping: list[BiMapping] = None,
        ):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements
            states_mapping: list
                A list of the mapping for the states between nodes. It should provide a mapping between 0 and i, where
                the first (0) link the controllers[0].state to a number of values using to_second. Thereafter, the
                to_first is used sequentially for all the controllers (meaning controllers[1] uses the
                states_mapping[0].to_first. Therefore, the dimension of the states_mapping
                should be 'len(controllers) - 1'

            Returns
            -------
            The difference between the state after and before
            """

            return MultinodePenaltyFunctions.Functions.states_equality(
                transition, controllers, "all", states_mapping=states_mapping
            )

        @staticmethod
        def continuous_controls(
            transition,
            controllers: list[PenaltyController, PenaltyController],
            controls_mapping: list[BiMapping] = None,
        ):
            """
            This continuity function is only relevant for ControlType.LINEAR_CONTINUOUS otherwise don't use it.

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements
            controls_mapping: list
                A list of the mapping for the states between nodes. It should provide a mapping between 0 and i, where
                the first (0) link the controllers[0].controls to a number of values using to_second. Thereafter, the
                to_first is used sequentially for all the controllers (meaning controllers[1] uses the
                controls_mapping[0].to_first. Therefore, the dimension of the states_mapping
                should be 'len(controllers) - 1'

            Returns
            -------
            The difference between the controls after and before
            """
            if controls_mapping is not None:
                raise NotImplementedError(
                    "Controls_mapping is not yet implemented "
                    "for continuous_controls with linear continuous control type."
                )

            return MultinodePenaltyFunctions.Functions.controls_equality(transition, controllers, "all")

        @staticmethod
        def discontinuous(transition, controllers: list[PenaltyController, PenaltyController]):
            """
            There is no continuity constraints on the states

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            return controllers.cx.zeros(0, 0)

        @staticmethod
        def cyclic(transition, controllers: list[PenaltyController, PenaltyController]) -> MX:
            """
            The continuity function applied to the last to first node

            Parameters
            ----------
            transition: PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node
            """

            return MultinodePenaltyFunctions.Functions.states_equality(transition, controllers, "all")

        @staticmethod
        def impact(transition, controllers: list[PenaltyController, PenaltyController]):
            """
            A discontinuous function that simulates an inelastic impact of a new contact point

            Parameters
            ----------
            transition: PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the last and first node after applying the impulse equations
            """

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(transition, controllers)

            ocp = controllers[0].ocp
            if ocp.nlp[transition.nodes_phase[0]].states.shape != ocp.nlp[transition.nodes_phase[1]].states.shape:
                raise RuntimeError(
                    "Impact transition without same nx is not possible, please provide a custom phase transition"
                )

            # Aliases
            pre, post = controllers
            if post.model.nb_rigid_contacts == 0:
                warn("The chosen model does not have any rigid contact")

            # Todo scaled?
            q_pre = pre.states["q"].cx
            qdot_pre = pre.states["qdot"].cx
            qdot_impact = post.model.qdot_from_impact()(q_pre, qdot_pre, pre.parameters.cx)

            val = []
            cx_start = []
            cx_end = []
            for key in pre.states:
                cx_end = vertcat(cx_end, pre.states[key].mapping.to_second.map(pre.states[key].cx))
                cx_start = vertcat(cx_start, post.states[key].mapping.to_second.map(post.states[key].cx))
                post_cx = post.states[key].cx
                continuity = post.states["qdot"].mapping.to_first.map(
                    qdot_impact - post_cx if key == "qdot" else pre.states[key].cx - post_cx
                )
                val = vertcat(val, continuity)

            return val

        @staticmethod
        def covariance_cyclic(
            transition,
            controllers: list[PenaltyController, PenaltyController],
        ):
            """
            The most common continuity function, that is the covariance before equals covariance after
            for stochastic ocp

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the covariance after and before
            """

            return MultinodePenaltyFunctions.Functions.controls_equality(transition, controllers, "cov")

        @staticmethod
        def covariance_continuous(
            transition,
            controllers: list[PenaltyController, PenaltyController],
        ):
            """
            The most common continuity function, that is the covariance before equals covariance after
            for stochastic ocp

            Parameters
            ----------
            transition : PhaseTransition
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the covariance after and before
            """

            return MultinodePenaltyFunctions.Functions.algebraic_states_equality(transition, controllers, "cov")


class PhaseTransitionFcn(FcnEnum):
    """
    Selection of valid phase transition functions
    """

    CONTINUOUS = (PhaseTransitionFunctions.Functions.continuous,)
    CONTINUOUS_CONTROLS = (PhaseTransitionFunctions.Functions.continuous_controls,)
    DISCONTINUOUS = (PhaseTransitionFunctions.Functions.discontinuous,)
    IMPACT = (PhaseTransitionFunctions.Functions.impact,)
    CYCLIC = (PhaseTransitionFunctions.Functions.cyclic,)
    COVARIANCE_CYCLIC = (PhaseTransitionFunctions.Functions.covariance_cyclic,)
    COVARIANCE_CONTINUOUS = (PhaseTransitionFunctions.Functions.covariance_continuous,)
    CUSTOM = (MultinodePenaltyFunctions.Functions.custom,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return PhaseTransitionFunctions
