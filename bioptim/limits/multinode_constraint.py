from typing import Callable, Any

from .constraints import Constraint
from .path_conditions import Bounds
from .objective_functions import ObjectiveFunction
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyController
from ..misc.enums import Node, InterpolationType, PenaltyType
from ..misc.fcn_enum import FcnEnum
from ..misc.options import UniquePerPhaseOptionList
from ..misc.mapping import BiMapping


class MultinodeConstraint(Constraint):
    """
    A placeholder for a multinode constraints

    Attributes
    ----------
    min_bound: list
        The minimal bound of the multinode constraints
    max_bound: list
        The maximal bound of the multinode constraints
    bounds: Bounds
        The bounds (will be filled with min_bound/max_bound)
    weight: float
        The weight of the cost function
    quadratic: bool
        If the objective function is quadratic
    nodes_phase: tuple[int, ...]
        The index of the phase for the corresponding node in nodes
    nodes: tuple[int | Node, ...]
        The nodes on which the constraint will be computed on
    dt: float
        The delta time
    node_idx: int
        The index of the node in nlp pre
    multinode_constraint: Callable | Any
        The nature of the cost function is the bi node constraint
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    """

    def __init__(
        self,
        nodes: tuple[int | Node, ...],
        nodes_phase: tuple[int, ...],
        multinode_constraint: Any | Callable = None,
        custom_function: Callable = None,
        min_bound: float = 0,
        max_bound: float = 0,
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

        force_multinode = False
        if "force_multinode" in params:
            # This is a hack to circumvent the apparatus that moves the functions to a custom function
            # It is necessary for PhaseTransition
            force_multinode = True
            del params["force_multinode"]

        if not isinstance(multinode_constraint, MultinodeConstraintFcn) and not force_multinode:
            custom_function = multinode_constraint
            multinode_constraint = MultinodeConstraintFcn.CUSTOM
        super(Constraint, self).__init__(penalty=multinode_constraint, custom_function=custom_function, **params)

        for node in nodes:
            if node not in (Node.START, Node.MID, Node.PENULTIMATE, Node.END):
                if not isinstance(node, int):
                    raise ValueError(
                        "Multinode constraint only works with Node.START, Node.MID, "
                        "Node.PENULTIMATE, Node.END or a node index (int)."
                    )
        for phase in nodes_phase:
            if not isinstance(phase, int):
                raise ValueError("nodes_phase should be all positive integers corresponding to the phase index")

        if len(nodes) != len(nodes_phase):
            raise ValueError("Each of the nodes must have a corresponding nodes_phase")

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(interpolation=InterpolationType.CONSTANT)

        self.multinode_constraint = True
        self.weight = weight if weight is not None else 0

        self.quadratic = True
        self.nodes_phase = nodes_phase
        self.nodes = nodes
        self.node = Node.MULTINODES
        self.dt = 1
        self.node_idx = [0]
        self.all_nodes_index = []  # This is filled when nodes are collapsed as actual time indices
        self.penalty_type = PenaltyType.INTERNAL

    def _add_penalty_to_pool(self, controller: list[PenaltyController, PenaltyController]):
        if not isinstance(controller, (list, tuple)):
            raise RuntimeError(
                "_add_penalty for multi constraints function was called without a list while it should not"
            )

        ocp = controller[0].ocp
        nlp = controller[0].get_nlp
        if self.weight:
            pool = nlp.J_internal if nlp else ocp.J_internal
        else:
            pool = nlp.g_internal if nlp else ocp.g_internal
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        if self.weight:
            g_to_add_to = nlp.J_internal if nlp else ocp.J_internal
        else:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal

        if self.list_index < 0:
            for i, j in enumerate(g_to_add_to):
                if not j:
                    self.list_index = i
                    return
            else:
                g_to_add_to.append([])
                self.list_index = len(g_to_add_to) - 1
        else:
            while self.list_index >= len(g_to_add_to):
                g_to_add_to.append([])
            g_to_add_to[self.list_index] = []


class MultinodeConstraintList(UniquePerPhaseOptionList):
    """
    A list of Bi Node Constraint

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new MultinodeConstraint to the list
    print(self)
        Print the MultinodeConstraintList to the console
    prepare_multinode_constraint(self, ocp) -> list
        Configure all the multinode_constraint and put them in a list
    """

    def add(self, multinode_constraint: Any, **extra_arguments: Any):
        """
        Add a new MultinodeConstraint to the list

        Parameters
        ----------
        multinode_constraint: Callable | MultinodeConstraintFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(multinode_constraint, MultinodeConstraintFcn):
            extra_arguments["custom_function"] = multinode_constraint
            multinode_constraint = MultinodeConstraintFcn.CUSTOM
        super(MultinodeConstraintList, self)._add(
            option_type=MultinodeConstraint, multinode_constraint=multinode_constraint, phase=-1, **extra_arguments
        )

    def print(self):
        """
        Print the MultinodeConstraintList to the console
        """
        raise NotImplementedError("Printing of MultinodeConstraintList is not ready yet")

    def prepare_multinode_constraints(self, ocp) -> list:
        """
        Configure all the phase transitions and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of all the bi_node constraints prepared
        """
        full_phase_multinode_constraint = []
        for mnc in self:
            for phase in mnc.nodes_phase:
                if phase < 0 or phase >= ocp.n_phases:
                    raise ValueError("nodes_phase of the multinode_constraint must be between 0 and number of phases")

            if mnc.weight:
                mnc.base = ObjectiveFunction.MayerFunction

            full_phase_multinode_constraint.append(mnc)

        return full_phase_multinode_constraint


class MultinodeConstraintFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """

    class Functions:
        """
        Implementation of all the Multinode Constraint
        """

        @staticmethod
        def states_equality(
            constraint,
            controllers: list[PenaltyController, ...],
            key: str = "all",
            states_mapping: list[BiMapping, ...] = None,
        ):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            constraint : MultinodeConstraint
                A reference to the phase transition
            controllers: list
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

            MultinodeConstraintFunctions.Functions._prepare_controller_cx(controllers)
            states_mapping = MultinodeConstraintFunctions.Functions._prepare_states_mapping(controllers, states_mapping)

            ctrl_0 = controllers[0]
            states_0 = states_mapping[0].to_second.map(ctrl_0.states[key].cx)
            out = ctrl_0.cx.zeros(states_0.shape)
            for i in range(1, len(controllers)):
                ctrl_i = controllers[i]
                states_i = states_mapping[i - 1].to_first.map(ctrl_i.states[key].cx)

                if states_0.shape != states_i.shape:
                    raise RuntimeError(
                        f"Continuity can't be established since the number of x to be matched is {states_0.shape} in "
                        f"the pre-transition phase and {states_i.shape} post-transition phase. Please use a custom "
                        f"transition or supply states_mapping"
                    )

                out += states_0 - states_i

            return out

        @staticmethod
        def controls_equality(constraint, controllers: list[PenaltyController, ...], key: str = "all"):
            """
            The controls before equals controls after

            Parameters
            ----------
            constraint : MultinodeConstraint
                A reference to the phase transition
            controllers: list[PenaltyController, ...]
                    The penalty node elements

            Returns
            -------
            The difference between the controls after and before
            """

            MultinodeConstraintFunctions.Functions._prepare_controller_cx(controllers)

            ctrl_0 = controllers[0]
            controls_0 = ctrl_0.controls[key].cx
            out = ctrl_0.cx.zeros(controls_0.shape)
            for i in range(1, len(controllers)):
                ctrl_i = controllers[i]
                controls_i = ctrl_i.controls[key].cx

                if controls_0.shape != controls_i.shape:
                    raise RuntimeError(
                        f"Continuity can't be established since the number of x to be matched is {controls_0.shape} in "
                        f"the pre-transition phase and {controls_i.shape} post phase. Please use a custom "
                        f"multi_node"
                    )

                out += controls_0 - controls_i

            return out

        @staticmethod
        def com_equality(constraint, controllers: list[PenaltyController, ...]):
            """
            The centers of mass are equals for the specified phases and the specified nodes

            Parameters
            ----------
            constraint : MultinodeConstraint
                A reference to the phase transition
            controllers: list[PenaltyController, ...]
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            MultinodeConstraintFunctions.Functions._prepare_controller_cx(controllers)

            com_0 = controllers[0].model.center_of_mass(controllers[0].states["q"].cx)

            out = controllers[0].cx.zeros((3, 1))
            for i in range(1, len(controllers)):
                com_i = controllers[i].model.center_of_mass(controllers[i].states["q"].cx)
                out += com_0 - com_i

            return out

        @staticmethod
        def com_velocity_equality(constraint, controllers: list[PenaltyController, ...]):
            """
            The centers of mass velocity are equals for the specified phases and the specified nodes

            Parameters
            ----------
            constraint : MultinodeConstraint
                A reference to the phase transition
            controllers: list[PenaltyController, ...]
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            MultinodeConstraintFunctions.Functions._prepare_controller_cx(controllers)

            com_dot_0 = controllers[0].model.center_of_mass_velocity(
                controllers[0].states["q"].cx, controllers[0].states["qdot"].cx
            )

            out = controllers[0].cx.zeros((3, 1))
            for i in range(1, len(controllers)):
                com_dot_i = controllers[i].model.center_of_mass_velocity(
                    controllers[i].states["q"].cx, controllers[i].states["qdot"].cx
                )
                out += com_dot_0 - com_dot_i

            return out

        @staticmethod
        def time_equality(constraint, controllers: list[PenaltyController, PenaltyController]):
            """
            The duration of one phase must be the same as the duration of another phase

            Parameters
            ----------
            constraint : MultinodeConstraint
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the duration of the phases
            """

            MultinodeConstraintFunctions.Functions._prepare_controller_cx(controllers)

            def get_time_parameter_idx(controller: PenaltyController, i_phase):
                time_idx = None
                for i in range(controller.parameters.cx.shape[0]):
                    param_name = controller.parameters.cx[i].name()
                    if param_name == "time_phase_" + str(controller.phase_idx):
                        time_idx = controller.phase_idx
                if time_idx is None:
                    raise RuntimeError(
                        f"Time constraint can't be established since the {i_phase}th phase has no time parameter. "
                        f"\nTime parameter can be added with : "
                        f"\nobjective_functions.add(ObjectiveFcn.[Mayer or Lagrange].MINIMIZE_TIME) or "
                        f"\nwith constraints.add(ConstraintFcn.TIME_CONSTRAINT)."
                    )
                return time_idx

            time_idx = [get_time_parameter_idx(controller, i) for i, controller in enumerate(controllers)]

            time_0 = controllers[0].parameters.cx[time_idx[0]]
            out = controllers[0].cx.zeros((1, 1))
            for i in range(1, len(controllers)):
                time_i = controllers[i].parameters.cx[time_idx[i]]
                out += time_0 - time_i

            return out

        @staticmethod
        def custom(constraint, controllers: list[PenaltyController, PenaltyController], **extra_params):
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            constraint: MultinodeConstraint
                A reference to the phase transition
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            MultinodeConstraintFunctions.Functions._prepare_controller_cx(controllers)
            return constraint.custom_function(controllers, **extra_params)

        @staticmethod
        def _prepare_controller_cx(controllers: list[PenaltyController, ...]):
            """
            Prepare the current_cx_to_get for each of the controller. Basically it finds if this constraint as more than
            one usage. If it does, it increments a counter of the cx used, up to the maximum. On assume_phase_dynamics
            being False, this is useless, as all the constraints uses cx_start.
            """
            existing_phases = []
            for controller in controllers:
                controller.cx_index_to_get = sum([i == controller.phase_idx for i in existing_phases])
                existing_phases.append(controller.phase_idx)

        @staticmethod
        def _prepare_states_mapping(controllers: list[PenaltyController, ...], states_mapping: list[BiMapping, ...]):
            """
            Prepare a new state_mappings if None is sent. Otherwise, it simply returns the current states_mapping

            Parameters
            ----------
            controllers: list
                The penalty node elements
            states_mapping: list
                A list of the mapping for the states between nodes. It should provide a mapping between 0 and i, where
                the first (0) link the controllers[0].state to a number of values using to_second. Thereafter, the
                to_first is used sequentially for all the controllers (meaning controllers[1] uses the
                states_mapping[0].to_first. Therefore, the dimension of the states_mapping
                should be 'len(controllers) - 1'
            """

            if states_mapping is None:
                states_mapping = []
                for controller in controllers:
                    states_mapping.append(BiMapping(range(controller.states.shape), range(controller.states.shape)))
            else:
                if not isinstance(states_mapping, (list, tuple)) and len(controllers) == 2:
                    states_mapping = [states_mapping]

            return states_mapping


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

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodeConstraintFunctions
