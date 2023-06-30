from typing import Callable, Any
from casadi import DM, horzcat, MX_eye, jacobian, Function, MX

from .constraints import PenaltyOption
from .objective_functions import ObjectiveFunction
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyController
from ..misc.enums import Node, PenaltyType
from ..misc.fcn_enum import FcnEnum
from ..misc.options import UniquePerPhaseOptionList
from ..misc.mapping import BiMapping


class MultinodePenalty(PenaltyOption):
    """
    A placeholder for a multinode constraints

    Attributes
    ----------
    weight: float
        The weight of the cost function
    quadratic: bool
        If the objective function is quadratic
    nodes_phase: tuple[int, ...]
        The index of the phase for the corresponding node in nodes
    nodes: tuple[int | Node, ...]
        The nodes on which the penalty will be computed on
    dt: float
        The delta time
    node_idx: int
        The index of the node in nlp pre
    multinode_penalty: Callable | Any
        The nature of the cost function is the binode penalty
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    """

    def __init__(
        self,
        _multinode_penalty_fcn: Any | type,
        nodes: tuple[int | Node, ...],
        nodes_phase: tuple[int, ...],
        multinode_penalty: Any | Callable = None,
        custom_function: Callable = None,
        **params: Any,
    ):
        if not isinstance(multinode_penalty, _multinode_penalty_fcn):
            custom_function = multinode_penalty
            multinode_penalty = _multinode_penalty_fcn.CUSTOM

        super(MultinodePenalty, self).__init__(penalty=multinode_penalty, custom_function=custom_function, **params)

        for node in nodes:
            if node not in (Node.START, Node.MID, Node.PENULTIMATE, Node.END):
                if not isinstance(node, int):
                    raise ValueError(
                        "Multinode penalties only works with Node.START, Node.MID, "
                        "Node.PENULTIMATE, Node.END or a node index (int)."
                    )
        for phase in nodes_phase:
            if not isinstance(phase, int):
                raise ValueError("nodes_phase should be all positive integers corresponding to the phase index")

        if len(nodes) != len(nodes_phase):
            raise ValueError("Each of the nodes must have a corresponding nodes_phase")

        self.multinode_penalty = True

        self.nodes_phase = nodes_phase
        self.nodes = nodes
        self.node = Node.MULTINODES
        self.dt = 1
        self.node_idx = [0]
        self.all_nodes_index = []  # This is filled when nodes are collapsed as actual time indices
        self.penalty_type = PenaltyType.INTERNAL

    def _get_pool_to_add_penalty(self, ocp, nlp):
        raise NotImplementedError("This is an abstract method and should be implemented by child")

    def _add_penalty_to_pool(self, controller: list[PenaltyController, PenaltyController]):
        if not isinstance(controller, (list, tuple)):
            raise RuntimeError(
                "_add_penalty for multinode_penalty function was called without a list while it should not"
            )

        ocp = controller[0].ocp
        nlp = controller[0].get_nlp
        pool = self._get_pool_to_add_penalty(ocp, nlp)
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        pool = self._get_pool_to_add_penalty(ocp, nlp)

        if self.list_index < 0:
            for i, j in enumerate(pool):
                if not j:
                    self.list_index = i
                    return
            else:
                pool.append([])
                self.list_index = len(pool) - 1
        else:
            while self.list_index >= len(pool):
                pool.append([])
            pool[self.list_index] = []


class MultinodePenaltyFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """

    class Functions:
        """
        Implementation of all the Multinode Constraint
        """

        @staticmethod
        def states_equality(
            penalty,
            controllers: list[PenaltyController, ...],
            key: str = "all",
            states_mapping: list[BiMapping, ...] = None,
        ):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the penalty
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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(controllers)
            states_mapping = MultinodePenaltyFunctions.Functions._prepare_states_mapping(controllers, states_mapping)

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
        def controls_equality(penalty, controllers: list[PenaltyController, ...], key: str = "all"):
            """
            The controls before equals controls after

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the penalty
            controllers: list[PenaltyController, ...]
                    The penalty node elements

            Returns
            -------
            The difference between the controls after and before
            """

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(controllers)

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
        def com_equality(penalty, controllers: list[PenaltyController, ...]):
            """
            The centers of mass are equals for the specified phases and the specified nodes

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the penalty
            controllers: list[PenaltyController, ...]
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(controllers)

            com_0 = controllers[0].model.center_of_mass(controllers[0].states["q"].cx)

            out = controllers[0].cx.zeros((3, 1))
            for i in range(1, len(controllers)):
                com_i = controllers[i].model.center_of_mass(controllers[i].states["q"].cx)
                out += com_0 - com_i

            return out

        @staticmethod
        def com_velocity_equality(penalty, controllers: list[PenaltyController, ...]):
            """
            The centers of mass velocity are equals for the specified phases and the specified nodes

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the penalty
            controllers: list[PenaltyController, ...]
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(controllers)

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
        def time_equality(penalty, controllers: list[PenaltyController, PenaltyController]):
            """
            The duration of one phase must be the same as the duration of another phase

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the phase penalty
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The difference between the duration of the phases
            """

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(controllers)

            def get_time_parameter_idx(controller: PenaltyController, i_phase):
                time_idx = None
                for i in range(controller.parameters.cx.shape[0]):
                    param_name = controller.parameters.cx[i].name()
                    if param_name == "time_phase_" + str(controller.phase_idx):
                        time_idx = controller.phase_idx
                if time_idx is None:
                    raise RuntimeError(
                        f"Time penalty can't be established since the {i_phase}th phase has no time parameter. "
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
        def m_equals_inverse_of_dg_dz(penalty, controllers: list[PenaltyController, PenaltyController], **unused_param):
            """
            ...
            """
            import numpy as np
            from ..examples.stochastic_optimal_control.arm_reaching_muscle_driven import stochastic_forward_dynamics

            if controllers[0].phase_idx != controllers[1].phase_idx:
                raise RuntimeError("For this constraint to make sens, the two nodes must belong to the same phase.")

            dt = controllers[0].tf / controllers[0].ns
            wM_std = 0.05
            wPq_std = 3e-4
            wPqdot_std = 0.0024
            wM_magnitude = DM(np.array([wM_std ** 2 / dt, wM_std ** 2 / dt]))
            wPq_magnitude = DM(np.array([wPq_std ** 2 / dt, wPq_std ** 2 / dt]))
            wPqdot_magnitude = DM(np.array([wPqdot_std ** 2 / dt, wPqdot_std ** 2 / dt]))

            wM = MX.sym("wM", 2, 1)
            wPq = MX.sym("wPq", 2, 1)
            wPqdot = MX.sym("wPqdot", 2, 1)

            nx = controllers[0].states.cx.shape[0]
            M_matrix = controllers[0].restore_matrix_from_vector(controllers[0].stochastic_variables, nx, nx, Node.START, "m")

            dx = stochastic_forward_dynamics(controllers[0].states.cx_start, controllers[0].controls.cx_start,
                                    controllers[0].parameters.cx_start, controllers[0].stochastic_variables.cx_start,
                                    controllers[0].get_nlp, wM, wPq, wPqdot,
                                    force_field_magnitude=unused_param["force_field_magnitude"], with_gains=True)

            DdZ_DX_fun = Function("DdZ_DX_fun", [controllers[0].states.cx_start,
                                                 controllers[0].controls.cx_start,
                                                 controllers[0].parameters.cx_start,
                                                 controllers[0].stochastic_variables.cx_start,
                                                 wM, wPq, wPqdot],
                                    [jacobian(dx.dxdt, controllers[0].states.cx_start)])

            DdZ_DX = DdZ_DX_fun(controllers[1].states.cx_start,
                                controllers[1].controls.cx_start,
                                controllers[1].parameters.cx_start,
                                controllers[1].stochastic_variables.cx_start,
                                wM_magnitude, wPq_magnitude, wPqdot_magnitude)

            DG_DZ = MX_eye(DdZ_DX.shape[0]) - DdZ_DX * dt / 2

            val = M_matrix @ DG_DZ - MX_eye(nx)

            out_vector = controllers[0].restore_vector_from_matrix(val)
            return out_vector

        @staticmethod
        def custom(penalty, controllers: list[PenaltyController, PenaltyController], **extra_params):
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            penalty: MultinodePenalty
                A reference to the penalty
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(controllers)
            return penalty.custom_function(controllers, **extra_params)

        @staticmethod
        def _prepare_controller_cx(controllers: list[PenaltyController, ...]):
            """
            Prepare the current_cx_to_get for each of the controller. Basically it finds if this penalty as more than
            one usage. If it does, it increments a counter of the cx used, up to the maximum. On assume_phase_dynamics
            being False, this is useless, as all the penalties uses cx_start.
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


class MultinodePenaltyFcn(FcnEnum):
    """
    Selection of valid multinode penalty functions
    """

    STATES_EQUALITY = (MultinodePenaltyFunctions.Functions.states_equality,)
    CONTROLS_EQUALITY = (MultinodePenaltyFunctions.Functions.controls_equality,)
    CUSTOM = (MultinodePenaltyFunctions.Functions.custom,)
    COM_EQUALITY = (MultinodePenaltyFunctions.Functions.com_equality,)
    COM_VELOCITY_EQUALITY = (MultinodePenaltyFunctions.Functions.com_velocity_equality,)
    TIME_CONSTRAINT = (MultinodePenaltyFunctions.Functions.time_equality,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return MultinodePenaltyFunctions


class MultinodePenaltyList(UniquePerPhaseOptionList):
    """
    A list of Multtnode penalty

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new MultinodePenalty to the list
    print(self)
        Print the MultinodeConstraintList to the console
    prepare_multinode_penalties(self, ocp) -> list
        Configure all the multinode penalties and put them in a list
    """

    def print(self):
        raise NotImplementedError("Print is not implemented for MultinodePenaltyList")

    def add(
        self,
        multinode_penalty: Any,
        option_type: type = None,
        _multinode_penalty_fcn: type | Any = None,
        **extra_arguments: Any,
    ):
        """
        Add a new MultinodePenalty to the list

        Parameters
        ----------
        multinode_penalty: Callable | MultinodePenaltyFcn
            The chosen phase transition
        option_type
             If the option is MultinodeConstraints
        _multinode_penalty_fcn:
            This is for the hack. We must know which kind of MultinodePenaltyFcn we are currently dealing with
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(multinode_penalty, _multinode_penalty_fcn):
            extra_arguments["custom_function"] = multinode_penalty
            multinode_penalty = _multinode_penalty_fcn.CUSTOM
        super(MultinodePenaltyList, self)._add(
            option_type=option_type, multinode_penalty=multinode_penalty, phase=-1, **extra_arguments
        )

    def add_or_replace_to_penalty_pool(self, ocp):
        """
        Configure all the multinode penalties and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of all the multinode penalties prepared
        """

        for mnc in self:
            for phase in mnc.nodes_phase:
                if phase < 0 or phase >= ocp.n_phases:
                    raise ValueError("nodes_phase of the multinode_penalty must be between 0 and number of phases")

            node_names = [
                f"Phase {phase} node {node.name if isinstance(node, Node) else node}, "
                for node, phase in zip(mnc.nodes, mnc.nodes_phase)
            ]
            mnc.name = "".join(("Multinode: ", *node_names))[:-2]

            if mnc.weight:
                mnc.base = ObjectiveFunction.MayerFunction

            # TODO this only adds, it does not replace
            mnc.list_index = -1
            mnc.add_or_replace_to_penalty_pool(ocp, ocp.nlp[mnc.nodes_phase[0]])
