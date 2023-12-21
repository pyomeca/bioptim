from typing import Callable, Any
from casadi import MX_eye, SX_eye, jacobian, Function, MX, SX, vertcat

from .constraints import PenaltyOption
from .objective_functions import ObjectiveFunction
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyController
from ..limits.penalty_helpers import PenaltyHelpers
from ..misc.enums import Node, PenaltyType
from ..misc.fcn_enum import FcnEnum
from ..misc.options import UniquePerPhaseOptionList
from ..misc.mapping import BiMapping
from ..models.protocols.stochastic_biomodel import StochasticBioModel


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

        self.phase_dynamics = []  # This is set in _prepare_controller_cx
        self.ns = []  # This is set in _prepare_controller_cx
        self.control_types = []  # This is set in _prepare_controller_cx

    def _get_pool_to_add_penalty(self, ocp, nlp):
        raise NotImplementedError("This is an abstract method and should be implemented by child")

    def _add_penalty_to_pool(self, controller: list[PenaltyController, ...]):
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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)
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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

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
        def algebraic_states_equality(
            penalty,
            controllers: list[PenaltyController, ...],
            key: str = "all",
        ):
            """
            The most common continuity function, that is the covariance before equals covariance after for algebraic_states ocp

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the penalty
            controllers: list
                The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

            ctrl_0 = controllers[0]
            algebraic_states_0 = ctrl_0.algebraic_states[key].cx
            out = ctrl_0.cx.zeros(algebraic_states_0.shape)
            for i in range(1, len(controllers)):
                ctrl_i = controllers[i]
                algebraic_states_i = ctrl_i.algebraic_states[key].cx

                if algebraic_states_0.shape != algebraic_states_i.shape:
                    raise RuntimeError(
                        f"Continuity can't be established since the number of x to be matched is {algebraic_states_0.shape} in "
                        f"the pre-transition phase and {algebraic_states_i.shape} post-transition phase. Please use a custom "
                        f"transition or supply states_mapping"
                    )

                out += algebraic_states_i - algebraic_states_0

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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

            times = [controller.tf for i, controller in enumerate(controllers)]

            time_0 = times[0]
            out = controllers[0].cx.zeros((1, 1))
            for i in range(1, len(times)):
                out += time_0 - times[i]

            return out

        @staticmethod
        def track_total_time(penalty, controllers: list[PenaltyController, PenaltyController]):
            """
            The total duration of the phases must be equal to a defined duration

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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)
            return sum([controller.tf for i, controller in enumerate(controllers)])

        @staticmethod
        def stochastic_helper_matrix_explicit(
            penalty,
            controllers: list[PenaltyController, PenaltyController],
        ):
            """
            This functions constrain the helper matrix to its actual value as in Gillis 2013.
            It is explained in more details here: https://doi.org/10.1109/CDC.2013.6761121
            0 = df/dz - dg/dz @ M
            Note that here, we assume that the only z (collocation states) is the next interval states, therefore M is
            not computed at the same node as the other values.

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the phase penalty
            controllers: list[PenaltyController, PenaltyController]
            """

            if not controllers[0].get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")
            if controllers[0].phase_idx != controllers[1].phase_idx:
                raise RuntimeError("For this constraint to make sens, the two nodes must belong to the same phase.")

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

            dt = controllers[0].dt

            M_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[0].algebraic_states["m"].cx, controllers[0].model.matrix_shape_m
            )

            dx = controllers[0].extra_dynamics(0)(
                controllers[0].time.cx,
                controllers[0].states.cx,
                controllers[0].controls.cx,
                controllers[0].parameters.cx,
                controllers[0].algebraic_states.cx
            )

            DdZ_DX_fun = Function(
                "DdZ_DX_fun",
                [
                    controllers[0].time.cx,
                    controllers[0].states.cx,
                    controllers[0].controls.cx,
                    controllers[0].parameters.cx,
                    controllers[0].algebraic_states.cx,
                    controllers[0].model.motor_noise_sym,
                    controllers[0].model.sensory_noise_sym,
                ],
                [jacobian(dx, controllers[0].states.cx)],
            )

            DdZ_DX = DdZ_DX_fun(
                controllers[1].time.cx,
                controllers[1].states.cx,
                controllers[1].controls.cx,
                controllers[1].parameters.cx,
                controllers[1].algebraic_states.cx,
                controllers[1].model.motor_noise_magnitude,
                controllers[1].model.sensory_noise_magnitude,
            )

            CX_eye = SX_eye if controllers[0].cx == SX else MX_eye
            DG_DZ = CX_eye(DdZ_DX.shape[0]) - DdZ_DX * dt / 2

            val = M_matrix @ DG_DZ - CX_eye(M_matrix.shape[0])

            out_vector = StochasticBioModel.reshape_to_vector(val)
            return out_vector

        @staticmethod
        def stochastic_helper_matrix_implicit(
            penalty,
            controllers: list[PenaltyController, PenaltyController],
        ):
            """
            This functions constrain the helper matrix to its actual value as in Gillis 2013.
            It is explained in more details here: https://doi.org/10.1109/CDC.2013.6761121
            0 = df/dz - dg/dz @ M
            Note that here, we assume that the only z (collocation states) is the next interval states, therefore M is
            not computed at the same node as the other values.

            Parameters
            ----------
            penalty : MultinodePenalty
                A reference to the phase penalty
            controllers: list[PenaltyController, PenaltyController]
                    The penalty node elements
            """
            if not controllers[0].get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")
            if controllers[0].phase_idx != controllers[1].phase_idx:
                raise RuntimeError("For this constraint to make sens, the two nodes must belong to the same phase.")

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

            dt = controllers[0].dt

            # TODO: Charbie -> This is only True for x=[q, qdot], u=[tau] (have to think on how to generalize it)
            nu = controllers[0].model.nb_q - controllers[0].model.nb_root
            m_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[0].algebraic_states["m"].cx, controllers[0].model.matrix_shape_m
            )
            a_plus_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[1].algebraic_states["a"].cx, controllers[1].model.matrix_shape_a
            )

            CX_eye = SX_eye if controllers[0].cx == SX else MX_eye
            DG_DZ = CX_eye(a_plus_matrix.shape[0]) - a_plus_matrix * dt / 2

            val = m_matrix @ DG_DZ - CX_eye(2 * nu)

            out_vector = StochasticBioModel.reshape_to_vector(val)
            return out_vector

        @staticmethod
        def stochastic_covariance_matrix_continuity_implicit(
            penalty,
            controllers: list[PenaltyController, PenaltyController],
        ):
            """
            This functions allows to implicitly integrate the covariance matrix.
            P_k+1 = M_k @ (dg/dx @ P @ dg/dx + dg/dw @ sigma_w @ dg/dw) @ M_k
            """
            # TODO: Charbie -> This is only True for x=[q, qdot], u=[tau] (have to think on how to generalize it)

            if not controllers[0].get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

            cov_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[0].algebraic_states["cov"].cx, controllers[0].model.matrix_shape_cov
            )
            cov_matrix_next = StochasticBioModel.reshape_to_matrix(
                controllers[1].algebraic_states["cov"].cx, controllers[1].model.matrix_shape_cov
            )
            a_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[0].algebraic_states["a"].cx, controllers[0].model.matrix_shape_a
            )
            c_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[0].algebraic_states["c"].cx, controllers[0].model.matrix_shape_c
            )
            m_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[0].algebraic_states["m"].cx, controllers[0].model.matrix_shape_m
            )


            sigma_w = vertcat(controllers[0].model.sensory_noise_magnitude, controllers[0].model.motor_noise_magnitude)
            dt = controllers[0].tf / controllers[0].ns
            dg_dw = -dt * c_matrix
            CX_eye = SX_eye if controllers[0].cx == SX else MX_eye
            dg_dx = -CX_eye(a_matrix.shape[0]) - dt / 2 * a_matrix

            cov_next_computed = m_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ m_matrix.T
            cov_implicit_deffect = cov_next_computed - cov_matrix_next

            out_vector = StochasticBioModel.reshape_to_vector(cov_implicit_deffect)
            return out_vector

        @staticmethod
        def stochastic_df_dw_implicit(
            penalty,
            controllers: list[PenaltyController],
        ):
            """
            This function constraints the stochastic matrix C to its actual value which is
            C = df/dw
            """
            # TODO: Charbie -> This is only True for x=[q, qdot], u=[tau] (have to think on how to generalize it)

            if not controllers[0].get_nlp.is_stochastic:
                raise RuntimeError("This function is only valid for stochastic problems")
            
            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)

            dt = controllers[0].dt

            nb_root = controllers[0].model.nb_root
            nu = controllers[0].model.nb_q - controllers[0].model.nb_root

            c_matrix = StochasticBioModel.reshape_to_matrix(
                controllers[0].algebraic_states["c"].cx, controllers[0].model.matrix_shape_c
            )

            q_root = MX.sym("q_root", nb_root, 1)
            q_joints = MX.sym("q_joints", nu, 1)
            qdot_root = MX.sym("qdot_root", nb_root, 1)
            qdot_joints = MX.sym("qdot_joints", nu, 1)
            tau_joints = MX.sym("tau_joints", nu, 1)
            parameters_sym = MX.sym("parameters_sym", controllers[0].parameters.shape, 1)
            algebraic_states_sym = MX.sym("algebraic_states_sym", controllers[0].algebraic_states.shape, 1)

            dx = controllers[0].extra_dynamics(0)(
                controllers[0].time.mx,
                vertcat(q_root, q_joints, qdot_root, qdot_joints),  # States
                tau_joints,
                parameters_sym,
                algebraic_states_sym,
            )

            non_root_index = list(range(nb_root, nb_root + nu)) + list(
                range(nb_root + nu + nb_root, nb_root + nu + nb_root + nu)
            )

            DF_DW_fun = Function(
                "DF_DW_fun",
                [
                    controllers[0].time.mx,
                    q_root,
                    q_joints,
                    qdot_root,
                    qdot_joints,
                    tau_joints,
                    parameters_sym,
                    algebraic_states_sym,
                    controllers[0].model.motor_noise_sym_mx,
                    controllers[0].model.sensory_noise_sym_mx,
                ],
                [
                    jacobian(
                        dx[non_root_index],
                        vertcat(controllers[0].model.motor_noise_sym_mx, controllers[0].model.sensory_noise_sym_mx),
                    )
                ],
            )

            DF_DW = DF_DW_fun(
                controllers[0].time.cx,
                controllers[0].states["q"].cx[:nb_root],
                controllers[0].states["q"].cx[nb_root:],
                controllers[0].states["qdot"].cx[:nb_root],
                controllers[0].states["qdot"].cx[nb_root:],
                controllers[0].controls["tau"].cx,
                controllers[0].parameters.cx,
                controllers[0].algebraic_states.cx,
                controllers[0].model.motor_noise_magnitude,
                controllers[0].model.sensory_noise_magnitude,
            )
            DF_DW_plus = DF_DW_fun(
                controllers[1].time.cx,
                controllers[1].states["q"].cx[:nb_root],
                controllers[1].states["q"].cx[nb_root:],
                controllers[1].states["qdot"].cx[:nb_root],
                controllers[1].states["qdot"].cx[nb_root:],
                controllers[1].controls.cx,
                controllers[1].parameters.cx,
                controllers[1].algebraic_states.cx,
                controllers[1].model.motor_noise_magnitude,
                controllers[1].model.sensory_noise_magnitude,
            )

            out = c_matrix - (-(DF_DW + DF_DW_plus) / 2 * dt)

            out_vector = StochasticBioModel.reshape_to_vector(out)
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

            MultinodePenaltyFunctions.Functions._prepare_controller_cx(penalty, controllers)
            return penalty.custom_function(controllers, **extra_params)

        @staticmethod
        def _prepare_controller_cx(penalty, controllers: list[PenaltyController, ...]):
            """
            This calls the _compute_controller_cx function for each of the controller then dispatch the cx appropriately
            to the controllers
            """

            # This will be set again in set_penalty, but we need it before
            penalty.phase_dynamics = [c.get_nlp.phase_dynamics for c in controllers]
            penalty.ns = [c.get_nlp.ns for c in controllers]
            penalty.control_types = [c.get_nlp.control_type for c in controllers]

            indices = PenaltyHelpers.get_multinode_penalty_subnodes_starting_index(penalty)
            for index, c in zip(indices, controllers):
                c.cx_index_to_get = index

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

        if "phase" in extra_arguments.keys():
            phase = extra_arguments["phase"]
            del extra_arguments["phase"]
        else:
            phase = -1

        super(MultinodePenaltyList, self)._add(
            option_type=option_type, multinode_penalty=multinode_penalty, phase=phase, **extra_arguments
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
