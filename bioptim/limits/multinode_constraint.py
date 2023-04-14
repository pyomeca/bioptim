from typing import Callable, Any

from casadi import vertcat, MX

from .constraints import Constraint
from .path_conditions import Bounds
from .objective_functions import ObjectiveFunction
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyNodeList
from ..misc.enums import Node, InterpolationType, PenaltyType, CXStep
from ..misc.fcn_enum import FcnEnum
from ..misc.options import UniquePerPhaseOptionList


class BinodeConstraint(Constraint):
    """
    A placeholder for a binode constraints

    Attributes
    ----------
    min_bound: list
        The minimal bound of the binode constraints
    max_bound: list
        The maximal bound of the binode constraints
    bounds: Bounds
        The bounds (will be filled with min_bound/max_bound)
    weight: float
        The weight of the cost function
    quadratic: bool
        If the objective function is quadratic
    phase_first_idx: int
        The first index of the phase of concern
    phase_second_idx: int
        The second index of the phase of concern
    first_node: Node
        The kind of the first node
    second_node: Node
        The kind of the second node
    dt: float
        The delta time
    node_idx: int
        The index of the node in nlp pre
    binode_constraint: Callable | Any
        The nature of the cost function is the bi node constraint
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    """

    def __init__(
        self,
        phase_first_idx: int,
        phase_second_idx: int,
        first_node: Node | int,
        second_node: Node | int,
        binode_constraint: Callable | Any = None,
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

        force_binode = False
        if "force_binode" in params:
            # This is a hack to circumvent the apparatus that moves the functions to a custom function
            # It is necessary for PhaseTransition
            force_binode = True
            del params["force_binode"]

        if not isinstance(binode_constraint, BinodeConstraintFcn) and not force_binode:
            custom_function = binode_constraint
            binode_constraint = BinodeConstraintFcn.CUSTOM
        super(Constraint, self).__init__(penalty=binode_constraint, custom_function=custom_function, **params)

        if first_node not in (Node.START, Node.MID, Node.PENULTIMATE, Node.END):
            if not isinstance(first_node, int):
                raise NotImplementedError(
                    "Binode Constraint only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int."
                )
        if second_node not in (Node.START, Node.MID, Node.PENULTIMATE, Node.END):
            if not isinstance(second_node, int):
                raise NotImplementedError(
                    "Binode Constraint only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int."
                )
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(interpolation=InterpolationType.CONSTANT)

        self.binode_constraint = True
        self.allnode_constraint = False
        self.weight = weight
        self.quadratic = True
        self.phase_first_idx = phase_first_idx
        self.phase_second_idx = phase_second_idx
        self.phase_pre_idx = phase_first_idx
        self.phase_post_idx = phase_second_idx
        self.first_node = first_node
        self.second_node = second_node
        self.node = self.first_node, self.second_node
        self.dt = 1
        self.node_idx = [0]
        self.penalty_type = PenaltyType.INTERNAL

    def _add_penalty_to_pool(self, all_pn: PenaltyNodeList | list | tuple):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp
        if self.weight == 0:
            pool = nlp.g_internal if nlp else ocp.g_internal
        else:
            pool = nlp.J_internal if nlp else ocp.J_internal
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        if self.weight == 0:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal
        else:
            g_to_add_to = nlp.J_internal if nlp else ocp.J_internal

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


class AllNodeConstraint(Constraint):
    """
    A placeholder for a binode constraints

    Attributes
    ----------
    min_bound: list
        The minimal bound of the binode constraints
    max_bound: list
        The maximal bound of the binode constraints
    bounds: Bounds
        The bounds (will be filled with min_bound/max_bound)
    weight: float
        The weight of the cost function
    quadratic: bool
        If the objective function is quadratic
    phase_idx: int
        The index of the phase of concern
    all_node: Node
        The kind of the node
    dt: float
        The delta time
    node_idx: int
        The index of the node in nlp pre
    allnode_constraint: Callable | Any
        The nature of the cost function is the binode constraint
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    """

    def __init__(
        self,
        phase_idx: int,
        node: Node | int,
        allnode_constraint: Callable | Any = None,
        custom_function: Callable = None,
        min_bound: float = 0,
        max_bound: float = 0,
        weight: float = 0,
        **params: Any,
    ):
        """
        Parameters
        ----------
        phase_idx: int
            The index of the phase of concern
        params:
            Generic parameters for options
        """

        force_allnode = False
        if "force_allnode" in params:
            force_allnode = True
            del params["force_allnode"]

        if not isinstance(allnode_constraint, AllNodeConstraintFcn) and not force_allnode:
            custom_function = allnode_constraint
            allnode_constraint = AllNodeConstraintFcn.CUSTOM
        super(Constraint, self).__init__(penalty=allnode_constraint, custom_function=custom_function, **params)

        if node is not Node.ALL:
            if not isinstance(node, int):
                raise NotImplementedError(
                    "Allnode Constraint only works with Node.ALL"
                )

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        self.allnode_constraint = True
        self.binode_constraint = False
        self.weight = weight
        self.quadratic = True
        self.phase_idx = phase_idx
        self.node = node
        self.dt = 1
        self.node_idx = [0]
        self.penalty_type = PenaltyType.INTERNAL

    def _add_penalty_to_pool(self, all_pn: PenaltyNodeList | list | tuple):
        ocp = all_pn[0].ocp
        nlp = all_pn[0].nlp
        if self.weight == 0:
            pool = nlp.g_internal if nlp else ocp.g_internal
        else:
            pool = nlp.J_internal if nlp else ocp.J_internal
        pool[self.list_index] = self

    def ensure_penalty_sanity(self, ocp, nlp):
        if self.weight == 0:
            g_to_add_to = nlp.g_internal if nlp else ocp.g_internal
        else:
            g_to_add_to = nlp.J_internal if nlp else ocp.J_internal

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


class BinodeConstraintList(UniquePerPhaseOptionList):
    """
    A list of Bi Node Constraint

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new BinodeConstraint to the list
    print(self)
        Print the BinodeConstraintList to the console
    prepare_binode_constraint(self, ocp) -> list
        Configure all the binode_constraint and put them in a list
    """

    def add(self, binode_constraint: Any, **extra_arguments: Any):
        """
        Add a new BiNodeConstraint to the list

        Parameters
        ----------
        binode_constraint: Callable | BinodeConstraintFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(binode_constraint, BinodeConstraintFcn):
            extra_arguments["custom_function"] = binode_constraint
            binode_constraint = BinodeConstraintFcn.CUSTOM
        super(BinodeConstraintList, self)._add(
            option_type=BinodeConstraint, binode_constraint=binode_constraint, phase=-1, **extra_arguments
        )

    def print(self):
        """
        Print the BiNodeConstraintList to the console
        """
        raise NotImplementedError("Printing of BiNodeConstraintList is not ready yet")

    def prepare_binode_constraints(self, ocp) -> list:
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
        full_phase_binode_constraint = []
        for mnc in self:
            if mnc.phase_first_idx >= ocp.n_phases or mnc.phase_second_idx >= ocp.n_phases:
                raise RuntimeError("Phase index of the binode_constraint is higher than the number of phases")
            if mnc.phase_first_idx < 0 or mnc.phase_second_idx < 0:
                raise RuntimeError("Phase index of the binode_constraint need to be positive")

            if mnc.weight:
                mnc.base = ObjectiveFunction.MayerFunction

            full_phase_binode_constraint.append(mnc)

        return full_phase_binode_constraint


class AllNodeConstraintList(UniquePerPhaseOptionList):
    """
    A list of All Node Constraint

    Methods
    -------
    add(self, transition: Callable | PhaseTransitionFcn, phase: int = -1, **extra_arguments)
        Add a new AllNodeConstraint to the list
    print(self)
        Print the AllConstraintList to the console
    prepare_allnode_constraint(self, ocp) -> list
        Configure all the allnode_constraint and put them in a list
    """

    def add(self, allnode_constraint: Any, **extra_arguments: Any):
        """
        Add a new AllConstraint to the list

        Parameters
        ----------
        allnode_constraint: Callable | AllConstraintFcn
            The chosen phase transition
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if not isinstance(allnode_constraint, AllNodeConstraintFcn):
            extra_arguments["custom_function"] = allnode_constraint
            allnode_constraint = AllNodeConstraintFcn.CUSTOM
        super(AllNodeConstraintList, self)._add(
            option_type=AllNodeConstraint, allnode_constraint=allnode_constraint, phase=-1, **extra_arguments
        )

    def print(self):
        """
        Print the AllNodeConstraintList to the console
        """
        raise NotImplementedError("Printing of AllConstraintList is not ready yet")

    def prepare_allnode_constraints(self, ocp) -> list:
        """
        Configure all the phase transitions and put them in a list

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp

        Returns
        -------
        The list of all node constraints prepared
        """
        full_phase_allnode_constraint = []
        for mnc in self:
            if mnc.phase_idx >= ocp.n_phases:
                raise RuntimeError("Phase index of the allnode_constraint is higher than the number of phases")
            if mnc.phase_idx < 0:
                raise RuntimeError("Phase index of the allnode_constraint need to be positive")

            if not mnc.weight: # ajouter un check, sinon mettre 1
                mnc.base = 1

            full_phase_allnode_constraint.append(mnc)

        return full_phase_allnode_constraint



class BinodeConstraintFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """

    class Functions:
        """
        Implementation of all the Binode Constraint
        """

        @staticmethod
        def states_equality(binode_constraint, all_pn, key: str = "all"):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            binode_constraint : BinodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            states_pre = binode_constraint.states_mapping.to_second.map(nlp_pre.states[0].get_cx(key, CXStep.CX_END))   # TODO: [0] to [node_index]
            states_post = binode_constraint.states_mapping.to_first.map(nlp_post.states[0].get_cx(key, CXStep.CX_START))    # TODO: [0] to [node_index]

            if states_pre.shape != states_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {states_pre.shape} in the "
                    f"pre-transition phase and {states_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            return states_pre - states_post

        @staticmethod
        def controls_equality(binode_constraint, all_pn, key: str = "all"):
            """
            The controls before equals controls after

            Parameters
            ----------
            binode_constraint : BinodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the controls after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            controls_pre = nlp_pre.controls[0].get_cx(key, CXStep.CX_END)   # TODO: [0] to [node_index]
            controls_post = nlp_post.controls[0].get_cx(key, CXStep.CX_START)   # TODO: [0] to [node_index]

            if controls_pre.shape != controls_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {controls_pre.shape} in the "
                    f"pre-transition phase and {controls_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            return controls_pre - controls_post

        @staticmethod
        def com_equality(binode_constraint, all_pn):
            """
            The centers of mass are equals for the specified phases and the specified nodes

            Parameters
            ----------
            binode_constraint : BinodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            states_pre = binode_constraint.states_mapping.to_second.map(nlp_pre.states[0].cx_end)   # TODO: [0] to [node_index]
            states_post = binode_constraint.states_mapping.to_first.map(nlp_post.states[0].cx_start)  # TODO: [0] to [node_index]

            states_post_sym_list = [MX.sym(f"{key}", *nlp_post.states[0][key].mx.shape) for key in nlp_post.states[0]]  # TODO: [0] to [node_index]
            states_post_sym = vertcat(*states_post_sym_list)

            if states_pre.shape != states_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {states_pre.shape} in the "
                    f"pre-transition phase and {states_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            pre_com = nlp_pre.model.center_of_mass(states_pre[nlp_pre.states[0]["q"].index, :]) # TODO: [0] to [node_index]
            post_com = nlp_post.model.center_of_mass(states_post_sym_list[0])

            pre_states_cx = nlp_pre.states[0].cx_end    # TODO: [0] to [node_index]
            post_states_cx = nlp_post.states[0].cx_start  # TODO: [0] to [node_index]

            return nlp_pre.to_casadi_func(
                "com_equality",
                pre_com - post_com,
                states_pre,
                states_post_sym,
            )(pre_states_cx, post_states_cx)

        @staticmethod
        def com_velocity_equality(binode_constraint, all_pn):
            """
            The centers of mass velocity are equals for the specified phases and the specified nodes

            Parameters
            ----------
            binode_constraint : BinodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            states_pre = binode_constraint.states_mapping.to_second.map(nlp_pre.states[0].cx_end)   # TODO: [0] to [node_index]
            states_post = binode_constraint.states_mapping.to_first.map(nlp_post.states[0].cx_start)    # TODO: [0] to [node_index]

            states_post_sym_list = [MX.sym(f"{key}", *nlp_post.states[0][key].mx.shape) for key in nlp_post.states[0]]  # TODO: [0] to [node_index]
            states_post_sym = vertcat(*states_post_sym_list)

            if states_pre.shape != states_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {states_pre.shape} in the "
                    f"pre-transition phase and {states_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            pre_com_dot = nlp_pre.model.center_of_mass_velocity(
                states_pre[nlp_pre.states[0]["q"].index, :], states_pre[nlp_pre.states[0]["qdot"].index, :] # TODO: [0] to [node_index]
            )
            post_com_dot = nlp_post.model.center_of_mass_velocity(states_post_sym_list[0], states_post_sym_list[1])

            pre_states_cx = nlp_pre.states[0].cx_end    # TODO: [0] to [node_index]
            post_states_cx = nlp_post.states[0].cx_start    # TODO: [0] to [node_index]

            return nlp_pre.to_casadi_func(
                "com_dot_equality",
                pre_com_dot - post_com_dot,
                states_pre,
                states_post_sym,
            )(pre_states_cx, post_states_cx)

        @staticmethod
        def time_equality(binode_constraint, all_pn):
            """
            The duration of one phase must be the same as the duration of another phase

            Parameters
            ----------
            binode_constraint : BinodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the duration of the phases
            """
            time_pre_idx = None
            for i in range(all_pn[0].nlp.parameters.cx.shape[0]):
                param_name = all_pn[0].nlp.parameters.cx[i].name()
                if param_name == "time_phase_" + str(all_pn[0].nlp.phase_idx):
                    time_pre_idx = all_pn[0].nlp.phase_idx
            if time_pre_idx is None:
                raise RuntimeError(
                    f"Time constraint can't be established since the first phase has no time parameter. "
                    f"\nTime parameter can be added with : "
                    f"\nobjective_functions.add(ObjectiveFcn.[Mayer or Lagrange].MINIMIZE_TIME) or "
                    f"\nwith constraints.add(ConstraintFcn.TIME_CONSTRAINT)."
                )

            time_post_idx = None
            for i in range(all_pn[1].nlp.parameters.cx.shape[0]):
                param_name = all_pn[1].nlp.parameters.cx[i].name()
                if param_name == "time_phase_" + str(all_pn[1].nlp.phase_idx):
                    time_post_idx = all_pn[1].nlp.phase_idx
            if time_post_idx is None:
                raise RuntimeError(
                    f"Time constraint can't be established since the second phase has no time parameter. Time parameter "
                    f"can be added with : objective_functions.add(ObjectiveFcn.[Mayer or Lagrange].MINIMIZE_TIME) or "
                    f"with constraints.add(ConstraintFcn.TIME_CONSTRAINT)."
                )

            time_pre, time_post = all_pn[0].nlp.parameters.cx[time_pre_idx], all_pn[1].nlp.parameters.cx[time_post_idx]

            return time_pre - time_post

        @staticmethod
        def custom(binode_constraint, all_pn, **extra_params):
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            binode_constraint: BinodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            return binode_constraint.custom_function(binode_constraint, nlp_pre, nlp_post, **extra_params)


class AllNodeConstraintFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """
    class Functions:
        """
        Implementation of all the AllNode Constraint
        """

        @staticmethod
        def custom(allnode_constraint, all_pn, **extra_params):
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            allnode_constraint: AllNodeConstraint
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            nlp_all = all_pn.nlp

            return allnode_constraint.custom_function(allnode_constraint, nlp_all, **extra_params)


class BinodeConstraintFcn(FcnEnum):
    """
    Selection of valid binode constraint functions
    """

    STATES_EQUALITY = (BinodeConstraintFunctions.Functions.states_equality,)
    CONTROLS_EQUALITY = (BinodeConstraintFunctions.Functions.controls_equality,)
    CUSTOM = (BinodeConstraintFunctions.Functions.custom,)
    COM_EQUALITY = (BinodeConstraintFunctions.Functions.com_equality,)
    COM_VELOCITY_EQUALITY = (BinodeConstraintFunctions.Functions.com_velocity_equality,)
    TIME_CONSTRAINT = (BinodeConstraintFunctions.Functions.time_equality,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return BinodeConstraintFunctions


class AllNodeConstraintFcn(FcnEnum):
    """
    Selection of valid allnode constraint functions
    """
    CUSTOM = (AllNodeConstraintFunctions.Functions.custom,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return AllNodeConstraintFunctions
