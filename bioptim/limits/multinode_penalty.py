from typing import Callable, Union, Any
from warnings import warn
from enum import Enum

import biorbd_casadi as biorbd
from casadi import vertcat, MX


from .path_conditions import Bounds
from ..limits.penalty import PenaltyFunctionAbstract, PenaltyNodeList
from ..misc.enums import Node, PenaltyType


class MultinodePenalty:
    """
    A placeholder for a multi node penalties

    Attributes
    ----------
    min_bound: list
        The minimal bound of the multi node penalties
    max_bound: list
        The maximal bound of the multi node penalties
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
    multinode_penalty: Union[Callable, Any]
        The nature of the cost function is the multi node penalty
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    """

    def __init__(
        self,
        phase_first_idx: int,
        phase_second_idx: int,
        first_node: Union[Node, int],
        second_node: Union[Node, int],
        multinode_penalty: Union[Callable, Any],
        penalty_type=PenaltyType.USER,
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

        if first_node not in (Node.START, Node.MID, Node.PENULTIMATE, Node.END) and not isinstance(first_node, int):
            raise NotImplementedError(
                "Multi Node Penalty only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int."
            )
        if second_node not in (Node.START, Node.MID, Node.PENULTIMATE, Node.END) and not isinstance(second_node, int):
            raise NotImplementedError(
                "Multi Node Penalty only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int."
            )

        self.multinode_penalty = multinode_penalty
        self.phase_first_idx = phase_first_idx
        self.phase_second_idx = phase_second_idx
        self.phase_pre_idx = phase_first_idx
        self.phase_post_idx = phase_second_idx
        self.first_node = first_node
        self.second_node = second_node
        self.node = self.first_node, self.second_node
        self.node_idx = [0]
        self.penalty_type = penalty_type  # TODO: fix with proper multiple inheritence


class MultinodePenaltyFunctions(PenaltyFunctionAbstract):
    """
    Internal implementation of the phase transitions
    """

    class Functions:
        """
        Implementation of all the Multi Node Penalty
        """

        @staticmethod
        def equality(multinode_penalty, all_pn):
            """
            The most common continuity function, that is state before equals state after

            Parameters
            ----------
            multinode_penalty : MultinodePenalty
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            states_pre = multinode_penalty.states_mapping.to_second.map(nlp_pre.states.cx_end)
            states_post = multinode_penalty.states_mapping.to_first.map(nlp_post.states.cx)

            if states_pre.shape != states_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {states_pre.shape} in the "
                    f"pre-transition phase and {states_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            return states_pre - states_post

        @staticmethod
        def com_equality(multinode_penalty, all_pn):
            """
            The centers of mass are equals for the specified phases and the specified nodes

            Parameters
            ----------
            multinode_penalty : MultinodePenalty
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            states_pre = multinode_penalty.states_mapping.to_second.map(nlp_pre.states.cx_end)
            states_post = multinode_penalty.states_mapping.to_first.map(nlp_post.states.cx)

            states_post_sym_list = [MX.sym(f"{key}", *nlp_post.states[key].mx.shape) for key in nlp_post.states.keys()]
            states_post_sym = vertcat(*states_post_sym_list)

            if states_pre.shape != states_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {states_pre.shape} in the "
                    f"pre-transition phase and {states_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            pre_com = nlp_pre.model.CoM(states_pre[nlp_pre.states["q"].index, :]).to_mx()
            post_com = nlp_post.model.CoM(states_post_sym_list[0]).to_mx()

            pre_states_cx = nlp_pre.states.cx_end
            post_states_cx = nlp_post.states.cx

            return biorbd.to_casadi_func(
                "com_equality",
                pre_com - post_com,
                states_pre,
                states_post_sym,
            )(pre_states_cx, post_states_cx)

        @staticmethod
        def com_velocity_equality(multinode_penalty, all_pn):
            """
            The centers of mass velocity are equals for the specified phases and the specified nodes

            Parameters
            ----------
            multinode_penalty : MultinodePenalty
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The difference between the state after and before
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            states_pre = multinode_penalty.states_mapping.to_second.map(nlp_pre.states.cx_end)
            states_post = multinode_penalty.states_mapping.to_first.map(nlp_post.states.cx)

            states_post_sym_list = [MX.sym(f"{key}", *nlp_post.states[key].mx.shape) for key in nlp_post.states.keys()]
            states_post_sym = vertcat(*states_post_sym_list)

            if states_pre.shape != states_post.shape:
                raise RuntimeError(
                    f"Continuity can't be established since the number of x to be matched is {states_pre.shape} in the "
                    f"pre-transition phase and {states_post.shape} post-transition phase. Please use a custom "
                    f"transition or supply states_mapping"
                )

            pre_com_dot = nlp_pre.model.CoMdot(
                states_pre[nlp_pre.states["q"].index, :], states_pre[nlp_pre.states["qdot"].index, :]
            ).to_mx()
            post_com_dot = nlp_post.model.CoMdot(states_post_sym_list[0], states_post_sym_list[1]).to_mx()

            pre_states_cx = nlp_pre.states.cx_end
            post_states_cx = nlp_post.states.cx

            return biorbd.to_casadi_func(
                "com_dot_equality",
                pre_com_dot - post_com_dot,
                states_pre,
                states_post_sym,
            )(pre_states_cx, post_states_cx)

        @staticmethod
        def custom(multinode_penalty, all_pn, **extra_params):
            """
            Calls the custom transition function provided by the user

            Parameters
            ----------
            multinode_penalty: MultinodePenalty
                A reference to the phase transition
            all_pn: PenaltyNodeList
                    The penalty node elements

            Returns
            -------
            The expected difference between the last and first node provided by the user
            """

            nlp_pre, nlp_post = all_pn[0].nlp, all_pn[1].nlp
            return multinode_penalty.custom_function(multinode_penalty, nlp_pre, nlp_post, **extra_params)

    @staticmethod
    def get_dt(_):
        return 1
