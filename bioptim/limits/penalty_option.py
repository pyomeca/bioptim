from typing import Any, Union, Callable

import biorbd_casadi as biorbd
from casadi import horzcat, Function, MX, SX
import numpy as np

from .penalty_node import PenaltyNodeList
from ..misc.enums import Node, PlotType, ControlType
from ..misc.mapping import Mapping
from ..misc.options import OptionGeneric


class PenaltyOption(OptionGeneric):
    """
    A placeholder for a penalty

    Attributes
    ----------
    node: Node
        The node within a phase on which the penalty is acting on
    quadratic: bool
        If the penalty is quadratic
    rows: Union[list, tuple, range, np.ndarray]
        The index of the rows in the penalty to keep
    cols: Union[list, tuple, range, np.ndarray]
        The index of the columns in the penalty to keep
    expand: bool
        If the penalty should be expanded or not
    target: np.array(target)
        A target to track for the penalty
    target_plot_name: str
        The plot name of the target
    target_to_plot: np.ndarray
        The subset of the target to plot
    plot_target: bool
        If the target should be plotted
    custom_function: Callable
        A user defined function to call to get the penalty
    node_idx: Union[list, tuple, Node]
        The index in nlp to apply the penalty to
    dt: float
        The delta time
    function: Function
        The casadi function of the penalty
    weighted_function: Function
        The casadi function of the penalty weighted
    derivative: bool
        If the minimization is applied on the numerical derivative of the state [f(t+1) - f(t)]
    explicit_derivative: bool
        If the minimization is applied to derivative of the penalty [f(t, t+1)]
    transition: bool
        If the penalty is a transition
    phase_pre_idx: int
        The index of the nlp of pre when penalty is transition
    phase_post_idx: int
        The index of the nlp of post when penalty is transition
    is_internal: bool
        If the penalty is from the user or from bioptim
    multi_thread: bool
        If the penalty is multithreaded

    Methods
    -------
    set_penalty(self, penalty: Union[MX, SX], all_pn: PenaltyNodeList)
        Prepare the dimension and index of the penalty (including the target)
    _set_dim_idx(self, dim: Union[list, tuple, range, np.ndarray], n_rows: int)
        Checks if the variable index is consistent with the requested variable.
    _check_target_dimensions(self, all_pn: PenaltyNodeList, n_time_expected: int)
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay
    _set_penalty_function(self, all_pn: Union[PenaltyNodeList, list, tuple], fcn: Union[MX, SX])
        Finalize the preparation of the penalty (setting function and weighted_function)
    add_target_to_plot(self, all_pn: PenaltyNodeList, combine_to: str)
        Interface to the plot so it can be properly added to the proper plot
    _finish_add_target_to_plot(self, all_pn: PenaltyNodeList)
        Internal interface to add (after having check the target dimensions) the target to the plot if needed
    add_or_replace_to_penalty_pool(self, ocp, nlp)
        Doing some configuration on the penalty and add it to the list of penalty
    _add_penalty_to_pool(self, all_pn: PenaltyNodeList)
        Return the penalty pool for the specified penalty (abstract)
    clear_penalty(self, ocp, nlp)
        Resets a penalty. A negative penalty index creates a new empty penalty (abstract)
    _get_penalty_node_list(self, ocp, nlp) -> PenaltyNodeList
        Get the actual node (time, X and U) specified in the penalty
    """

    def __init__(
        self,
        penalty: Any,
        phase: int = 0,
        node: Union[Node, list, tuple] = Node.DEFAULT,
        target: np.ndarray = None,
        quadratic: bool = None,
        weight: float = 1,
        derivative: bool = False,
        explicit_derivative: bool = False,
        integrate: bool = False,
        index: list = None,
        rows: Union[list, tuple, range, np.ndarray] = None,
        cols: Union[list, tuple, range, np.ndarray] = None,
        custom_function: Callable = None,
        is_internal: bool = False,
        multi_thread: bool = None,
        expand: bool = False,
        **params: Any,
    ):
        """
        Parameters
        ----------
        penalty: PenaltyType
            The actual penalty
        phase: int
            The phase the penalty is acting on
        node: Union[Node, list, tuple]
            The node within a phase on which the penalty is acting on
        target: np.ndarray
            A target to track for the penalty
        quadratic: bool
            If the penalty is quadratic
        weight: float
            The weighting applied to this specific penalty
        derivative: bool
            If the function should be evaluated at X and X+1
        explicit_derivative: bool
            If the function should be evaluated at [X, X+1]
        index: int
            The component index the penalty is acting on
        custom_function: Callable
            A user defined function to call to get the penalty
        is_internal: bool
            If the penalty is internally defined [True] or by the user
        **params: dict
            Generic parameters for the penalty
        """

        super(PenaltyOption, self).__init__(phase=phase, type=penalty, **params)
        self.node: Union[Node, list, tuple] = node
        self.quadratic = quadratic

        if index is not None and rows is not None:
            raise ValueError("rows and index cannot be defined simultaneously since they are the same variable")
        self.rows = rows if rows is not None else index
        self.cols = cols
        self.expand = expand

        self.target = None
        if target is not None:
            self.target = np.array(target)
            if len(self.target.shape) == 0:
                self.target = self.target[np.newaxis]
            if len(self.target.shape) == 1:
                self.target = self.target[:, np.newaxis]
        self.target_plot_name = None
        self.target_to_plot = None
        self.plot_target = True

        self.custom_function = custom_function

        self.node_idx = []
        self.dt = 0
        self.weight = weight
        self.function: Union[Function, None] = None
        self.weighted_function: Union[Function, None] = None
        self.derivative = derivative
        self.explicit_derivative = explicit_derivative
        self.integrate = integrate
        self.transition = False
        self.phase_pre_idx = None
        self.phase_post_idx = None
        if self.derivative and self.explicit_derivative:
            raise ValueError("derivative and explicit_derivative cannot be both True")
        self.is_internal = is_internal

        self.multi_thread = multi_thread

    def set_penalty(self, penalty: Union[MX, SX], all_pn: PenaltyNodeList):
        """
        Prepare the dimension and index of the penalty (including the target)

        Parameters
        ----------
        penalty: Union[MX, SX],
            The actual penalty function
        all_pn: PenaltyNodeList
            The penalty node elements
        """

        self.rows = self._set_dim_idx(self.rows, penalty.rows())
        self.cols = self._set_dim_idx(self.cols, penalty.columns())
        if self.target is not None:
            self._check_target_dimensions(all_pn, len(all_pn.t))
            if self.plot_target:
                self._finish_add_target_to_plot(all_pn)
        self._set_penalty_function(all_pn, penalty)
        self._add_penalty_to_pool(all_pn)

    def _set_dim_idx(self, dim: Union[list, tuple, range, np.ndarray], n_rows: int):
        """
        Checks if the variable index is consistent with the requested variable.

        Parameters
        ----------
        dim: Union[list, tuple, range]
            The dimension to set
        n_rows: int
            The expected row shape

        Returns
        -------
        The formatted indices
        """

        if dim is None:
            dim = range(n_rows)
        else:
            if isinstance(dim, int):
                dim = [dim]
            if max(dim) > n_rows:
                raise RuntimeError(f"{self.name} index cannot be higher than nx ({n_rows})")
        dim = np.array(dim)
        if not np.issubdtype(dim.dtype, np.integer):
            raise RuntimeError(f"{self.name} index must be a list of integer")
        return dim

    def _check_target_dimensions(self, all_pn: PenaltyNodeList, n_time_expected: int):
        """
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements
        n_time_expected: Union[list, tuple]
            The expected shape (n_rows, ns) of the data to track
        """

        n_dim = len(self.target.shape)
        if n_dim != 2 and n_dim != 3:
            raise RuntimeError(f"target cannot be a vector (it can be a matrix with time dimension equals to 1 though)")
        if self.target.shape[-1] == 1:
            self.target = np.repeat(self.target, n_time_expected, axis=-1)

        shape = (len(self.rows), n_time_expected) if n_dim == 2 else (len(self.rows), len(self.cols), n_time_expected)
        if self.target.shape != shape:
            raise RuntimeError(
                f"target {self.target.shape} does not correspond to expected size {shape} for penalty {self.name}"
            )

        # If the target is on controls and control is constant, there will be one value missing
        if all_pn is not None:
            if (
                all_pn.nlp.control_type == ControlType.CONSTANT
                and all_pn.nlp.ns in all_pn.t
                and self.target.shape[-1] == all_pn.nlp.ns
            ):
                if all_pn.t[-1] != all_pn.nlp.ns:
                    raise NotImplementedError("Modifying target for END not being last is not implemented yet")
                self.target = np.concatenate((self.target, np.nan * np.zeros((self.target.shape[0], 1))), axis=1)

    def _set_penalty_function(self, all_pn: Union[PenaltyNodeList, list, tuple], fcn: Union[MX, SX]):
        """
        Finalize the preparation of the penalty (setting function and weighted_function)

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The nodes
        fcn: Union[MX, SX]
            The value of the penalty function
        """

        # Sanity checks
        if self.transition and self.explicit_derivative:
            raise ValueError("transition and explicit_derivative cannot be true simultaneously")
        if self.transition and self.derivative:
            raise ValueError("transition and derivative cannot be true simultaneously")
        if self.derivative and self.explicit_derivative:
            raise ValueError("derivative and explicit_derivative cannot be true simultaneously")

        if self.transition:
            ocp = all_pn[0].ocp
            nlp = all_pn[0].nlp
            nlp_post = all_pn[1].nlp
            name = self.name.replace("->", "_").replace(" ", "_")
            state_cx = horzcat(nlp.states.cx_end, nlp_post.states.cx)
            control_cx = horzcat(nlp.controls.cx_end, nlp_post.controls.cx)

        else:
            ocp = all_pn.ocp
            nlp = all_pn.nlp
            name = self.name
            if self.integrate:
                state_cx = horzcat(*([all_pn.nlp.states.cx] + all_pn.nlp.states.cx_intermediates_list))
                control_cx = all_pn.nlp.controls.cx
            else:
                state_cx = all_pn.nlp.states.cx
                control_cx = all_pn.nlp.controls.cx
            if self.explicit_derivative:
                if self.derivative:
                    raise RuntimeError("derivative and explicit_derivative cannot be simultaneously true")
                state_cx = horzcat(state_cx, all_pn.nlp.states.cx_end)
                control_cx = horzcat(control_cx, all_pn.nlp.controls.cx_end)

        param_cx = nlp.cx(nlp.parameters.cx)

        # Do not use nlp.add_casadi_func because all functions must be registered
        self.function = biorbd.to_casadi_func(
            name, fcn[self.rows, self.cols], state_cx, control_cx, param_cx, expand=self.expand
        )
        if self.derivative:
            state_cx = horzcat(all_pn.nlp.states.cx_end, all_pn.nlp.states.cx)
            control_cx = horzcat(all_pn.nlp.controls.cx_end, all_pn.nlp.controls.cx)
            self.function = biorbd.to_casadi_func(
                f"{name}",
                self.function(all_pn.nlp.states.cx_end, all_pn.nlp.controls.cx_end, param_cx)
                - self.function(all_pn.nlp.states.cx, all_pn.nlp.controls.cx, param_cx),
                state_cx,
                control_cx,
                param_cx,
            )

        modified_fcn = self.function(state_cx, control_cx, param_cx)

        dt_cx = nlp.cx.sym("dt", 1, 1)
        weight_cx = nlp.cx.sym("weight", 1, 1)
        target_cx = nlp.cx.sym("target", modified_fcn.shape)
        modified_fcn = modified_fcn - target_cx

        if self.weight:
            modified_fcn = modified_fcn ** 2 if self.quadratic else modified_fcn
            modified_fcn = weight_cx * modified_fcn * dt_cx
        else:
            modified_fcn = modified_fcn * dt_cx

        # Do not use nlp.add_casadi_func because all of them must be registered
        self.weighted_function = Function(
            name, [state_cx, control_cx, param_cx, weight_cx, target_cx, dt_cx], [modified_fcn]
        )

        if ocp.n_threads > 1 and self.multi_thread and len(self.node_idx) > 1:
            self.function = self.function.map(len(self.node_idx), "thread", ocp.n_threads)
            self.weighted_function = self.weighted_function.map(len(self.node_idx), "thread", ocp.n_threads)
        else:
            self.multi_thread = False  # Override the multi_threading, since only one node is optimized

        if self.expand:
            self.function = self.function.expand()
            self.weighted_function = self.weighted_function.expand()

    def add_target_to_plot(self, all_pn: PenaltyNodeList, combine_to: str):
        """
        Interface to the plot so it can be properly added to the proper plot

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements
        combine_to: str
            The name of the underlying plot to combine the tracking data to
        """

        if self.target is None or combine_to is None:
            return

        self.target_plot_name = combine_to
        if self.target.shape[1] == all_pn.nlp.ns:
            self.target_to_plot = np.concatenate((self.target, np.nan * np.ndarray((self.target.shape[0], 1))), axis=1)
        else:
            self.target_to_plot = self.target

    def _finish_add_target_to_plot(self, all_pn: PenaltyNodeList):
        """
        Internal interface to add (after having check the target dimensions) the target to the plot if needed

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements

        """

        if self.target_to_plot is not None:
            all_pn.ocp.add_plot(
                self.target_plot_name,
                lambda t, x, u, p: self.target_to_plot,
                plot_type=PlotType.POINT,
                phase=all_pn.nlp.phase_idx,
                axes_idx=Mapping(self.rows),
                node_idx=self.node_idx,
            )

    def add_or_replace_to_penalty_pool(self, ocp, nlp):
        """
        Doing some configuration on the penalty and add it to the list of penalty

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """
        if not self.name:
            if self.type.name == "CUSTOM":
                self.name = self.custom_function.__name__
            else:
                self.name = self.type.name

        penalty_type = self.type.get_type()
        if self.node == Node.TRANSITION:
            all_pn = []

            # Make sure the penalty behave like a PhaseTransition, even though it may be an Objective or Constraint
            self.node = Node.END
            self.node_idx = [0]
            self.transition = True
            self.dt = 1
            self.phase_pre_idx = nlp.phase_idx
            self.phase_post_idx = (nlp.phase_idx + 1) % ocp.n_phases

            all_pn.append(self._get_penalty_node_list(ocp, nlp))
            all_pn[0].u = [nlp.U[-1]]  # Make an exception to the fact that U is not available for the last node

            nlp = ocp.nlp[(nlp.phase_idx + 1) % ocp.n_phases]
            self.node = Node.START
            all_pn.append(self._get_penalty_node_list(ocp, nlp))

            self.node = Node.TRANSITION

            penalty_type.validate_penalty_time_index(self, all_pn[0])
            penalty_type.validate_penalty_time_index(self, all_pn[1])
            self.clear_penalty(ocp, all_pn[0].nlp)

        else:
            all_pn = self._get_penalty_node_list(ocp, nlp)
            penalty_type.validate_penalty_time_index(self, all_pn)
            self.clear_penalty(all_pn.ocp, all_pn.nlp)
            self.dt = penalty_type.get_dt(all_pn.nlp)
            self.node_idx = all_pn.t

        penalty_function = self.type.value[0](self, all_pn, **self.params)
        self.set_penalty(penalty_function, all_pn)

    def _add_penalty_to_pool(self, all_pn: PenaltyNodeList):
        """
        Return the penalty pool for the specified penalty (abstract)

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements
        """

        raise RuntimeError("get_dt cannot be called from an abstract class")

    def clear_penalty(self, ocp, nlp):
        """
        Resets a penalty. A negative penalty index creates a new empty penalty (abstract)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        raise RuntimeError("_reset_penalty cannot be called from an abstract class")

    def _get_penalty_node_list(self, ocp, nlp) -> PenaltyNodeList:
        """
        Get the actual node (time, X and U) specified in the penalty

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp

        Returns
        -------
        The actual node (time, X and U) specified in the penalty
        """

        if not isinstance(self.node, (list, tuple)):
            self.node = (self.node,)

        t = []
        for node in self.node:
            if isinstance(node, int):
                if node < 0 or node > nlp.ns:
                    raise RuntimeError(f"Invalid node, {node} must be between 0 and {nlp.ns}")
                t.append(node)
            elif node == Node.START:
                t.append(0)
            elif node == Node.MID:
                if nlp.ns % 2 == 1:
                    raise (ValueError("Number of shooting points must be even to use MID"))
                t.append(nlp.ns // 2)
            elif node == Node.INTERMEDIATES:
                t.extend(list(i for i in range(1, nlp.ns - 1)))
            elif node == Node.PENULTIMATE:
                if nlp.ns < 2:
                    raise (ValueError("Number of shooting points must be greater than 1"))
                t.append(nlp.ns - 1)
            elif node == Node.END:
                t.append(nlp.ns)
            elif node == Node.ALL_SHOOTING:
                t.extend(range(nlp.ns))
            elif node == Node.ALL:
                t.extend(range(nlp.ns + 1))
            else:
                raise RuntimeError(" is not a valid node")

        x = [nlp.X[idx] for idx in t]
        u = [nlp.U[idx] for idx in t if idx != nlp.ns]
        return PenaltyNodeList(ocp, nlp, t, x, u, nlp.parameters.cx)
