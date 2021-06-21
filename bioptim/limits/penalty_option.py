from typing import Any, Union, Callable

import biorbd
from casadi import Function, MX, SX
import numpy as np

from .penalty_node import PenaltyNodeList
from ..misc.enums import Node, PlotType
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
    index: list
        The component index the penalty is acting on
    target: np.array(target)
        A target to track for the penalty
    sliced_target: np.array(target)
        The sliced version of the target to track, this is the one actually tracked
    custom_function: Callable
        A user defined function to call to get the penalty
    derivative: bool
        If the minimization is applied on the numerical derivative of the state
    """

    def __init__(
        self,
        penalty: Any,
        phase: int = 0,
        node: Node = Node.DEFAULT,
        get_all_nodes_at_once: bool = False,
        target: np.ndarray = None,
        quadratic: bool = None,
        derivative: bool = False,
        index: list = None,
        rows: Union[list, tuple, range, np.ndarray] = None,
        cols: Union[list, tuple, range, np.ndarray] = None,
        custom_function: Callable = None,
        **params: Any,
    ):
        """
        Parameters
        ----------
        penalty: PenaltyType
            The actual penalty
        phase: int
            The phase the penalty is acting on
        node: Node
            The node within a phase on which the penalty is acting on
        target: np.ndarray
            A target to track for the penalty
        quadratic: bool
            If the penalty is quadratic
        derivative: bool
            If the minimization is applied on the numerical derivative of the state
        index: int
            The component index the penalty is acting on
        custom_function: Callable
            A user defined function to call to get the penalty
        **params: dict
            Generic parameters for the penalty
        """

        super(PenaltyOption, self).__init__(phase=phase, type=penalty, **params)
        self.node = node
        self.get_all_nodes_at_once = get_all_nodes_at_once
        self.quadratic = quadratic

        if index is not None and rows is not None:
            raise ValueError("rows and index cannot be defined simultaneously since they are the same variable")
        self.rows = rows if rows is not None else index
        self.cols = cols

        self.target = np.array(target) if np.any(target) else None
        if self.target is not None:
            if len(self.target.shape) == 0:
                self.target = self.target[np.newaxis]
            if len(self.target.shape) == 1:
                self.target = self.target[:, np.newaxis]

        self.custom_function = custom_function

        self.node_idx = []
        self.dt = 0
        self.function: Union[Function, None] = None
        self.weighted_function: Union[Function, None] = None
        self.derivative = derivative

    def set_penalty(
            self, penalty: Union[MX, SX], all_pn: PenaltyNodeList, combine_to: str = None, target_ns: int = -1
    ):
        """
        Prepare the dimension and index of the penalty (including the target)

        Parameters
        ----------
        penalty: Union[MX, SX],
            The actual penalty function
        target_ns: Union[list, tuple]
            The expected shape (n_rows, ns) of the data to track
        all_pn: PenaltyNodeList
            The penalty node elements
        combine_to: str
            The name of the underlying plot to combine the tracking data to

        """
        self.rows = self._set_dim_idx(self.rows, penalty.rows())
        self.cols = self._set_dim_idx(self.cols, penalty.columns())
        if self.target is not None:
            self._check_target_dimensions(target_ns)
            if combine_to is not None:
                self.add_target_to_plot(all_pn, combine_to)
        self._set_penalty_function(all_pn, penalty)

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

    def _check_target_dimensions(self, ns: int):
        """
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay

        Parameters
        ----------
        ns: Union[list, tuple]
            The expected shape (n_rows, ns) of the data to track
        """

        if len(self.target.shape) == 1:
            raise RuntimeError(
                f"target cannot be a vector (it can be a matrix with time dimension equals to 1 though)"
            )
        if self.target.shape[1] == 1:
            self.target = np.repeat(self.target, ns, axis=1)

        if self.target.shape != (len(self.rows), ns):
            raise RuntimeError(
                f"target {self.target.shape} does not correspond to expected size {(len(self.rows), ns)}"
            )

    def _set_penalty_function(self, all_pn: PenaltyNodeList, fcn: Union[MX, SX]):
        param_cx = all_pn.nlp.cx(all_pn.nlp.parameters.cx)

        # Do not use nlp.add_casadi_func because all functions must be registered
        self.function = biorbd.to_casadi_func(
            f"{self.name}", fcn[self.rows, self.cols], all_pn.nlp.states.cx, all_pn.nlp.controls.cx, param_cx,
        )

        state_cx, control_cx = (all_pn.nlp.states._cx, all_pn.nlp.controls._cx) if self.derivative else (all_pn.nlp.states.cx, all_pn.nlp.controls.cx)

        if self.derivative:
            self.function = biorbd.to_casadi_func(
                f"{self.name}",
                self.function(all_pn.nlp.states.cx_end, all_pn.nlp.controls.cx_end, param_cx) - self.function(all_pn.nlp.states.cx, all_pn.nlp.controls.cx, param_cx),
                state_cx,
                control_cx,
                param_cx,
            )

        modified_fcn = self.function(state_cx, control_cx, param_cx)

        dt_cx = all_pn.nlp.cx.sym("dt", 1, 1)
        weight_cx = all_pn.nlp.cx.sym("weight", 1, 1)
        target_cx = all_pn.nlp.cx.sym("target", modified_fcn.shape)

        modified_fcn = modified_fcn - target_cx
        modified_fcn = modified_fcn ** 2 if self.quadratic else modified_fcn

        self.weighted_function = Function(  # Do not use nlp.add_casadi_func because all of them must be registered
            f"{self.name}",
            [state_cx, control_cx, param_cx, weight_cx, target_cx, dt_cx],
            [weight_cx * modified_fcn * dt_cx]
        ).expand()

    def add_target_to_plot(self, all_pn: PenaltyNodeList, combine_to: str, rows: Union[range, list, tuple, np.ndarray] = None, axes_index: Union[range, list, tuple, np.ndarray] = None):
        """
        Interface to the plot so it can be properly added to the proper plot

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements
        combine_to: str
            The name of the underlying plot to combine the tracking data to
        rows: Union[range, list, tuple, np.ndarray]
            The rows of the target to plot
        axes_index: Union[range, list, tuple, np.ndarray]
            The position of the target to plot, self.rows is used if None
        """

        data = np.c_[self.target, self.target[:, -1]] if self.target.shape[1] == all_pn.nlp.ns else self.target
        axes_index = self.rows if axes_index is None else axes_index
        if rows is not None:
            data = data[rows, :]
        all_pn.ocp.add_plot(
            combine_to,
            lambda x, u, p: data,
            color="tab:red",
            linestyle=".-",
            plot_type=PlotType.STEP,
            phase=all_pn.nlp.phase_idx,
            axes_idx=Mapping(axes_index)
        )

    def add_multiple_target_to_plot(self, names: list, suffix: str, all_pn: PenaltyNodeList):
        """
        Easy plot adder for multiple values

        Parameters
        ----------
        names: list
            The list of names in optim_var to add
        suffix: str
            The suffix name to add (either "states" or "controls").
            This will determine if nlp.states or nlp.controls is used
        all_pn: PenaltyNodeList
            The penalty node elements
        """
        if self.target is None:
            return

        offset = 0
        if suffix == "states":
            optim_var = all_pn.nlp.states
        elif suffix == "controls":
            optim_var = all_pn.nlp.controls
        else:
            raise ValueError("suffix for add_multiple_target_to_plot can only be 'states' or 'controls'")
        for name in names:
            n_elt = self.rows.shape[0]
            self.add_target_to_plot(all_pn, combine_to=f"{name}_{suffix}", rows=range(offset, offset + n_elt), axes_index=self.rows)
            offset += n_elt

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
        penalty_type.adjust_penalty_parameters(self)
        if self.node == Node.TRANSITION:
            all_pn = []
            self.node = Node.END
            all_pn.append(self._get_penalty_node_list(ocp, nlp))
            # u = [nlp.U[-1]]  # Make an exception to the fact that U is not available for the last node

            nlp = ocp.nlp[(nlp.phase_idx + 1) % ocp.n_phases]
            self.node = Node.START
            all_pn.append(self._get_penalty_node_list(ocp, nlp))

            self.node = Node.TRANSITION
            self.get_all_nodes_at_once = True

            penalty_type.validate_penalty_time_index(self, all_pn[0])
            penalty_type.validate_penalty_time_index(self, all_pn[1])
            penalty_type.clear_penalty(ocp, None, self)
        else:
            all_pn = self._get_penalty_node_list(ocp, nlp)
            penalty_type.validate_penalty_time_index(self, all_pn)
            penalty_type.clear_penalty(all_pn.ocp, all_pn.nlp, self)

        self.dt = penalty_type.get_dt(all_pn.nlp)
        self.node_idx = all_pn.t
        self.type.value[0](self, all_pn, **self.params)
        penalty_type.get_penalty_pool(all_pn)[self.list_index] = self

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
                t.append(i for i in range(1, nlp.ns - 1))
            elif node == Node.PENULTIMATE:
                if nlp.ns < 2:
                    raise (ValueError("Number of shooting points must be greater than 1"))
                t.append(nlp.ns - 1)
            elif node == Node.END:
                t.append(nlp.ns)
            elif node == Node.ALL:
                t.extend([i for i in range(nlp.ns + 1)])
            else:
                raise RuntimeError(" is not a valid node")
        x = [nlp.X[idx] for idx in t]
        u = [nlp.U[idx] for idx in t if idx != nlp.ns]
        return PenaltyNodeList(ocp, nlp, t, x, u, nlp.parameters.cx)
