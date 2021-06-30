from typing import Any, Union, Callable

import biorbd
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
    index: list
        The component index the penalty is acting on
    target: np.array(target)
        A target to track for the penalty
    custom_function: Callable
        A user defined function to call to get the penalty
    derivative: bool
        If the minimization is applied on the numerical derivative of the state
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
        index: list = None,
        rows: Union[list, tuple, range, np.ndarray] = None,
        cols: Union[list, tuple, range, np.ndarray] = None,
        custom_function: Callable = None,
        is_internal: bool = False,
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
        self.node = node
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
        self.weight = weight
        self.function: Union[Function, None] = None
        self.weighted_function: Union[Function, None] = None
        self.derivative = derivative
        self.explicit_derivative = explicit_derivative
        self.transition = False
        if self.derivative and self.explicit_derivative:
            raise ValueError("derivative and explicit_derivative cannot be both True")
        self.is_internal = is_internal

        self.multi_thread = False

    def set_penalty(
            self, penalty: Union[MX, SX], all_pn: PenaltyNodeList, combine_to: str = None, target_ns: int = -1, expand: bool = True, plot_target: bool = True
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
        expand: bool
            If the penalty function should be expanded
        plot_target: bool
            If there is a target, it can be automatically added to plot, assuming the target is 2-dimensions
        """

        self.rows = self._set_dim_idx(self.rows, penalty.rows())
        self.cols = self._set_dim_idx(self.cols, penalty.columns())
        if self.target is not None and plot_target:
            self._check_target_dimensions(all_pn, target_ns)
            if combine_to is not None:
                self.add_target_to_plot(all_pn, combine_to)
        self._set_penalty_function(all_pn, penalty, expand)

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

    def _check_target_dimensions(self, all_pn: PenaltyNodeList, n_col_expected: int):
        """
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements
        n_col_expected: Union[list, tuple]
            The expected shape (n_rows, ns) of the data to track
        """

        if len(self.target.shape) != 2:
            raise RuntimeError(
                f"target cannot be a vector (it can be a matrix with time dimension equals to 1 though)"
            )
        if self.target.shape[1] == 1:
            self.target = np.repeat(self.target, n_col_expected, axis=1)

        if self.target.shape != (len(self.rows), n_col_expected):
            raise RuntimeError(
                f"target {self.target.shape} does not correspond to expected size {(len(self.rows), n_col_expected)}"
            )

        # If the target is on controls and control is constant, there will be one value missing
        if all_pn is not None:
            if all_pn.nlp.control_type == ControlType.CONSTANT and all_pn.nlp.ns in all_pn.t and self.target.shape[1] == all_pn.nlp.ns:
                if all_pn.t[-1] != all_pn.nlp.ns:
                    raise NotImplementedError("Modifying target for END not being last is not implemented yet")
                self.target = np.concatenate((self.target, np.nan * np.zeros((self.target.shape[0], 1))), axis=1)

    def _set_penalty_function(self, all_pn: [PenaltyNodeList, list, tuple], fcn: Union[MX, SX], expand: bool = True):
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
            state_cx = all_pn.nlp.states.cx
            control_cx = all_pn.nlp.controls.cx
            if self.explicit_derivative:
                state_cx = horzcat(state_cx, all_pn.nlp.states.cx_end)
                control_cx = horzcat(control_cx, all_pn.nlp.controls.cx_end)

        param_cx = nlp.cx(nlp.parameters.cx)

        # Do not use nlp.add_casadi_func because all functions must be registered
        self.function = biorbd.to_casadi_func(
            name, fcn[self.rows, self.cols], state_cx, control_cx, param_cx, expand=expand
        )

        # if self.integration:
        #     if all_pn.nlp.integration == TRAPEZE
        #         self.function = biorbd.to_casadi_func(
        #             f"{name}",
        #             (self.function(all_pn.nlp.states.cx_end, all_pn.nlp.controls.cx_end, param_cx) + self.function(
        #                 state_cx, control_cx, param_cx))/2,
        #             state_cx,
        #             control_cx,
        #             param_cx,
        #         )

        if self.derivative:
            state_cx = horzcat(all_pn.nlp.states.cx_end, all_pn.nlp.states.cx)
            control_cx = horzcat(all_pn.nlp.controls.cx_end, all_pn.nlp.controls.cx)
            self.function = biorbd.to_casadi_func(
                f"{name}",
                self.function(all_pn.nlp.states.cx_end, all_pn.nlp.controls.cx_end, param_cx) - self.function(all_pn.nlp.states.cx, all_pn.nlp.controls.cx, param_cx),
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

        if self.multi_thread:
            self.weighted_function = self.weighted_function.map(nlp.ns, "thread", ocp.n_threads)

        if expand:
            self.weighted_function.expand()

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

        self.type.value[0](self, all_pn, **self.params)
        self.get_penalty_pool(all_pn)[self.list_index] = self

    def get_penalty_pool(self, all_pn: PenaltyNodeList):
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
                t.append(i for i in range(1, nlp.ns - 1))
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
