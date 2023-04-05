from typing import Any, Callable

import biorbd_casadi as biorbd
from casadi import horzcat, vertcat, Function, MX, SX
import numpy as np

from .penalty_node import PenaltyNodeList
from ..misc.enums import Node, PlotType, ControlType, PenaltyType, IntegralApproximation
from ..misc.mapping import Mapping, BiMapping
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
    rows: list | tuple | range | np.ndarray
        The index of the rows in the penalty to keep
    cols: list | tuple | range | np.ndarray
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
    node_idx: list | tuple | Node
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
    integration_rule: IntegralApproximation
        The integration rule to use for the penalty
    transition: bool
        If the penalty is a transition
    phase_pre_idx: int
        The index of the nlp of pre when penalty is transition
    phase_post_idx: int
        The index of the nlp of post when penalty is transition
    penalty_type: PenaltyType
        If the penalty is from the user or from bioptim (implicit or internal)
    multi_thread: bool
        If the penalty is multithreaded

    Methods
    -------
    set_penalty(self, penalty: MX | SX, all_pn: PenaltyNodeList)
        Prepare the dimension and index of the penalty (including the target)
    _set_dim_idx(self, dim: list | tuple | range | np.ndarray, n_rows: int)
        Checks if the variable index is consistent with the requested variable.
    _check_target_dimensions(self, all_pn: PenaltyNodeList, n_time_expected: int)
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay
    _set_penalty_function(self, all_pn: PenaltyNodeList | list | tuple, fcn: MX | SX)
        Finalize the preparation of the penalty (setting function and weighted_function)
    add_target_to_plot(self, all_pn: PenaltyNodeList, combine_to: str)
        Interface to the plot so it can be properly added to the proper plot
    _finish_add_target_to_plot(self, all_pn: PenaltyNodeList)
        Internal interface to add (after having check the target dimensions) the target to the plot if needed
    add_or_replace_to_penalty_pool(self, ocp, nlp)
        Doing some configuration on the penalty and add it to the list of penalty
    _add_penalty_to_pool(self, all_pn: PenaltyNodeList)
        Return the penalty pool for the specified penalty (abstract)
    ensure_penalty_sanity(self, ocp, nlp)
        Resets a penalty. A negative penalty index creates a new empty penalty (abstract)
    _get_penalty_node_list(self, ocp, nlp) -> PenaltyNodeList
        Get the actual node (time, X and U) specified in the penalty
    """

    def __init__(
        self,
        penalty: Any,
        phase: int = 0,
        node: Node | list | tuple = Node.DEFAULT,
        target: int | float | np.ndarray | list[int] | list[float] | list[np.ndarray] = None,
        quadratic: bool = None,
        weight: float = 1,
        derivative: bool = False,
        explicit_derivative: bool = False,
        integrate: bool = False,
        integration_rule: IntegralApproximation = IntegralApproximation.DEFAULT,
        index: list = None,
        rows: list | tuple | range | np.ndarray = None,
        cols: list | tuple | range | np.ndarray = None,
        states_mapping: BiMapping = None,
        custom_function: Callable = None,
        penalty_type: PenaltyType = PenaltyType.USER,
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
        node: Node | list | tuple
            The node within a phase on which the penalty is acting on
        target: int | float | np.ndarray | list[int] | list[float] | list[np.ndarray]
            A target to track for the penalty
        quadratic: bool
            If the penalty is quadratic
        weight: float
            The weighting applied to this specific penalty
        derivative: bool
            If the function should be evaluated at X and X+1
        explicit_derivative: bool
            If the function should be evaluated at [X, X+1]
        integrate: bool
            If the function should be integrated
        integration_rule: IntegralApproximation
            The rule to use for the integration
        index: int
            The component index the penalty is acting on
        custom_function: Callable
            A user defined function to call to get the penalty
        penalty_type: PenaltyType
            If the penalty is from the user or from bioptim (implicit or internal)
        **params: dict
            Generic parameters for the penalty
        """

        super(PenaltyOption, self).__init__(phase=phase, type=penalty, **params)
        self.node: Node | list | tuple = node
        self.quadratic = quadratic
        self.integration_rule = integration_rule

        if index is not None and rows is not None:
            raise ValueError("rows and index cannot be defined simultaneously since they are the same variable")
        self.rows = rows if rows is not None else index
        self.cols = cols
        self.expand = expand

        self.target = None
        if target is not None:
            target = np.array(target)
            if isinstance(target, int) or isinstance(target, float) or isinstance(target, np.ndarray):
                target = [target]
            self.target = []
            for t in target:
                self.target.append(np.array(t))
                if len(self.target[-1].shape) == 0:
                    self.target[-1] = self.target[-1][np.newaxis]
                if len(self.target[-1].shape) == 1:
                    self.target[-1] = self.target[-1][:, np.newaxis]
            if len(self.target) == 1 and (
                self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                or self.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
            ):
                if self.node == Node.ALL or self.node == Node.DEFAULT:
                    self.target = [self.target[0][:, :-1], self.target[0][:, 1:]]
                else:
                    raise NotImplementedError(
                        f"A list of 2 elements is required with {self.node} and TRAPEZOIDAL Integration"
                        f"except for Node.NODE_ALL and Node.NODE_DEFAULT"
                        "which can be automatically generated"
                    )

        self.target_plot_name = None
        self.target_to_plot = None
        # todo: not implemented yet for trapezoidal integration
        self.plot_target = (
            False
            if (
                self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                or self.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
            )
            else True
        )

        self.states_mapping = states_mapping

        self.custom_function = custom_function

        self.node_idx = []
        self.dt = 0
        self.weight = weight
        self.function: Function | None = None
        self.function_non_threaded: Function | None = None
        self.weighted_function: Function | None = None
        self.weighted_function_non_threaded: Function | None = None
        self.derivative = derivative
        self.explicit_derivative = explicit_derivative
        self.integrate = integrate
        self.transition = False
        self.binode_constraint = False
        self.allnode_constraint = False
        self.phase_pre_idx = None
        self.phase_post_idx = None
        if self.derivative and self.explicit_derivative:
            raise ValueError("derivative and explicit_derivative cannot be both True")
        self.penalty_type = penalty_type

        self.multi_thread = multi_thread

    def set_penalty(self, penalty: MX | SX, all_pn: PenaltyNodeList):
        """
        Prepare the dimension and index of the penalty (including the target)

        Parameters
        ----------
        penalty: MX | SX,
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

    def _set_dim_idx(self, dim: list | tuple | range | np.ndarray, n_rows: int):
        """
        Checks if the variable index is consistent with the requested variable.

        Parameters
        ----------
        dim: list | tuple | range | np.ndarray
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
        n_time_expected: int
            The expected shape (n_rows, ns) of the data to track
        """

        if self.integration_rule == IntegralApproximation.RECTANGLE:
            n_dim = len(self.target[0].shape)
            if n_dim != 2 and n_dim != 3:
                raise RuntimeError(
                    f"target cannot be a vector (it can be a matrix with time dimension equals to 1 though)"
                )
            if self.target[0].shape[-1] == 1:
                if len(self.rows) == 1:
                    self.target[0] = np.reshape(self.target[0], (1, len(self.target[0])))
                else:
                    self.target = np.repeat(self.target, n_time_expected, axis=-1)

            shape = (
                (len(self.rows), n_time_expected) if n_dim == 2 else (len(self.rows), len(self.cols), n_time_expected)
            )
            if self.target[0].shape != shape:
                raise RuntimeError(
                    f"target {self.target[0].shape} does not correspond to expected size {shape} for penalty {self.name}"
                )

            # If the target is on controls and control is constant, there will be one value missing
            if all_pn is not None:
                if (
                    all_pn.nlp.control_type == ControlType.CONSTANT
                    and all_pn.nlp.ns in all_pn.t
                    and self.target[0].shape[-1] == all_pn.nlp.ns
                ):
                    if all_pn.t[-1] != all_pn.nlp.ns:
                        raise NotImplementedError("Modifying target for END not being last is not implemented yet")
                    self.target[0] = np.concatenate(
                        (self.target[0], np.nan * np.zeros((self.target[0].shape[0], 1))), axis=1
                    )
        elif (
            self.integration_rule == IntegralApproximation.TRAPEZOIDAL
            or self.integration_rule == IntegralApproximation.TRAPEZOIDAL
        ):
            target_dim = len(self.target)
            if target_dim != 2:
                raise RuntimeError(f"targets with trapezoidal integration rule need to get a list of two elements.")

            for target in self.target:
                n_dim = len(target.shape)
                if n_dim != 2 and n_dim != 3:
                    raise RuntimeError(
                        f"target cannot be a vector (it can be a matrix with time dimension equals to 1 though)"
                    )
                if target.shape[-1] == 1:
                    target = np.repeat(target, n_time_expected, axis=-1)

            shape = (
                (len(self.rows), n_time_expected - 1)
                if n_dim == 2
                else (len(self.rows), len(self.cols), n_time_expected - 1)
            )

            for target in self.target:
                if target.shape != shape:
                    raise RuntimeError(
                        f"target {target.shape} does not correspond to expected size {shape} for penalty {self.name}"
                    )

            # If the target is on controls and control is constant, there will be one value missing
            if all_pn is not None:
                if (
                    all_pn.nlp.control_type == ControlType.CONSTANT
                    and all_pn.nlp.ns in all_pn.t
                    and self.target[0].shape[-1] == all_pn.nlp.ns - 1
                    and self.target[1].shape[-1] == all_pn.nlp.ns - 1
                ):
                    if all_pn.t[-1] != all_pn.nlp.ns:
                        raise NotImplementedError("Modifying target for END not being last is not implemented yet")
                    self.target = np.concatenate((self.target, np.nan * np.zeros((self.target.shape[0], 1))), axis=1)

    def _set_penalty_function(self, all_pn: PenaltyNodeList | list | tuple, fcn: MX | SX):
        """
        Finalize the preparation of the penalty (setting function and weighted_function)

        Parameters
        ----------
        all_pn: PenaltyNodeList | list | tuple
            The nodes
        fcn: MX | SX
            The value of the penalty function
        """

        # Sanity checks
        if self.transition and self.explicit_derivative:
            raise ValueError("transition and explicit_derivative cannot be true simultaneously")
        if self.transition and self.derivative:
            raise ValueError("transition and derivative cannot be true simultaneously")
        if self.derivative and self.explicit_derivative:
            raise ValueError("derivative and explicit_derivative cannot be true simultaneously")

        def get_u(nlp, u: MX | SX, dt: MX | SX):
            """
            Get the control at a given time

            Parameters
            ----------
            nlp: NonlinearProgram
                The nonlinear program
            u: MX | SX
                The control matrix
            dt: MX | SX
                The time a which control should be computed

            Returns
            -------
            The control at a given time
            """

            if nlp.control_type == ControlType.CONSTANT:
                return u
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                return u[:, 0] + (u[:, 1] - u[:, 0]) * dt
            else:
                raise RuntimeError(f"{nlp.control_type} ControlType not implemented yet")

        if self.binode_constraint or self.transition:
            ocp = all_pn[0].ocp
            nlp = all_pn[0].nlp
            nlp_post = all_pn[1].nlp
            name = self.name.replace("->", "_").replace(" ", "_").replace(",", "_")
            states_pre_scaled = nlp.states[0]["scaled"].cx_end  # TODO: [0] to [node_index]
            states_post_scaled = nlp_post.states[0]["scaled"].cx_start  # TODO: [0] to [node_index]
            controls_pre_scaled = nlp.controls[0]["scaled"].cx_end  # TODO: [0] to [node_index]
            controls_post_scaled = nlp_post.controls[0]["scaled"].cx_start  # TODO: [0] to [node_index]
            state_cx_scaled = vertcat(states_pre_scaled, states_post_scaled)
            control_cx_scaled = vertcat(controls_pre_scaled, controls_post_scaled)

        elif self.allnode_constraint:
            ocp = all_pn.ocp
            nlp = all_pn[0].nlp
            nlp_all = all_pn.nlp
            name = self.name.replace("->", "_").replace(" ", "_").replace(",", "_")
            states_all_scaled = nlp_all.states[0]["scaled"].cx_start    # TODO: [0] to [node_index]
            controls_all_scaled = nlp_all.controls[0]["scaled"].cx_start    # TODO: [0] to [node_index]
            state_cx_scaled = vertcat(states_all_scaled)
            control_cx_scaled = vertcat(controls_all_scaled)

        else:
            ocp = all_pn.ocp
            nlp = all_pn.nlp
            name = self.name
            if self.integrate:
                state_cx_scaled = horzcat(
                    *([all_pn.nlp.states[0]["scaled"].cx_start] + all_pn.nlp.states[0]["scaled"].cx_intermediates_list) # TODO: [0] to [node_index]
                )
                control_cx_scaled = all_pn.nlp.controls[0]["scaled"].cx_start  # TODO: [0] to [node_index]
            else:
                state_cx_scaled = all_pn.nlp.states[0]["scaled"].cx_start   # TODO: [0] to [node_index]
                control_cx_scaled = all_pn.nlp.controls[0]["scaled"].cx_start   # TODO: [0] to [node_index]
            if self.explicit_derivative:
                if self.derivative:
                    raise RuntimeError("derivative and explicit_derivative cannot be simultaneously true")
                state_cx_scaled = horzcat(state_cx_scaled, all_pn.nlp.states[0]["scaled"].cx_end)   # TODO: [0] to [node_index]
                control_cx_scaled = horzcat(control_cx_scaled, all_pn.nlp.controls[0]["scaled"].cx_end) # TODO: [0] to [node_index]

        param_cx = nlp.cx(nlp.parameters.cx)

        # Do not use nlp.add_casadi_func because all functions must be registered
        sub_fcn = fcn[self.rows, self.cols]
        self.function = nlp.to_casadi_func(
            name, sub_fcn, state_cx_scaled, control_cx_scaled, param_cx, expand=self.expand
        )
        self.function_non_threaded = self.function

        if self.derivative:
            state_cx_scaled = horzcat(all_pn.nlp.states[0]["scaled"].cx_end, all_pn.nlp.states[0]["scaled"].cx_start) # TODO: [0] to [node_index]
            control_cx_scaled = horzcat(all_pn.nlp.controls[0]["scaled"].cx_end, all_pn.nlp.controls[0]["scaled"].cx_start)   # TODO: [0] to [node_index]
            self.function = biorbd.to_casadi_func(
                f"{name}",
                self.function(all_pn.nlp.states[0]["scaled"].cx_end, all_pn.nlp.controls[0]["scaled"].cx_end, param_cx) # TODO: [0] to [node_index]
                - self.function(all_pn.nlp.states[0]["scaled"].cx_start, all_pn.nlp.controls[0]["scaled"].cx_start, param_cx),   # TODO: [0] to [node_index]
                state_cx_scaled,
                control_cx_scaled,
                param_cx,
            )

        dt_cx = nlp.cx.sym("dt", 1, 1)
        is_trapezoidal = (
            self.integration_rule == IntegralApproximation.TRAPEZOIDAL
            or self.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
        )
        target_shape = tuple(
            [
                len(self.rows),
                len(self.cols) + 1 if is_trapezoidal else len(self.cols),
            ]
        )
        target_cx = nlp.cx.sym("target", target_shape)
        weight_cx = nlp.cx.sym("weight", 1, 1)
        exponent = 2 if self.quadratic and self.weight else 1

        if is_trapezoidal:
            # Hypothesis: the function is continuous on states
            # it neglects the discontinuities at the beginning of the optimization
            state_cx_scaled = (
                horzcat(all_pn.nlp.states[0]["scaled"].cx_start, all_pn.nlp.states[0]["scaled"].cx_end)   # TODO: [0] to [node_index]
                if self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                else all_pn.nlp.states[0]["scaled"].cx_start    # TODO: [0] to [node_index]
            )
            state_cx = (
                horzcat(all_pn.nlp.states[0].cx_start, all_pn.nlp.states[0].cx_end)   # TODO: [0] to [node_index]
                if self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                else all_pn.nlp.states[0].cx_start  # TODO: [0] to [node_index]
            )
            # to handle piecewise constant in controls we have to compute the value for the end of the interval
            # which only relies on the value of the control at the beginning of the interval
            control_cx_scaled = (
                horzcat(all_pn.nlp.controls[0]["scaled"].cx_start)    # TODO: [0] to [node_index]
                if nlp.control_type == ControlType.CONSTANT
                else horzcat(all_pn.nlp.controls[0]["scaled"].cx_start, all_pn.nlp.controls[0]["scaled"].cx_end)    # TODO: [0] to [node_index]
            )
            control_cx = (
                horzcat(all_pn.nlp.controls[0].cx_start)    # TODO: [0] to [node_index]
                if nlp.control_type == ControlType.CONSTANT
                else horzcat(all_pn.nlp.controls[0].cx_start, all_pn.nlp.controls[0].cx_end)  # TODO: [0] to [node_index]
            )
            control_cx_end_scaled = get_u(nlp, control_cx_scaled, dt_cx)
            control_cx_end = get_u(nlp, control_cx, dt_cx)
            state_cx_end_scaled = (
                all_pn.nlp.states[0]["scaled"].cx_end   # TODO: [0] to [node_index]
                if self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                else nlp.dynamics[0](x0=state_cx, p=control_cx_end, params=nlp.parameters.cx)["xf"]
            )
            self.modified_function = nlp.to_casadi_func(
                f"{name}",
                (
                    (
                        self.function(all_pn.nlp.states[0]["scaled"].cx_start, all_pn.nlp.controls[0]["scaled"].cx_start, param_cx) # TODO: [0] to [node_index]
                        - target_cx[:, 0]
                    )
                    ** exponent
                    + (self.function(state_cx_end_scaled, control_cx_end_scaled, param_cx) - target_cx[:, 1])
                    ** exponent
                )
                / 2,
                state_cx_scaled,
                control_cx_scaled,
                param_cx,
                target_cx,
                dt_cx,
            )
            modified_fcn = self.modified_function(state_cx_scaled, control_cx_scaled, param_cx, target_cx, dt_cx)
        else:
            modified_fcn = (self.function(state_cx_scaled, control_cx_scaled, param_cx) - target_cx) ** exponent

        # for the future bioptim adventurer: here lies the reason that a constraint must have weight = 0.
        modified_fcn = weight_cx * modified_fcn * dt_cx if self.weight else modified_fcn * dt_cx

        # Do not use nlp.add_casadi_func because all of them must be registered
        self.weighted_function = Function(
            name, [state_cx_scaled, control_cx_scaled, param_cx, weight_cx, target_cx, dt_cx], [modified_fcn]
        )
        self.weighted_function_non_threaded = self.weighted_function

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
        # if the target is n x ns, we need to add a dimension (n x ns + 1) to make it compatible with the plot
        if self.target[0].shape[1] == all_pn.nlp.ns:
            self.target_to_plot = np.concatenate(
                (self.target[0], np.nan * np.ndarray((self.target[0].shape[0], 1))), axis=1
            )
        else:
            self.target_to_plot = self.target[0]

    def _finish_add_target_to_plot(self, all_pn: PenaltyNodeList):
        """
        Internal interface to add (after having check the target dimensions) the target to the plot if needed

        Parameters
        ----------
        all_pn: PenaltyNodeList
            The penalty node elements

        """

        def plot_function(t, x, u, p):
            if isinstance(t, (list, tuple)):
                return self.target_to_plot[:, [self.node_idx.index(_t) for _t in t]]
            else:
                return self.target_to_plot[:, self.node_idx.index(t)]

        if self.target_to_plot is not None:
            if self.target_to_plot.shape[1] > 1:
                plot_type = PlotType.STEP
            else:
                plot_type = PlotType.POINT

            all_pn.ocp.add_plot(
                self.target_plot_name,
                plot_function,
                color="tab:red",
                plot_type=plot_type,
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
            if not self.states_mapping:
                self.states_mapping = BiMapping(range(nlp.states[0].shape), range(nlp.states[0].shape)) # TODO: [0] to [node_index]

            all_pn.append(self._get_penalty_node_list(ocp, nlp))
            all_pn[0].u = [nlp.U[-1]]  # Make an exception to the fact that U is not available for the last node

            nlp = ocp.nlp[(nlp.phase_idx + 1) % ocp.n_phases]
            self.node = Node.START
            all_pn.append(self._get_penalty_node_list(ocp, nlp))

            self.node = Node.TRANSITION

            penalty_type.validate_penalty_time_index(self, all_pn[0])
            penalty_type.validate_penalty_time_index(self, all_pn[1])
            self.ensure_penalty_sanity(ocp, all_pn[0].nlp)

        elif isinstance(self.node, tuple) and self.binode_constraint:
            all_pn = []
            self.node_list = self.node
            # Make sure the penalty behave like a BinodeConstraint, even though it may be an Objective or Constraint
            # self.transition = True
            self.dt = 1
            # self.phase_pre_idx
            # self.phase_post_idx = (nlp.phase_idx + 1) % ocp.n_phases
            if not self.states_mapping:
                self.states_mapping = BiMapping(range(nlp.states[0].shape), range(nlp.states[0].shape)) # TODO: [0] to [node_index]
            self.node = self.node_list[0]
            nlp = ocp.nlp[self.phase_first_idx]
            all_pn.append(self._get_penalty_node_list(ocp, nlp))
            if self.node == Node.END:
                all_pn[0].u = [nlp.U[-1]]
                # Make an exception to the fact that U is not available for the last node

            self.node = self.node_list[1]
            nlp = ocp.nlp[self.phase_second_idx]
            all_pn.append(self._get_penalty_node_list(ocp, nlp))
            if self.node == Node.END:
                all_pn[1].u = [nlp.U[-1]]
                # Make an exception to the fact that U is not available for the last node

            # reset the node list
            self.node = self.node_list

            penalty_type.validate_penalty_time_index(self, all_pn[0])
            penalty_type.validate_penalty_time_index(self, all_pn[1])
            self.node_idx = [all_pn[0].t[0], all_pn[1].t[0]]
            self.ensure_penalty_sanity(ocp, all_pn[0].nlp)

        elif self.allnode_constraint:   # TODO: Peut etre a changer
            #all_pn = []
            # Make sure the penalty behave like a BinodeConstraint, even though it may be an Objective or Constraint
            # self.transition = True
            self.dt = 1
            #if not self.states_mapping:
            #    self.states_mapping = BiMapping(range(nlp.states.shape), range(nlp.states.shape))
            nlp = ocp.nlp[self.phase_idx]
            #all_pn.append(self._get_penalty_node_list(ocp, nlp))
            all_pn = self._get_penalty_node_list(ocp, nlp)
            penalty_type.validate_penalty_time_index(self, all_pn)
            # self.node_idx = [all_pn[0].t[0]] # t?
            penalty_type.validate_penalty_time_index(self, all_pn)
            self.node_idx = all_pn.t
            self.ensure_penalty_sanity(ocp, all_pn.nlp)

        else:
            all_pn = self._get_penalty_node_list(ocp, nlp)
            penalty_type.validate_penalty_time_index(self, all_pn)
            self.ensure_penalty_sanity(all_pn.ocp, all_pn.nlp)
            # self.dt = penalty_type.get_dt(all_pn.nlp) # TODO: A remettre
            self.dt = 1
            self.node_idx = (
                all_pn.t[:-1]
                if (
                    self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                    or self.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                )
                and self.target is not None
                else all_pn.t
            )

        penalty_function = self.type(self, all_pn, **self.params)
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

    def ensure_penalty_sanity(self, ocp, nlp):
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
        x_scaled = [nlp.X_scaled[idx] for idx in t]
        u = [nlp.U[idx] for idx in t if idx != nlp.ns]
        u_scaled = [nlp.U_scaled[idx] for idx in t if idx != nlp.ns]
        return PenaltyNodeList(ocp, nlp, t, x, u, x_scaled, u_scaled, nlp.parameters.cx)    #nlp.parameters.cx[0]
