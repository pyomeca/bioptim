from typing import Any, Callable

import biorbd_casadi as biorbd
from casadi import horzcat, vertcat, Function, MX, SX
import numpy as np

from .penalty_controller import PenaltyController
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
    set_penalty(self, penalty: MX | SX, controller: PenaltyController)
        Prepare the dimension and index of the penalty (including the target)
    _set_dim_idx(self, dim: list | tuple | range | np.ndarray, n_rows: int)
        Checks if the variable index is consistent with the requested variable.
    _check_target_dimensions(self, controller: PenaltyController, n_time_expected: int)
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay
    _set_penalty_function(self, controller: PenaltyController | list | tuple, fcn: MX | SX)
        Finalize the preparation of the penalty (setting function and weighted_function)
    add_target_to_plot(self, controller: PenaltyController, combine_to: str)
        Interface to the plot so it can be properly added to the proper plot
    _finish_add_target_to_plot(self, controller: PenaltyController)
        Internal interface to add (after having check the target dimensions) the target to the plot if needed
    add_or_replace_to_penalty_pool(self, ocp, nlp)
        Doing some configuration on the penalty and add it to the list of penalty
    _add_penalty_to_pool(self, controller: PenaltyController | list[PenaltyController, ...])
        Return the penalty pool for the specified penalty (abstract)
    ensure_penalty_sanity(self, ocp, nlp)
        Resets a penalty. A negative penalty index creates a new empty penalty (abstract)
    _get_penalty_node_list(self, ocp, nlp) -> PenaltyController
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
        self.cols_is_set = False  # This is an internal variable that is set after 'set_idx_columns' is called
        self.rows_is_set = False
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
        self.function: list[Function | None, ...] = []
        self.function_non_threaded: list[Function | None, ...] = []
        self.weighted_function: list[Function | None, ...] = []
        self.weighted_function_non_threaded: list[Function | None, ...] = []
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

    def set_penalty(self, penalty: MX | SX, controller: PenaltyController | list[PenaltyController, PenaltyController]):
        """
        Prepare the dimension and index of the penalty (including the target)

        Parameters
        ----------
        penalty: MX | SX,
            The actual penalty function
        controller: PenaltyController | list[PenaltyController, PenaltyController]
            The penalty node elements
        """

        self.rows = self._set_dim_idx(self.rows, penalty.rows())
        self.cols = self._set_dim_idx(self.cols, penalty.columns())
        if self.target is not None:
            if isinstance(controller, list):
                raise RuntimeError("Multinode constraints should call a self defined set_penalty")

            self._check_target_dimensions(controller, len(controller.t))
            if self.plot_target:
                self._finish_add_target_to_plot(controller)
        self._set_penalty_function(controller, penalty)
        self._add_penalty_to_pool(controller)

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

    def _check_target_dimensions(self, controller: PenaltyController, n_time_expected: int):
        """
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        n_time_expected: int
            The expected number of columns (n_rows, n_cols) of the data to track
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
                # A second chance the shape is correct is if the targets are declared but assume_phase_dynamics is False
                if not controller.ocp.assume_phase_dynamics and self.target[0].shape[-1] == len(self.node_idx):
                    pass
                else:
                    raise RuntimeError(
                        f"target {self.target[0].shape} does not correspond to expected size {shape} for penalty {self.name}"
                    )

            # If the target is on controls and control is constant, there will be one value missing
            if controller is not None:
                if (
                    controller.control_type == ControlType.CONSTANT
                    and controller.ns in controller.t
                    and self.target[0].shape[-1] == controller.ns
                ):
                    if controller.t[-1] != controller.ns:
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

            n_dim = None
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
                    # A second chance the shape is correct if assume_phase_dynamics is False
                    if not controller.ocp.assume_phase_dynamics and target.shape[-1] == len(self.node_idx):
                        pass
                    else:
                        raise RuntimeError(
                            f"target {target.shape} does not correspond to expected size {shape} for penalty {self.name}"
                        )

            # If the target is on controls and control is constant, there will be one value missing
            if controller is not None:
                if (
                    controller.control_type == ControlType.CONSTANT
                    and controller.ns in controller.t
                    and self.target[0].shape[-1] == controller.ns - 1
                    and self.target[1].shape[-1] == controller.ns - 1
                ):
                    if controller.t[-1] != controller.ns:
                        raise NotImplementedError("Modifying target for END not being last is not implemented yet")
                    self.target = np.concatenate((self.target, np.nan * np.zeros((self.target.shape[0], 1))), axis=1)

    def _set_penalty_function(
        self, controller: PenaltyController | list[PenaltyController, PenaltyController], fcn: MX | SX
    ):
        """
        Finalize the preparation of the penalty (setting function and weighted_function)

        Parameters
        ----------
        controller: PenaltyController | list[PenaltyController, PenaltyController]
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

        def get_u(u: MX | SX, dt: MX | SX):
            """
            Get the control at a given time

            Parameters
            ----------
            u: MX | SX
                The control matrix
            dt: MX | SX
                The time a which control should be computed

            Returns
            -------
            The control at a given time
            """

            if controller.control_type == ControlType.CONSTANT:
                return u
            elif controller.control_type == ControlType.LINEAR_CONTINUOUS:
                return u[:, 0] + (u[:, 1] - u[:, 0]) * dt
            else:
                raise RuntimeError(f"{controller.control_type} ControlType not implemented yet")

        # TODO: Add loop node_index here Benjamin
        if self.binode_constraint or self.transition:
            controllers = controller
            pre = controllers[0]
            post = controllers[1]
            controller = post  # Recast controller as a normal variable (instead of a list)

            ocp = controller.ocp
            name = self.name.replace("->", "_").replace(" ", "_").replace(",", "_")

            states_pre_scaled = pre.states_scaled.cx_end
            states_post_scaled = post.states_scaled.cx_start
            controls_pre_scaled = pre.controls_scaled.cx_end
            controls_post_scaled = post.controls_scaled.cx_start
            state_cx_scaled = vertcat(states_pre_scaled, states_post_scaled)
            control_cx_scaled = vertcat(controls_pre_scaled, controls_post_scaled)

        elif self.allnode_constraint:
            raise NotImplementedError("allnode_constraints seem to be obsolete")
            ocp = controller.ocp
            nlp_all = controller.nlp
            name = self.name.replace("->", "_").replace(" ", "_").replace(",", "_")
            states_all_scaled = nlp_all.states_scaled.cx_start
            controls_all_scaled = nlp_all.controls_scaled.cx_start
            state_cx_scaled = vertcat(states_all_scaled)
            control_cx_scaled = vertcat(controls_all_scaled)

        else:
            ocp = controller.ocp
            name = self.name
            if self.integrate:
                state_cx_scaled = horzcat(
                    *([controller.states_scaled.cx_start] + controller.states_scaled.cx_intermediates_list)
                )
                control_cx_scaled = controller.controls_scaled.cx_start
            else:
                state_cx_scaled = controller.states_scaled.cx_start
                control_cx_scaled = controller.controls_scaled.cx_start
            if self.explicit_derivative:
                if self.derivative:
                    raise RuntimeError("derivative and explicit_derivative cannot be simultaneously true")
                state_cx_scaled = horzcat(state_cx_scaled, controller.states_scaled.cx_end)
                control_cx_scaled = horzcat(control_cx_scaled, controller.controls_scaled.cx_end)

        # Alias some variables
        node = controller.node_index
        param_cx = controller.parameters.cx_start

        # Sanity check on outputs
        if len(self.function) <= node:
            for _ in range(len(self.function), node + 1):
                self.function.append(None)
                self.weighted_function.append(None)
                self.function_non_threaded.append(None)
                self.weighted_function_non_threaded.append(None)

        # Do not use nlp.add_casadi_func because all functions must be registered
        sub_fcn = fcn[self.rows, self.cols]
        self.function[node] = controller.to_casadi_func(
            name, sub_fcn, state_cx_scaled, control_cx_scaled, param_cx, expand=self.expand
        )
        self.function_non_threaded[node] = self.function[node]

        if self.derivative:
            state_cx_scaled = horzcat(controller.states_scaled.cx_end, controller.states_scaled.cx_start)
            control_cx_scaled = horzcat(controller.controls_scaled.cx_end, controller.controls_scaled.cx_start)
            self.function[node] = biorbd.to_casadi_func(
                f"{name}",
                self.function[node](controller.states_scaled.cx_end, controller.controls_scaled.cx_end, param_cx)
                - self.function[node](controller.states_scaled.cx_start, controller.controls_scaled.cx_start, param_cx),
                state_cx_scaled,
                control_cx_scaled,
                param_cx,
            )

        dt_cx = controller.cx.sym("dt", 1, 1)
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
        target_cx = controller.cx.sym("target", target_shape)
        weight_cx = controller.cx.sym("weight", 1, 1)
        exponent = 2 if self.quadratic and self.weight else 1

        if is_trapezoidal:
            # Hypothesis: the function is continuous on states
            # it neglects the discontinuities at the beginning of the optimization
            state_cx_scaled = (
                horzcat(controller.states_scaled.cx_start, controller.states_scaled.cx_end)
                if self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                else controller.states_scaled.cx_start
            )
            state_cx = (
                horzcat(controller.states.cx_start, controller.states.cx_end)
                if self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                else controller.states.cx_start
            )
            # to handle piecewise constant in controls we have to compute the value for the end of the interval
            # which only relies on the value of the control at the beginning of the interval
            control_cx_scaled = (
                horzcat(controller.controls_scaled.cx_start)
                if controller.control_type == ControlType.CONSTANT
                else horzcat(controller.controls_scaled.cx_start, controller.controls_scaled.cx_end)
            )
            control_cx = (
                horzcat(controller.controls.cx_start)
                if controller.control_type == ControlType.CONSTANT
                else horzcat(controller.controls.cx_start, controller.controls.cx_end)
            )
            control_cx_end_scaled = get_u(control_cx_scaled, dt_cx)
            control_cx_end = get_u(control_cx, dt_cx)
            state_cx_end_scaled = (
                controller.states_scaled.cx_end
                if self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                else controller.integrate(x0=state_cx, p=control_cx_end, params=controller.parameters.cx_start)["xf"]
            )
            modified_function = controller.to_casadi_func(
                f"{name}",
                (
                    (
                        self.function[node](
                            controller.states_scaled.cx_start, controller.controls_scaled.cx_start, param_cx
                        )
                        - target_cx[:, 0]
                    )
                    ** exponent
                    + (self.function[node](state_cx_end_scaled, control_cx_end_scaled, param_cx) - target_cx[:, 1])
                    ** exponent
                )
                / 2,
                state_cx_scaled,
                control_cx_scaled,
                param_cx,
                target_cx,
                dt_cx,
            )
            modified_fcn = modified_function(state_cx_scaled, control_cx_scaled, param_cx, target_cx, dt_cx)
        else:
            modified_fcn = (self.function[node](state_cx_scaled, control_cx_scaled, param_cx) - target_cx) ** exponent

        # for the future bioptim adventurer: here lies the reason that a constraint must have weight = 0.
        modified_fcn = weight_cx * modified_fcn * dt_cx if self.weight else modified_fcn * dt_cx

        # Do not use nlp.add_casadi_func because all of them must be registered
        self.weighted_function[node] = Function(
            name, [state_cx_scaled, control_cx_scaled, param_cx, weight_cx, target_cx, dt_cx], [modified_fcn]
        )
        self.weighted_function_non_threaded[node] = self.weighted_function[node]

        if ocp.n_threads > 1 and self.multi_thread and len(self.node_idx) > 1:
            self.function[node] = self.function[node].map(len(self.node_idx), "thread", ocp.n_threads)
            self.weighted_function[node] = self.weighted_function[node].map(len(self.node_idx), "thread", ocp.n_threads)
        else:
            self.multi_thread = False  # Override the multi_threading, since only one node is optimized

        if self.expand:
            self.function[node] = self.function[node].expand()
            self.weighted_function[node] = self.weighted_function[node].expand()

    def add_target_to_plot(self, controller: PenaltyController, combine_to: str):
        """
        Interface to the plot so it can be properly added to the proper plot

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        combine_to: str
            The name of the underlying plot to combine the tracking data to
        """

        if self.target is None or combine_to is None:
            return

        self.target_plot_name = combine_to
        # if the target is n x ns, we need to add a dimension (n x ns + 1) to make it compatible with the plot
        if self.target[0].shape[1] == controller.ns:
            self.target_to_plot = np.concatenate(
                (self.target[0], np.nan * np.ndarray((self.target[0].shape[0], 1))), axis=1
            )
        else:
            self.target_to_plot = self.target[0]

    def _finish_add_target_to_plot(self, controller: PenaltyController):
        """
        Internal interface to add (after having check the target dimensions) the target to the plot if needed

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        """

        def plot_function(t, x, u, p, penalty=None):
            if isinstance(t, (list, tuple)):
                return self.target_to_plot[:, [self.node_idx.index(_t) for _t in t]]
            else:
                return self.target_to_plot[:, self.node_idx.index(t)]

        if self.target_to_plot is not None:
            if len(self.node_idx) == self.target_to_plot.shape[1]:
                plot_type = PlotType.STEP
            else:
                plot_type = PlotType.POINT

            controller.ocp.add_plot(
                self.target_plot_name,
                plot_function,
                penalty=self if plot_type == PlotType.POINT else None,
                color="tab:red",
                plot_type=plot_type,
                phase=controller.phase_idx,
                axes_idx=Mapping(self.rows),  # TODO verify if not all elements has target
                node_idx=controller.t,
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
            # Make sure the penalty behave like a PhaseTransition, even though it may be an Objective or Constraint
            self.node = Node.END
            self.node_idx = [0]
            self.transition = True
            self.dt = 1
            self.phase_pre_idx = nlp.phase_idx
            self.phase_post_idx = (nlp.phase_idx + 1) % ocp.n_phases
            if not self.states_mapping:
                self.states_mapping = BiMapping(range(nlp.states.shape), range(nlp.states.shape))

            pre = self._get_penalty_controller(ocp, nlp)
            pre.u = [nlp.U[-1]]  # Make an exception to the fact that U is not available for the last node

            nlp = ocp.nlp[(nlp.phase_idx + 1) % ocp.n_phases]
            self.node = Node.START
            post = self._get_penalty_controller(ocp, nlp)

            self.node = Node.TRANSITION

            penalty_type.validate_penalty_time_index(self, pre)
            penalty_type.validate_penalty_time_index(self, post)
            self.ensure_penalty_sanity(ocp, pre.get_nlp)
            controllers = [pre, post]

        elif isinstance(self.node, tuple) and self.binode_constraint:
            nodes = self.node
            # Make sure the penalty behave like a BinodeConstraint, even though it may be an Objective or Constraint
            # self.transition = True
            self.dt = 1
            # self.phase_pre_idx
            # self.phase_post_idx = (nlp.phase_idx + 1) % ocp.n_phases
            if not self.states_mapping:
                self.states_mapping = BiMapping(range(nlp.states.shape), range(nlp.states.shape))
            self.node = nodes[0]
            nlp = ocp.nlp[self.phase_first_idx]
            pre = self._get_penalty_controller(ocp, nlp)
            if self.node == Node.END:
                pre.u = [nlp.U[-1]]
                # Make an exception to the fact that U is not available for the last node

            self.node = nodes[1]
            nlp = ocp.nlp[self.phase_second_idx]
            post = self._get_penalty_controller(ocp, nlp)
            if self.node == Node.END:
                post.u = [nlp.U[-1]]
                # Make an exception to the fact that U is not available for the last node

            # reset the node list
            self.node = nodes

            penalty_type.validate_penalty_time_index(self, pre)
            penalty_type.validate_penalty_time_index(self, post)
            self.node_idx = [pre.t[0], post.t[0]]
            self.ensure_penalty_sanity(ocp, pre.get_nlp)
            controllers = [pre, post]

        elif self.allnode_constraint:
            NotImplementedError("This path seems obsolete")  # TODO verify that. If true remove
            # controller = []
            # Make sure the penalty behave like a BinodeConstraint, even though it may be an Objective or Constraint
            # self.transition = True
            self.dt = 1
            # self.dt = penalty_type.get_dt(controller.nlp)
            # if not self.states_mapping:
            #    self.states_mapping = BiMapping(range(nlp.states.shape), range(nlp.states.shape))
            nlp = ocp.nlp[self.phase_idx]
            # controller.append(self._get_penalty_node_list(ocp, nlp))
            controllers = [self._get_penalty_controller(ocp, nlp)]
            penalty_type.validate_penalty_time_index(self, controllers[0])
            # self.node_idx = [controller[0].t[0]] # t?
            penalty_type.validate_penalty_time_index(self, controllers[0])
            self.node_idx = controllers[0].t
            self.ensure_penalty_sanity(ocp, nlp)

        else:
            controllers = [self._get_penalty_controller(ocp, nlp)]
            penalty_type.validate_penalty_time_index(self, controllers[0])
            self.ensure_penalty_sanity(ocp, nlp)
            self.dt = penalty_type.get_dt(nlp)
            self.node_idx = (
                controllers[0].t[:-1]
                if (
                    self.integration_rule == IntegralApproximation.TRAPEZOIDAL
                    or self.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                )
                and self.target is not None
                else controllers[0].t
            )

        if ocp.assume_phase_dynamics:
            penalty_function = self.type(self, controllers if len(controllers) > 1 else controllers[0], **self.params)
            self.set_penalty(penalty_function, controllers if len(controllers) > 1 else controllers[0])

            # Define much more function than needed, but we don't mind since they are reference copy of each other
            ns = (
                max(controllers[0].get_nlp.ns, controllers[1].get_nlp.ns)
                if len(controllers) > 1
                else controllers[0].get_nlp.ns
            ) + 1
            self.function = self.function * ns
            self.weighted_function = self.weighted_function * ns
            self.function_non_threaded = self.function_non_threaded * ns
            self.weighted_function_non_threaded = self.weighted_function_non_threaded * ns
        else:
            # The active controller is always last
            node_indices = [t for t in controllers[-1].t]
            if self.custom_function and len(node_indices) > 1:
                raise NotImplementedError(
                    "Setting custom function for more than one node at a time when assume_phase_dynamics is "
                    "set to False is not Implemented"
                )
            for node_index in node_indices:
                controllers[-1].t = [node_index]
                controllers[-1].node_index = node_index
                penalty_function = self.type(
                    self, controllers if len(controllers) > 1 else controllers[0], **self.params
                )
                self.set_penalty(penalty_function, controllers if len(controllers) > 1 else controllers[0])

    def _add_penalty_to_pool(self, controller: PenaltyController | list[PenaltyController, ...]):
        """
        Return the penalty pool for the specified penalty (abstract)

        Parameters
        ----------
        controller: PenaltyController
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

    def _get_penalty_controller(self, ocp, nlp) -> PenaltyController:
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
        return PenaltyController(ocp, nlp, t, x, u, x_scaled, u_scaled, nlp.parameters.cx_start)
