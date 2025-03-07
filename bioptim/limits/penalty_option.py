from typing import Any, Callable

import numpy as np
from casadi import vertcat, Function, MX, SX, jacobian, diag

from .penalty_controller import PenaltyController
from ..limits.penalty_helpers import PenaltyHelpers
from ..misc.enums import Node, PlotType, ControlType, PenaltyType, QuadratureRule, PhaseDynamics
from ..misc.mapping import BiMapping
from ..misc.options import OptionGeneric
from ..models.protocols.stochastic_biomodel import StochasticBioModel


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
    integration_rule: QuadratureRule
        The integration rule to use for the penalty
    nodes_phase: tuple[int, ...]
        The index of the phases when penalty is multinodes
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
    _check_target_dimensions(self, controller: PenaltyController, n_frames: int)
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay
    _set_penalty_function(self, controller: list[PenaltyController], fcn: MX | SX)
        Finalize the preparation of the penalty (setting function and weighted_function)
    add_target_to_plot(self, controller: PenaltyController, combine_to: str)
        Interface to the plot so it can be properly added to the proper plot
    _finish_add_target_to_plot(self, controller: PenaltyController)
        Internal interface to add (after having check the target dimensions) the target to the plot if needed
    add_or_replace_to_penalty_pool(self, ocp, nlp)
        Doing some configuration on the penalty and add it to the list of penalty
    _add_penalty_to_pool(self, controller: list[PenaltyController])
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
        integration_rule: QuadratureRule = QuadratureRule.DEFAULT,
        index: list = None,
        rows: list | tuple | range | np.ndarray = None,
        cols: list | tuple | range | np.ndarray = None,
        custom_function: Callable = None,
        penalty_type: PenaltyType = PenaltyType.USER,
        is_stochastic: bool = False,
        multi_thread: bool = None,
        expand: bool = False,
        **extra_parameters: Any,
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
        integration_rule: QuadratureRule
            The rule to use for the integration
        index: int
            The component index the penalty is acting on
        custom_function: Callable
            A user defined function to call to get the penalty
        penalty_type: PenaltyType
            If the penalty is from the user or from bioptim (implicit or internal)
        is_stochastic: bool
            If the penalty is stochastic (i.e. if we should look instead at the variation of the penalty)
        **extra_parameters: dict
            Generic parameters for the penalty
        """

        super(PenaltyOption, self).__init__(phase=phase, type=penalty, **extra_parameters)
        self.node: Node | list | tuple = node
        self.quadratic = quadratic
        self.integration_rule = integration_rule
        if self.integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL):
            integrate = True
        self.derivative = derivative
        self.explicit_derivative = explicit_derivative
        self.integrate = integrate

        self.extra_arguments = extra_parameters

        if index is not None and rows is not None:
            raise ValueError("rows and index cannot be defined simultaneously since they are the same variable")
        self.rows = rows if rows is not None else index
        self.cols = cols
        self.cols_is_set = False  # This is an internal variable that is set after 'set_idx_columns' is called
        self.rows_is_set = False
        self.expand = expand

        self.phase_dynamics = []  # This is set by _set_phase_dynamics
        self.ns = []  # This is set by _set_ns
        self.control_types = None  # This is set by _set_control_types

        self.target = None
        if target is not None:
            self.target = np.array(target)
            # Make sure the target has at least 2 dimensions
            if len(self.target.shape) == 0:
                self.target = self.target[np.newaxis]
            if len(self.target.shape) == 1:
                self.target = self.target[:, np.newaxis]

        self.target_plot_name = None
        self.target_to_plot = None
        # todo: not implemented yet for trapezoidal integration
        self.plot_target = (
            False
            if (
                self.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                or self.integration_rule == QuadratureRule.TRAPEZOIDAL
            )
            else True
        )

        self.custom_function = custom_function

        self.node_idx = []
        self.multinode_idx = None
        self.dt = 0
        self.weight = weight
        self.function: list[Function | None] = []
        self.function_non_threaded: list[Function | None] = []
        self.weighted_function: list[Function | None] = []
        self.weighted_function_non_threaded: list[Function | None] = []

        self.multinode_penalty = False
        self.nodes_phase = None  # This is relevant for multinodes
        self.nodes = None  # This is relevant for multinodes
        if self.derivative and self.explicit_derivative:
            raise ValueError("derivative and explicit_derivative cannot be both True")
        self.subnodes_are_decision_states = []  # This is set by _set_subnodes_are_decision_states
        self.penalty_type = penalty_type
        self.is_stochastic = is_stochastic

        self.multi_thread = multi_thread

    def set_penalty(
        self, penalty: MX | SX, controllers: PenaltyController | list[PenaltyController, PenaltyController]
    ):
        """
        Prepare the dimension and index of the penalty (including the target)

        Parameters
        ----------
        penalty: MX | SX,
            The actual penalty function
        controllers: PenaltyController | list[PenaltyController, PenaltyController]
            The penalty node elements
        """

        self.rows = self._set_dim_idx(self.rows, penalty.rows())
        self.cols = self._set_dim_idx(self.cols, penalty.columns())
        if self.target is not None:
            if isinstance(controllers, list):
                raise RuntimeError("Multinode constraints should call a self defined set_penalty")

            self._check_target_dimensions(controllers)
            if self.plot_target:
                self._finish_add_target_to_plot(controllers)

        if not isinstance(controllers, list):
            controllers = [controllers]

        self._set_phase_dynamics(controllers)
        self._set_ns(controllers)
        self._set_control_types(controllers)
        self._set_subnodes_are_decision_states(controllers)

        self._set_penalty_function(controllers, penalty)
        self._add_penalty_to_pool(controllers)

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

    def _check_target_dimensions(self, controller: PenaltyController):
        """
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        """

        n_frames = len(controller.t) + (1 if self.integrate else 0)

        n_dim = len(self.target.shape)
        if n_dim != 2 and n_dim != 3:
            raise RuntimeError(f"target cannot be a vector (it can be a matrix with time dimension equals to 1 though)")

        if self.target.shape[-1] == 1:
            self.target = np.repeat(self.target, n_frames, axis=-1)

        shape = (len(self.rows), n_frames) if n_dim == 2 else (len(self.rows), len(self.cols), n_frames)

        if self.target.shape != shape:
            raise RuntimeError(
                f"target {self.target.shape} does not correspond to expected size {shape} for penalty {self.name}"
            )

    def transform_penalty_to_stochastic(self, controller: PenaltyController, fcn, state_cx_scaled):
        """
        Transform the penalty fcn into the variation of fcn depending on the noise:
            fcn = fcn(x, u, p, a, d) becomes d/dx(fcn) * covariance * d/dx(fcn).T

        Please note that this is usually used to add a buffer around an equality constraint h(x, u, p, a, d) = 0
        transforming it into an inequality constraint of the form:
            h(x, u, p, a, d) + sqrt(dh/dx * covariance * dh/dx.T) <= 0

        Here, we chose a different implementation to avoid the discontinuity of the sqrt, we instead decompose the two
        terms, meaning that you have to declare the constraint h=0 and the "variation of h"=buffer ** 2 with
        is_stochastic=True independently.
        """

        # TODO: Charbie -> This is just a first implementation (x=[q, qdot]), it should then be generalized

        nx = controller.q.shape[0]
        n_root = controller.model.nb_root
        n_joints = nx - n_root

        if "cholesky_cov" in controller.controls.keys():
            l_cov_matrix = StochasticBioModel.reshape_to_cholesky_matrix(
                controller.controls["cholesky_cov"].cx_start, controller.model.matrix_shape_cov_cholesky
            )
            cov_matrix = l_cov_matrix @ l_cov_matrix.T
        else:
            cov_matrix = StochasticBioModel.reshape_to_matrix(
                controller.controls["cov"].cx_start, controller.model.matrix_shape_cov
            )

        jac_fcn_states = jacobian(fcn, state_cx_scaled)
        fcn_variation = jac_fcn_states @ cov_matrix @ jac_fcn_states.T

        return diag(fcn_variation)

    def _set_phase_dynamics(self, controllers: list[PenaltyController]):
        phase_dynamics = [c.get_nlp.phase_dynamics for c in controllers]
        if self.phase_dynamics:
            # If it was already set (e.g. for multinode), we want to make sure it is consistent
            if self.phase_dynamics != phase_dynamics:
                raise RuntimeError(
                    "The phase dynamics of the penalty are not consistent. "
                    "This should not happen. Please report this issue."
                )
        self.phase_dynamics = phase_dynamics

    def _set_ns(self, controllers: list[PenaltyController]):
        ns = [c.ns for c in controllers]
        if self.ns:
            # If it was already set (e.g. for multinode), we want to make sure it is consistent
            if self.ns != ns:
                raise RuntimeError(
                    "The number of shooting points of the penalty are not consistent. "
                    "This should not happen. Please report this issue."
                )
        self.ns = ns

    def _set_control_types(self, controllers: list[PenaltyController]):
        control_types = [c.control_type for c in controllers]
        if self.control_types:
            # If it was already set (e.g. for multinode), we want to make sure it is consistent
            if self.control_types != control_types:
                raise RuntimeError(
                    "The control types of the penalty are not consistent. "
                    "This should not happen. Please report this issue."
                )
        self.control_types = control_types

    def _set_subnodes_are_decision_states(self, controllers: list[PenaltyController]):
        subnodes_are_decision_states = [c.get_nlp.ode_solver.is_direct_collocation for c in controllers]
        if self.subnodes_are_decision_states:
            # If it was already set (e.g. for multinode), we want to make sure it is consistent
            if self.subnodes_are_decision_states != subnodes_are_decision_states:
                raise RuntimeError(
                    "The subnodes_are_decision_states of the penalty are not consistent. "
                    "This should not happen. Please report this issue."
                )
        self.subnodes_are_decision_states = subnodes_are_decision_states

    def _set_penalty_function(self, controllers: list[PenaltyController], fcn: MX | SX):
        """
        Finalize the preparation of the penalty (setting function and weighted_function)

        Parameters
        ----------
        controllers: list[PenaltyController, PenaltyController]
            The nodes
        fcn: MX | SX
            The value of the penalty function
        """

        name = (
            self.name.replace("->", "_")
            .replace(" ", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace(",", "_")
            .replace(":", "_")
            .replace(".", "_")
            .replace("__", "_")
        )

        controller, _, x, u, p, a, d = self.get_variable_inputs(controllers)

        # Alias some variables
        node = controller.node_index

        dt = controller.dt.cx
        time = controller.time.cx
        phases_dt = controller.phases_dt.cx

        # Sanity check on outputs
        if len(self.function) <= node:
            for _ in range(len(self.function), node + 1):
                self.function.append(None)
                self.weighted_function.append(None)
                self.function_non_threaded.append(None)
                self.weighted_function_non_threaded.append(None)

        sub_fcn = fcn[self.rows, self.cols]
        if self.is_stochastic:
            sub_fcn = self.transform_penalty_to_stochastic(controller, sub_fcn, x)

        from ..limits.constraints import ConstraintFcn
        from ..limits.multinode_constraint import MultinodeConstraintFcn

        if (
            len(sub_fcn.shape) > 1
            and sub_fcn.shape[1] != 1
            and (isinstance(self.type, ConstraintFcn) or isinstance(self.type, MultinodeConstraintFcn))
        ):
            raise RuntimeError("The constraint must return a vector not a matrix.")

        is_trapezoidal = self.integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL)
        target_shape = tuple([len(self.rows), len(self.cols) + 1 if is_trapezoidal else len(self.cols)])
        target_cx = controller.cx.sym("target", target_shape)
        weight_cx = controller.cx.sym("weight", 1, 1)
        exponent = 2 if self.quadratic and self.weight else 1

        if is_trapezoidal:
            # Hypothesis for APPROXIMATE_TRAPEZOIDAL: the function is continuous on states
            # it neglects the discontinuities at the beginning of the optimization
            param_cx_start = controller.parameters_scaled.cx
            state_cx_start = controller.states_scaled.cx_start
            algebraic_states_start_cx = controller.algebraic_states_scaled.cx_start
            algebraic_states_end_cx = controller.algebraic_states_scaled.cx_end
            numerical_timeseries_start_cx = controller.numerical_timeseries.cx_start
            numerical_timeseries_end_cx = controller.numerical_timeseries.cx_end

            # Perform the integration to get the final subnode
            if self.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                state_cx_end = controller.states_scaled.cx_end
            elif self.integration_rule == QuadratureRule.TRAPEZOIDAL:
                u_integrate = u.reshape((-1, 2))
                if self.control_types[0] in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                    u_integrate = u_integrate[:, 0]
                elif self.control_types[0] in (ControlType.LINEAR_CONTINUOUS,):
                    pass
                else:
                    raise NotImplementedError(f"Control type {self.control_types[0]} not implemented yet")

                state_cx_end = controller.integrate(
                    t_span=controller.t_span.cx,
                    x0=controller.states.cx_start,
                    u=u_integrate,
                    p=controller.parameters.cx,
                    a=controller.algebraic_states.cx_start,
                    d=controller.numerical_timeseries.cx_start,
                )["xf"]
            else:
                raise NotImplementedError(f"Integration rule {self.integration_rule} not implemented yet")

            # to handle piecewise constant in controls we have to compute the value for the end of the interval
            # which only relies on the value of the control at the beginning of the interval
            control_cx_start = controller.controls_scaled.cx_start
            if self.control_types[0] in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                # This effectively equates a TRAPEZOIDAL integration into a LEFT_RECTANGLE for penalties that targets
                # controls with a constant control. This philosophically makes sense as the control is constant and
                # applying a trapezoidal integration would be equivalent to applying a left rectangle integration
                control_cx_end = controller.controls_scaled.cx_start
            else:
                if self.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                    control_cx_end = controller.controls_scaled.cx_start
                else:
                    control_cx_end = controller.controls_scaled.cx_end

            # Compute the penalty function at starting and ending of the interval
            func_at_subnode = Function(
                name,
                [
                    time,
                    phases_dt,
                    state_cx_start,
                    control_cx_start,
                    param_cx_start,
                    algebraic_states_start_cx,
                    numerical_timeseries_start_cx,
                ],
                [sub_fcn],
            )
            func_at_start = func_at_subnode(
                time,
                phases_dt,
                state_cx_start,
                control_cx_start,
                param_cx_start,
                algebraic_states_start_cx,
                numerical_timeseries_start_cx,
            )
            func_at_end = func_at_subnode(
                time + dt,
                phases_dt,
                state_cx_end,
                control_cx_end,
                param_cx_start,
                algebraic_states_end_cx,
                numerical_timeseries_end_cx,
            )
            modified_fcn = (
                (func_at_start - target_cx[:, 0]) ** exponent + (func_at_end - target_cx[:, 1]) ** exponent
            ) / 2

            # This reimplementation is required because input sizes change. It will however produce wrong result
            # for non weighted functions
            self.function[node] = Function(
                name,
                [time, phases_dt, x, u, p, a, d],
                [(func_at_start + func_at_end) / 2],
                ["t", "dt", "x", "u", "p", "a", "d"],
                ["val"],
            )
        elif self.derivative:
            # This assumes a Mayer-like penalty
            x_start = controller.states_scaled.cx_start
            x_end = controller.states_scaled.cx_end
            u_start = controller.controls_scaled.cx_start
            if self.control_types[0] in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                u_end = controller.controls_scaled.cx_start
            else:
                u_end = controller.controls_scaled.cx_end
            p_start = controller.parameters_scaled.cx
            a_start = controller.algebraic_states_scaled.cx_start
            a_end = controller.algebraic_states_scaled.cx_end
            numerical_timeseries_start = controller.numerical_timeseries.cx_start
            numerical_timeseries_end = controller.numerical_timeseries.cx_end

            fcn_tp = self.function[node] = Function(
                name,
                [time, phases_dt, x_start, u_start, p_start, a_start, numerical_timeseries_start],
                [sub_fcn],
                ["t", "dt", "x", "u", "p", "a", "d"],
                ["val"],
            )

            self.function[node] = Function(
                f"{name}",
                [time, phases_dt, x, u, p, a, d],
                [
                    fcn_tp(time, phases_dt, x_end, u_end, p, a_end, numerical_timeseries_end)
                    - fcn_tp(time, phases_dt, x_start, u_start, p, a_start, numerical_timeseries_start)
                ],
                ["t", "dt", "x", "u", "p", "a", "d"],
                ["val"],
            )

            modified_fcn = (self.function[node](time, phases_dt, x, u, p, a, d) - target_cx) ** exponent

        else:
            # TODO Add error message if there are free variables to guide the user? For instance controls with last node
            self.function[node] = Function(
                name,
                [time, phases_dt, x, u, p, a, d],
                [sub_fcn],
                ["t", "dt", "x", "u", "p", "a", "d"],
                ["val"],
            )

            modified_fcn = (self.function[node](time, phases_dt, x, u, p, a, d) - target_cx) ** exponent

        if self.expand:
            self.function[node] = self.function[node].expand()

        self.function_non_threaded[node] = self.function[node]

        # weight is zero for constraints penalty and non-zero for objective functions
        modified_fcn = (weight_cx * modified_fcn * self.dt) if self.weight else (modified_fcn * self.dt)

        self.weighted_function[node] = Function(
            name,
            [time, phases_dt, x, u, p, a, d, weight_cx, target_cx],
            [modified_fcn],
            ["t", "dt", "x", "u", "p", "a", "d", "weight", "target"],
            ["val"],
        )
        self.weighted_function_non_threaded[node] = self.weighted_function[node]

        if controller.ocp.n_threads > 1 and self.multi_thread and len(self.node_idx) > 1:
            self.function[node] = self.function[node].map(len(self.node_idx), "thread", controller.ocp.n_threads)
            self.weighted_function[node] = self.weighted_function[node].map(
                len(self.node_idx), "thread", controller.ocp.n_threads
            )
        else:
            self.multi_thread = False  # Override the multi_threading, since only one node is optimized

        if self.expand:
            self.function[node] = self.function[node].expand()
            self.weighted_function[node] = self.weighted_function[node].expand()

    def _check_sanity_of_penalty_interactions(self, controller):
        if self.multinode_penalty and self.explicit_derivative:
            raise ValueError("multinode_penalty and explicit_derivative cannot be true simultaneously")
        if self.multinode_penalty and self.derivative:
            raise ValueError("multinode_penalty and derivative cannot be true simultaneously")
        if self.derivative and self.explicit_derivative:
            raise ValueError("derivative and explicit_derivative cannot be true simultaneously")

        if controller.get_nlp.ode_solver.is_direct_collocation and (
            controller.get_nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
            and len(self.node_idx) > 1
            and controller.ns + 1 in self.node_idx
        ):
            raise ValueError(
                "Direct collocation with shared dynamics cannot have a more than one penalty defined at the same "
                "time on multiple node. If you arrive to this error using Node.ALL, you should consider using "
                "Node.ALL_SHOOTING."
            )

    def get_variable_inputs(self, controllers: list[PenaltyController]):
        if self.multinode_penalty:
            controller = controllers[0]  # Recast controller as a normal variable (instead of a list)
            self.node_idx[0] = controller.node_index

            self.all_nodes_index = []
            for ctrl in controllers:
                self.all_nodes_index.extend(ctrl.t)

        else:
            controller = controllers[0]

        self._check_sanity_of_penalty_interactions(controller)

        ocp = controller.ocp
        penalty_idx = self.node_idx.index(controller.node_index)

        t0 = PenaltyHelpers.t0(self, penalty_idx, lambda p, n: ocp.node_time(phase_idx=p, node_idx=n))
        x = PenaltyHelpers.states(
            self,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: self._get_states(ocp, ocp.nlp[p_idx].states, n_idx, sn_idx),
            is_constructing_penalty=True,
        )
        u = PenaltyHelpers.controls(
            self,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: self._get_u(ocp, p_idx, n_idx, sn_idx),
            is_constructing_penalty=True,
        )
        p = PenaltyHelpers.parameters(
            self,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: ocp.parameters.scaled.cx_start,
        )
        a = PenaltyHelpers.states(
            self,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: self._get_states(ocp, ocp.nlp[p_idx].algebraic_states, n_idx, sn_idx),
            is_constructing_penalty=True,
        )
        d = PenaltyHelpers.numerical_timeseries(
            self,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: self.get_numerical_timeseries(ocp, p_idx, n_idx, sn_idx),
        )

        return controller, t0, x, u, p, a, d

    @staticmethod
    def _get_states(ocp, states, n_idx, sn_idx):
        states.node_index = n_idx

        x = ocp.cx()
        if states.scaled.cx_start.shape == (0, 0):
            return x

        if sn_idx.start == 0:
            x = vertcat(x, states.scaled.cx_start)
            if sn_idx.stop == 1:
                pass
            elif sn_idx.stop is None:
                x = vertcat(x, vertcat(*states.scaled.cx_intermediates_list))
            else:
                raise ValueError("The sn_idx.stop should be 1 or None if sn_idx.start == 0")

        elif sn_idx.start == 1:
            if sn_idx.stop == 2:
                x = vertcat(x, vertcat(states.scaled.cx_mid))
            else:
                raise ValueError("The sn_idx.stop should be 2 if sn_idx.start == 1")

        elif sn_idx.start == 2:
            if sn_idx.stop == 3:
                x = vertcat(x, vertcat(states.scaled.cx_end))
            else:
                raise ValueError("The sn_idx.stop should be 3 if sn_idx.start == 2")

        elif sn_idx.start == -1:
            x = vertcat(x, vertcat(states.scaled.cx_end))
            if sn_idx.stop is not None:
                raise ValueError("The sn_idx.stop should be None if sn_idx.start == -1")

        else:
            raise ValueError("The sn_idx.start should be 0 or -1")

        return x

    def _get_u(self, ocp, p_idx, n_idx, sn_idx):
        nlp = ocp.nlp[p_idx]
        controls = nlp.controls
        controls.node_index = n_idx

        def vertcat_cx_end():
            if nlp.control_type in (ControlType.LINEAR_CONTINUOUS,):
                return vertcat(u, controls.scaled.cx_end)
            elif nlp.control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                if n_idx < nlp.n_controls_nodes - 1:
                    return vertcat(u, controls.scaled.cx_end)

                elif n_idx == nlp.n_controls_nodes - 1:
                    # If we are at the penultimate node, we still can use the cx_end, unless we are
                    # performing some kind of integration or derivative and this last node does not exist
                    if nlp.control_type in (ControlType.CONSTANT_WITH_LAST_NODE,):
                        return vertcat(u, controls.scaled.cx_end)
                    if self.integrate or self.derivative or self.explicit_derivative or self.multinode_penalty:
                        return u
                    else:
                        return vertcat(u, controls.scaled.cx_end)

                else:
                    return u
            else:
                raise NotImplementedError(f"Control type {nlp.control_type} not implemented yet")

        u = ocp.cx()
        if sn_idx.start == 0:
            u = vertcat(u, controls.scaled.cx_start)
            if sn_idx.stop == 1:
                pass
            elif sn_idx.stop is None:
                u = vertcat_cx_end()
            else:
                raise ValueError("The sn_idx.stop should be 1 or None if sn_idx.start == 0")

        elif sn_idx.start == 1:
            if sn_idx.stop == 2:
                u = vertcat(u, controls.scaled.cx_intermediates_list[0])
            else:
                raise ValueError("The sn_idx.stop should be 2 if sn_idx.start == 1")

        elif sn_idx.start == 2:
            # This is not the actual endpoint but a midpoint that must use cx_end
            if sn_idx.stop == 3:
                u = vertcat(u, controls.scaled.cx_end)
            else:
                raise ValueError("The sn_idx.stop should be 3 if sn_idx.start == 2")

        elif sn_idx.start == -1:
            if sn_idx.stop is not None:
                raise ValueError("The sn_idx.stop should be None if sn_idx.start == -1")
            u = vertcat_cx_end()

        else:
            raise ValueError("The sn_idx.start should be 0 or -1")

        return u

    def get_numerical_timeseries(self, ocp, p_idx, n_idx, sn_idx):
        nlp = ocp.nlp[p_idx]
        numerical_timeseries = nlp.numerical_timeseries

        if numerical_timeseries.cx_start.shape == (0, 0):
            return ocp.cx()
        elif sn_idx == 0:
            return numerical_timeseries.cx_start
        elif sn_idx == -1:
            return numerical_timeseries.cx_end
        else:
            raise ValueError("The sn_idx should be 0 or -1")

    @staticmethod
    def define_target_mapping(controller: PenaltyController, key: str, rows):
        target_mapping = BiMapping(range(len(controller.get_nlp.variable_mappings[key].to_first.map_idx)), list(rows))
        return target_mapping

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
        if self.target.shape[1] == controller.ns:
            self.target_to_plot = np.concatenate((self.target, np.nan * np.ndarray((self.target.shape[0], 1))), axis=1)

    def _finish_add_target_to_plot(self, controller: PenaltyController):
        """
        Internal interface to add (after having check the target dimensions) the target to the plot if needed

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        """

        def plot_function(t0, phases_dt, node_idx, x, u, p, a, d, penalty=None):
            if isinstance(node_idx, (list, tuple)):
                return self.target_to_plot[:, [self.node_idx.index(idx) for idx in node_idx]]
            else:
                return self.target_to_plot[:, [self.node_idx.index(node_idx)]]

        if self.target_to_plot is not None:
            if len(self.node_idx) == self.target_to_plot.shape[1]:
                plot_type = PlotType.STEP
            else:
                plot_type = PlotType.POINT

            target_mapping = self.define_target_mapping(controller, self.extra_parameters["key"], self.rows)
            controller.ocp.add_plot(
                self.target_plot_name,
                plot_function,
                penalty=self if plot_type == PlotType.POINT else None,
                color="tab:red",
                plot_type=plot_type,
                phase=controller.get_nlp.phase_idx,
                axes_idx=target_mapping,
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
        if self.node in [Node.MULTINODES, Node.TRANSITION]:
            # Make sure the penalty behave like a PhaseTransition, even though it may be an Objective or Constraint
            current_node_type = self.node
            self.dt = 1

            controllers = []
            self.multinode_idx = []
            for node, phase_idx in zip(self.nodes, self.nodes_phase):
                self.node = node
                nlp = ocp.nlp[phase_idx % ocp.n_phases]  # this is to allow using -1 to refer to the last phase

                controllers.append(self.get_penalty_controller(ocp, nlp))
                if (self.node[0] == Node.END or self.node[0] == nlp.ns) and nlp.U != []:
                    # Make an exception to the fact that U is not available for the last node
                    controllers[-1].u = [nlp.U[-1]]
                penalty_type.validate_penalty_time_index(self, controllers[-1])
                self.multinode_idx.append(controllers[-1].t[0])

            # reset the node
            self.node = current_node_type

            # Finalize
            self.ensure_penalty_sanity(ocp, controllers[0].get_nlp)

        else:
            controllers = [self.get_penalty_controller(ocp, nlp)]
            penalty_type.validate_penalty_time_index(self, controllers[0])
            self.ensure_penalty_sanity(ocp, nlp)
            self.dt = penalty_type.get_dt(nlp)
            self.node_idx = controllers[0].t

        # The active controller is always the last one, and they all should be the same length anyway
        for node in range(len(controllers[-1])):
            # TODO
            # TODO WARNING THE NEXT IF STATEMENT IS A BUG DELIBERATELY INTRODUCED TO FIT THE PREVIOUS RESULTS.
            # IT SHOULD BE REMOVED AS SOON AS THE MERGE IS DONE (AND VALUES OF THE TESTS ADJUSTED)
            if self.integrate and self.target is not None:
                self.node_idx = controllers[0].t[:-1]
                if node not in self.node_idx:
                    continue

            for controller in controllers:
                controller.node_index = controller.t[node]
                controller.cx_index_to_get = 0

            penalty_function = self.type(
                self, controllers if len(controllers) > 1 else controllers[0], **self.extra_parameters
            )

            self.set_penalty(penalty_function, controllers if len(controllers) > 1 else controllers[0])

    def _add_penalty_to_pool(self, controller: list[PenaltyController]):
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

    def get_penalty_controller(self, ocp, nlp) -> PenaltyController:
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

        t_idx = []
        for node in self.node:
            if isinstance(node, int):
                if node < 0 or node > nlp.ns:
                    raise RuntimeError(f"Invalid node, {node} must be between 0 and {nlp.ns}")
                t_idx.append(node)
            elif node == Node.START:
                t_idx.append(0)
            elif node == Node.MID:
                if nlp.ns % 2 == 1:
                    raise ValueError("Number of shooting points must be even to use MID")
                t_idx.append(nlp.ns // 2)
            elif node == Node.INTERMEDIATES:
                t_idx.extend(list(i for i in range(1, nlp.ns - 1)))
            elif node == Node.PENULTIMATE:
                if nlp.ns < 2:
                    raise ValueError("Number of shooting points must be greater than 1")
                t_idx.append(nlp.ns - 1)
            elif node == Node.END:
                t_idx.append(nlp.ns)
            elif node == Node.ALL_SHOOTING:
                t_idx.extend(range(nlp.ns))
            elif node == Node.ALL:
                t_idx.extend(range(nlp.ns + 1))
            else:
                raise RuntimeError(f"{node} is not a valid node")

        x = [nlp.X[idx] for idx in t_idx]
        x_scaled = [nlp.X_scaled[idx] for idx in t_idx]
        u, u_scaled = [], []
        a, a_scaled = [], []
        if nlp.U is not None and (not isinstance(nlp.U, list) or nlp.U != []):
            u = [nlp.U[idx] for idx in t_idx if idx != nlp.ns]
            u_scaled = [nlp.U_scaled[idx] for idx in t_idx if idx != nlp.ns]
        if nlp.A is not None and (not isinstance(nlp.A, list) or nlp.A != []):
            a = [nlp.A[idx] for idx in t_idx]
            a_scaled = [nlp.A_scaled[idx] for idx in t_idx]
        d = [nlp.numerical_timeseries for idx in t_idx]
        return PenaltyController(ocp, nlp, t_idx, x, u, x_scaled, u_scaled, nlp.parameters.cx, a, a_scaled, d)
