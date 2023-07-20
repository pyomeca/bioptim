from typing import Callable, Any

from casadi import MX, vertcat, Function, DM
import numpy as np

from .fatigue.fatigue_dynamics import FatigueList, MultiFatigueInterface
from .ode_solver import OdeSolver
from ..gui.plot import CustomPlot
from ..limits.path_conditions import Bounds
from ..misc.enums import (
    PlotType,
    ControlType,
    VariableType,
)
from ..misc.mapping import BiMapping


class NewVariableConfiguration:
    def __init__(
            self,
            name: str,
            name_elements: list,
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_states_dot: bool = False,
            as_stochastic: bool = False,
            fatigue: FatigueList = None,
            combine_name: str = None,
            combine_state_control_plot: bool = False,
            skip_plot: bool = False,
            axes_idx: BiMapping = None,
    ):
        """
        Add a new variable to the states/controls pool

        Parameters
        ----------
        name: str
            The name of the new variable to add
        name_elements: list[str]
            The name of each element of the vector
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the new variable should be added to the state variable set
        as_states_dot: bool
            If the new variable should be added to the state_dot variable set
        as_controls: bool
            If the new variable should be added to the control variable set
        as_stochastic: bool
            If the new variable should be added to the stochastic variable set
        fatigue: FatigueList
            The list of fatigable item
        combine_name: str
            The name of a previously added plot to combine to
        combine_state_control_plot: bool
            If states and controls plot should be combined. Only effective if as_states and as_controls are both True
        skip_plot: bool
            If no plot should be automatically added
        axes_idx: BiMapping
            The axes index to use for the plot
        """

        self.name = name
        self.name_elements = name_elements
        self.ocp = ocp
        self.nlp = nlp
        self.as_states = as_states
        self.as_controls = as_controls
        self.as_states_dot = as_states_dot
        self.as_stochastic = as_stochastic
        self.fatigue = fatigue
        self.combine_name = combine_name
        self.combine_state_control_plot = combine_state_control_plot
        self.skip_plot = skip_plot
        self.axes_idx = axes_idx

        self.copy_states = False
        self.copy_states_dot = False
        self.copy_controls = False

        self.mx_states = None
        self.mx_states_dot = None
        self.mx_controls = None
        self.mx_stochastic = None

        self._check_combine_state_control_plot()

        if _manage_fatigue_to_new_variable(
                name, name_elements, ocp, nlp, as_states, as_controls, fatigue
        ):
            # If the element is fatigable, this function calls back configure_new_variable to fill everything.
            # Therefore, we can exit now
            return

        self._declare_auto_variable_mapping()
        self._check_phase_mapping_of_variable()
        self._declare_phase_copy_booleans()

        self._declare_initial_guess()
        self._declare_variable_scaling()
        self._use_copy()

        # plot

        self.legend = None

        self._declare_auto_axes_idx()
        if not skip_plot:
            self._declare_legend()
        self._declare_cx_and_plot()


    def _check_combine_state_control_plot(self):
        """ Check if combine_state_control_plot and combine_name are defined simultaneously """
        if self.combine_state_control_plot and self.combine_name is not None:
            raise ValueError("combine_name and combine_state_control_plot cannot be defined simultaneously")

    def _check_phase_mapping_of_variable(self):
        """ Check if the use of states from another phases is compatible with assume_phase_dynamics """
        if not self.ocp.assume_phase_dynamics and (
            self.nlp.use_states_from_phase_idx != self.nlp.phase_idx
            or self.nlp.use_states_dot_from_phase_idx != self.nlp.phase_idx
            or self.nlp.use_controls_from_phase_idx != self.nlp.phase_idx
        ):
            # This check allows to use states[0], controls[0] in the following copy
            raise ValueError("map_state=True must be used alongside with assume_phase_dynamics=True")

    def _declare_phase_copy_booleans(self):
        """ Use of states[0] and controls[0] is permitted since ocp.assume_phase_dynamics is True """
        self.copy_states = (
                self.nlp.use_states_from_phase_idx is not None
                and self.nlp.use_states_from_phase_idx < self.nlp.phase_idx
                and self.name in self.ocp.nlp[self.nlp.use_states_from_phase_idx].states[0]
        )
        self.copy_controls = (
                self.nlp.use_controls_from_phase_idx is not None
                and self.nlp.use_controls_from_phase_idx < self.nlp.phase_idx
                and self.name in self.ocp.nlp[self.nlp.use_controls_from_phase_idx].controls[0]
        )
        self.copy_states_dot = (
                self.nlp.use_states_dot_from_phase_idx is not None
                and self.nlp.use_states_dot_from_phase_idx < self.nlp.phase_idx
                and self.name in self.ocp.nlp[self.nlp.use_states_dot_from_phase_idx].states_dot[0]
        )

    def define_cx_scaled(self, n_col: int, n_shooting: int, initial_node) -> list:
        _cx = [self.nlp.cx() for _ in range(n_shooting + 1)]
        for node_index in range(n_shooting + 1):
            _cx[node_index] = [self.nlp.cx() for _ in range(n_col)]
        for idx in self.nlp.variable_mappings[self.name].to_first.map_idx:
            for node_index in range(n_shooting + 1):
                for j in range(n_col):
                    sign = "-" if np.sign(idx) < 0 else ""
                    _cx[node_index][j] = vertcat(
                        _cx[node_index][j],
                        self.nlp.cx.sym(
                            f"{sign}{self.name}_{self.name_elements[abs(idx)]}_phase{self.nlp.phase_idx}_node{node_index + initial_node}.{j}",
                            1,
                            1,
                        ),
                    )
        return _cx

    def define_cx_unscaled(self, _cx_scaled: list, scaling: np.ndarray) -> list:
        _cx = [self.nlp.cx() for _ in range(len(_cx_scaled))]
        for node_index in range(len(_cx_scaled)):
            _cx[node_index] = [self.nlp.cx() for _ in range(len(_cx_scaled[0]))]

        for node_index in range(len(_cx_scaled)):
            for j in range(len(_cx_scaled[0])):
                _cx[node_index][j] = _cx_scaled[node_index][j] * scaling
        return _cx

    def _declare_auto_variable_mapping(self):
        """ Declare the mapping of the new variable if not already declared """
        if self.name not in self.nlp.variable_mappings:
            self.nlp.variable_mappings[self.name] = BiMapping(range(len(self.name_elements)), range(len(self.name_elements)))

    def _declare_initial_guess(self):
        if self.as_states and self.name not in self.nlp.x_init:
            self.nlp.x_init.add(self.name, initial_guess=np.zeros(len(self.nlp.variable_mappings[self.name].to_first.map_idx)))
        if self.as_controls and self.name not in self.nlp.u_init:
            self.nlp.u_init.add(self.name, initial_guess=np.zeros(len(self.nlp.variable_mappings[self.name].to_first.map_idx)))

    def _declare_variable_scaling(self):
        if self.as_states and self.name not in self.nlp.x_scaling:
            self.nlp.x_scaling.add(self.name, scaling=np.ones(len(self.nlp.variable_mappings[self.name].to_first.map_idx)))
        if self.as_states_dot and self.name not in self.nlp.xdot_scaling:
            self.nlp.xdot_scaling.add(self.name, scaling=np.ones(len(self.nlp.variable_mappings[self.name].to_first.map_idx)))
        if self.as_controls and self.name not in self.nlp.u_scaling:
            self.nlp.u_scaling.add(self.name, scaling=np.ones(len(self.nlp.variable_mappings[self.name].to_first.map_idx)))

    def _use_copy(self):
        """ Use of states[0] and controls[0] is permitted since ocp.assume_phase_dynamics is True """
        self.mx_states = [] if not self.copy_states else [self.ocp.nlp[self.nlp.use_states_from_phase_idx].states[0][self.name].mx]
        self.mx_states_dot = (
            [] if not self.copy_states_dot else [self.ocp.nlp[self.nlp.use_states_dot_from_phase_idx].states_dot[0][self.name].mx]
        )
        self.mx_controls = [] if not self.copy_controls else [self.ocp.nlp[self.nlp.use_controls_from_phase_idx].controls[0][self.name].mx]
        self.mx_stochastic = []

        # todo: if mapping on variables, what do we do with mapping on the nodes
        for i in self.nlp.variable_mappings[self.name].to_second.map_idx:
            var_name = f"{'-' if np.sign(i) < 0 else ''}{self.name}_{self.name_elements[abs(i)]}_MX" if i is not None else "zero"

            if not self.copy_states:
                self.mx_states.append(MX.sym(var_name, 1, 1))

            if not self.copy_states_dot:
                self.mx_states_dot.append(MX.sym(var_name, 1, 1))

            if not self.copy_controls:
                self.mx_controls.append(MX.sym(var_name, 1, 1))

            self.mx_stochastic.append(MX.sym(var_name, 1, 1))

        self.mx_states = vertcat(*self.mx_states)
        self.mx_states_dot = vertcat(*self.mx_states_dot)
        self.mx_controls = vertcat(*self.mx_controls)
        self.mx_stochastic = vertcat(*self.mx_stochastic)

    def _declare_auto_axes_idx(self):
        """ Declare the axes index if not already declared """
        if not self.axes_idx:
            self.axes_idx = BiMapping(to_first=range(len(self.name_elements)), to_second=range(len(self.name_elements)))

    def _declare_legend(self):
        """ Declare the legend if not already declared """
        self.legend = []
        for idx, name_el in enumerate(self.name_elements):
            if idx is not None and idx in self.axes_idx.to_first.map_idx:
                current_legend = f"{self.name}_{name_el}"
                for i in range(self.ocp.n_phases):
                    if self.as_states:
                        current_legend += f"-{self.ocp.nlp[i].use_states_from_phase_idx}"
                    if self.as_controls:
                        current_legend += f"-{self.ocp.nlp[i].use_controls_from_phase_idx}"
                self.legend += [current_legend]

    def _declare_cx_and_plot(self):
        if self.as_states:
            for node_index in range((0 if self.ocp.assume_phase_dynamics else self.nlp.ns) + 1):
                n_cx = self.nlp.ode_solver.polynomial_degree + 2 if isinstance(self.nlp.ode_solver, OdeSolver.COLLOCATION) else 3
                cx_scaled = (
                    self.ocp.nlp[self.nlp.use_states_from_phase_idx].states[node_index][self.name].original_cx
                    if self.copy_states
                    else self.define_cx_scaled(n_col=n_cx, n_shooting=0, initial_node=node_index)
                )
                cx = (
                    self.ocp.nlp[self.nlp.use_states_from_phase_idx].states[node_index][self.name].original_cx
                    if self.copy_states
                    else self.define_cx_unscaled(cx_scaled, self.nlp.x_scaling[self.name].scaling)
                )
                self.nlp.states.append(self.name, cx[0], cx_scaled[0], self.mx_states, self.nlp.variable_mappings[self.name], node_index)
                if not self.skip_plot:
                    self.nlp.plot[f"{self.name}_states"] = CustomPlot(
                        lambda t, x, u, p, s: x[self.nlp.states[self.name].index, :],
                        plot_type=PlotType.INTEGRATED,
                        axes_idx=self.axes_idx,
                        legend=self.legend,
                        combine_to=self.combine_name,
                    )

        if self.as_controls:
            for node_index in range(
                (
                    1
                    if self.ocp.assume_phase_dynamics
                    else (self.nlp.ns + (1 if self.nlp.control_type == ControlType.LINEAR_CONTINUOUS else 0))
                )
            ):
                cx_scaled = (
                    self.ocp.nlp[self.nlp.use_controls_from_phase_idx].controls[node_index][self.name].original_cx
                    if self.copy_controls
                    else self.define_cx_scaled(n_col=3, n_shooting=0, initial_node=node_index)
                )
                cx = (
                    self.ocp.nlp[self.nlp.use_controls_from_phase_idx].controls[node_index][self.name].original_cx
                    if self.copy_controls
                    else self.define_cx_unscaled(cx_scaled, self.nlp.u_scaling[self.name].scaling)
                )
                self.nlp.controls.append(self.name, cx[0], cx_scaled[0], self.mx_controls, self.nlp.variable_mappings[self.name], node_index)

                plot_type = PlotType.PLOT if self.nlp.control_type == ControlType.LINEAR_CONTINUOUS else PlotType.STEP
                if not self.skip_plot:
                    self.nlp.plot[f"{self.name}_controls"] = CustomPlot(
                        lambda t, x, u, p, s: u[self.nlp.controls[self.name].index, :],
                        plot_type=plot_type,
                        axes_idx=self.axes_idx,
                        legend=self.legend,
                        combine_to=f"{self.name}_states" if self.as_states and self.combine_state_control_plot else self.combine_name,
                    )

        if self.as_states_dot:
            for node_index in range((0 if self.ocp.assume_phase_dynamics else self.nlp.ns) + 1):
                n_cx = self.nlp.ode_solver.polynomial_degree + 1 if isinstance(self.nlp.ode_solver, OdeSolver.COLLOCATION) else 3
                if n_cx < 3:
                    n_cx = 3
                cx_scaled = (
                    self.ocp.nlp[self.nlp.use_states_dot_from_phase_idx].states_dot[node_index][self.name].original_cx
                    if self.copy_states_dot
                    else self.define_cx_scaled(n_col=n_cx, n_shooting=1, initial_node=node_index)
                )
                cx = (
                    self.ocp.nlp[self.nlp.use_states_dot_from_phase_idx].states_dot[node_index][self.name].original_cx
                    if self.copy_states_dot
                    else self.define_cx_unscaled(cx_scaled, self.nlp.xdot_scaling[self.name].scaling)
                )
                self.nlp.states_dot.append(self.name, cx[0], cx_scaled[0], self.mx_states_dot, self.nlp.variable_mappings[self.name], node_index)

        if self.as_stochastic:
            for node_index in range((0 if self.ocp.assume_phase_dynamics else self.nlp.ns) + 1):
                n_cx = self.nlp.ode_solver.polynomial_degree + 1 if isinstance(self.nlp.ode_solver, OdeSolver.COLLOCATION) else 3
                if n_cx < 3:
                    n_cx = 3
                cx_scaled = self.define_cx_scaled(n_col=n_cx, n_shooting=1, initial_node=node_index)
                self.nlp.stochastic_variables.append(
                    self.name, cx_scaled[0], cx_scaled[0], self.mx_stochastic, self.nlp.variable_mappings[self.name], node_index
                )


def _manage_fatigue_to_new_variable(
    name: str,
    name_elements: list,
    ocp,
    nlp,
    as_states: bool,
    as_controls: bool,
    fatigue: FatigueList = None,
):
    """
    Manage the fatigue variables and add them to the nlp

    Parameters
    ----------
    name: str
        The name of the variable
    name_elements: list
        The name of the elements of the variable
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the nlp
    as_states: bool
        If the fatigue is applied on the states
    as_controls: bool
        If the fatigue is applied on the controls
    fatigue: FatigueList
        The fatigue elements to apply
    """
    if fatigue is None or name not in fatigue:
        return False

    if not as_controls:
        raise NotImplementedError("Fatigue not applied on controls is not implemented yet")

    fatigue_var = fatigue[name]
    meta_suffixes = fatigue_var.suffix

    # Only homogeneous fatigue model are implement
    fatigue_suffix = fatigue_var[0].models.models[meta_suffixes[0]].suffix(VariableType.STATES)
    multi_interface = isinstance(fatigue_var[0].models, MultiFatigueInterface)
    split_controls = fatigue_var[0].models.split_controls
    for dof in fatigue_var:
        for key in dof.models.models:
            if dof.models.models[key].suffix(VariableType.STATES) != fatigue_suffix:
                raise ValueError(f"Fatigue for {name} must be of all same types")
            if isinstance(dof.models, MultiFatigueInterface) != multi_interface:
                raise ValueError("multi_interface must be the same for all the elements")
            if dof.models.split_controls != split_controls:
                raise ValueError("split_controls must be the same for all the elements")

    # Prepare the plot that will combine everything
    n_elements = len(name_elements)

    legend = [f"{name}_{i}" for i in name_elements]
    fatigue_plot_name = f"fatigue_{name}"
    nlp.plot[fatigue_plot_name] = CustomPlot(
        lambda t, x, u, p, s: x[:n_elements, :] * np.nan,
        plot_type=PlotType.INTEGRATED,
        legend=legend,
        bounds=Bounds(None, -1, 1),
    )
    control_plot_name = f"{name}_controls" if not multi_interface and split_controls else f"{name}"
    nlp.plot[control_plot_name] = CustomPlot(
        lambda t, x, u, p, s: u[:n_elements, :] * np.nan, plot_type=PlotType.STEP, legend=legend
    )

    var_names_with_suffix = []
    color = fatigue_var[0].models.color()
    fatigue_color = [fatigue_var[0].models.models[m].color() for m in fatigue_var[0].models.models]
    plot_factor = fatigue_var[0].models.plot_factor()
    for i, meta_suffix in enumerate(meta_suffixes):
        var_names_with_suffix.append(f"{name}_{meta_suffix}" if not multi_interface else f"{name}")

        if split_controls:
            NewVariableConfiguration(
                var_names_with_suffix[-1], name_elements, ocp, nlp, as_states, as_controls, skip_plot=True
            )
            nlp.plot[f"{var_names_with_suffix[-1]}_controls"] = CustomPlot(
                lambda t, x, u, p, s, key: u[nlp.controls[key].index, :],
                plot_type=PlotType.STEP,
                combine_to=control_plot_name,
                key=var_names_with_suffix[-1],
                color=color[i],
            )
        elif i == 0:
            NewVariableConfiguration(
                f"{name}", name_elements, ocp, nlp, as_states, as_controls, skip_plot=True
            )
            nlp.plot[f"{name}_controls"] = CustomPlot(
                lambda t, x, u, p, s, key: u[nlp.controls[key].index, :],
                plot_type=PlotType.STEP,
                combine_to=control_plot_name,
                key=f"{name}",
                color=color[i],
            )

        for p, params in enumerate(fatigue_suffix):
            name_tp = f"{var_names_with_suffix[-1]}_{params}"
            NewVariableConfiguration(name_tp, name_elements, ocp, nlp, True, False, skip_plot=True)
            nlp.plot[name_tp] = CustomPlot(
                lambda t, x, u, p, s, key, mod: mod * x[nlp.states[key].index, :],
                plot_type=PlotType.INTEGRATED,
                combine_to=fatigue_plot_name,
                key=name_tp,
                color=fatigue_color[i][p],
                mod=plot_factor[i],
            )

    # Create a fake accessor for the name of the controls so it can be directly accessed in nlp.controls
    for node_index in range(nlp.ns):
        nlp.controls.node_index = node_index
        if split_controls:
            append_faked_optim_var(name, nlp.controls.scaled, var_names_with_suffix)
            append_faked_optim_var(name, nlp.controls.unscaled, var_names_with_suffix)
        else:
            for meta_suffix in var_names_with_suffix:
                append_faked_optim_var(meta_suffix, nlp.controls.scaled, [name])
                append_faked_optim_var(meta_suffix, nlp.controls.unscaled, [name])

    nlp.controls.node_index = nlp.states.node_index
    nlp.states_dot.node_index = nlp.states.node_index
    return True


def append_faked_optim_var(name: str, optim_var, keys: list):
    """
    Add a fake optim var by combining vars in keys

    Parameters
    ----------
    name: str
        The name of the new variable
    optim_var: OptimizationVariableList
        states or controls
    keys: list
        The list of keys to combine
    """

    index = []
    mx = MX()
    to_second = []
    to_first = []
    for key in keys:
        index.extend(list(optim_var[key].index))
        mx = vertcat(mx, optim_var[key].mx)
        to_second.extend(list(np.array(optim_var[key].mapping.to_second.map_idx) + len(to_second)))
        to_first.extend(list(np.array(optim_var[key].mapping.to_first.map_idx) + len(to_first)))

    optim_var.append_fake(name, index, mx, BiMapping(to_second, to_first))
