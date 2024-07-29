from typing import Any, Callable

from casadi import DM, Function
import numpy as np

from ..dynamics.ode_solver import OdeSolver
from ..limits.penalty_option import PenaltyOption
from ..misc.mapping import BiMapping
from ..misc.enums import PlotType, QuadratureRule


class CasadiFunctionSerializable:
    _size_in: dict[str, int]

    def __init__(self, size_in: dict[str, int]):
        self._size_in = size_in

    @classmethod
    def from_casadi_function(cls, casadi_function):
        casadi_function: Function = casadi_function

        return cls(
            size_in={
                "x": casadi_function.size_in("x"),
                "u": casadi_function.size_in("u"),
                "p": casadi_function.size_in("p"),
                "a": casadi_function.size_in("a"),
                "d": casadi_function.size_in("d"),
            }
        )

    def serialize(self):
        return {
            "size_in": self._size_in,
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            size_in=data["size_in"],
        )

    def size_in(self, key: str) -> int:
        return self._size_in[key]


class PenaltySerializable:
    function: list[CasadiFunctionSerializable | None]

    def __init__(self, function: list[CasadiFunctionSerializable | None]):
        self.function = function

    @classmethod
    def from_penalty(cls, penalty):
        penalty: PenaltyOption = penalty

        function = []
        for f in penalty.function:
            function.append(None if f is None else CasadiFunctionSerializable.from_casadi_function(f))
        return cls(
            function=function,
        )

    def serialize(self):
        return {
            "function": [None if f is None else f.serialize() for f in self.function],
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            function=[None if f is None else CasadiFunctionSerializable.deserialize(f) for f in data["function"]],
        )


class MappingSerializable:
    map_idx: list[int]
    oppose: list[int]

    def __init__(self, map_idx: list, oppose: list):
        self.map_idx = map_idx
        self.oppose = oppose

    def map(self, obj):
        from ..misc.mapping import Mapping

        return Mapping.map(self, obj)

    @classmethod
    def from_mapping(cls, mapping):
        return cls(
            map_idx=mapping.map_idx,
            oppose=mapping.oppose,
        )

    def serialize(self):
        return {
            "map_idx": self.map_idx,
            "oppose": self.oppose,
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            map_idx=data["map_idx"],
            oppose=data["oppose"],
        )


class BiMappingSerializable:
    to_first: MappingSerializable
    to_second: MappingSerializable

    def __init__(self, to_first: MappingSerializable, to_second: MappingSerializable):
        self.to_first = to_first
        self.to_second = to_second

    @classmethod
    def from_bimapping(cls, bimapping):
        return cls(
            to_first=MappingSerializable.from_mapping(bimapping.to_first),
            to_second=MappingSerializable.from_mapping(bimapping.to_second),
        )

    def serialize(self):
        return {
            "to_first": self.to_first.serialize(),
            "to_second": self.to_second.serialize(),
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            to_first=MappingSerializable.deserialize(data["to_first"]),
            to_second=MappingSerializable.deserialize(data["to_second"]),
        )


class BoundsSerializable:
    min: np.ndarray | DM
    max: np.ndarray | DM

    def __init__(self, min: np.ndarray | DM, max: np.ndarray | DM):
        self.min = min
        self.max = max

    @classmethod
    def from_bounds(cls, bounds):
        return cls(
            min=np.array(bounds.min),
            max=np.array(bounds.max),
        )

    def serialize(self):
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            min=np.array(data["min"]),
            max=np.array(data["max"]),
        )


class CustomPlotSerializable:
    _function: Callable
    type: PlotType
    phase_mappings: BiMappingSerializable
    legend: tuple | list
    combine_to: str
    color: str
    linestyle: str
    ylim: tuple | list
    bounds: BoundsSerializable
    node_idx: list | slice | range
    label: list
    compute_derivative: bool
    integration_rule: QuadratureRule
    parameters: dict[str, Any]
    all_variables_in_one_subplot: bool

    def __init__(
        self,
        function: Callable,
        plot_type: PlotType,
        phase_mappings: BiMapping,
        legend: tuple | list,
        combine_to: str,
        color: str,
        linestyle: str,
        ylim: tuple | list,
        bounds: BoundsSerializable,
        node_idx: list | slice | range,
        label: list,
        compute_derivative: bool,
        integration_rule: QuadratureRule,
        parameters: dict[str, Any],
        all_variables_in_one_subplot: bool,
    ):
        self._function = function
        self.type = plot_type
        self.phase_mappings = phase_mappings
        self.legend = legend
        self.combine_to = combine_to
        self.color = color
        self.linestyle = linestyle
        self.ylim = ylim
        self.bounds = bounds
        self.node_idx = node_idx
        self.label = label
        self.compute_derivative = compute_derivative
        self.integration_rule = integration_rule
        self.parameters = parameters
        self.all_variables_in_one_subplot = all_variables_in_one_subplot

    @classmethod
    def from_custom_plot(cls, custom_plot):
        from .plot import CustomPlot

        custom_plot: CustomPlot = custom_plot

        _function = None
        parameters = {}
        for key in custom_plot.parameters.keys():
            if key == "penalty":
                # This is a hack to emulate what PlotOcp._create_plots needs while not being able to actually serialize
                # the function
                parameters[key] = PenaltySerializable.from_penalty(custom_plot.parameters[key])

                penalty = custom_plot.parameters[key]

                casadi_function = penalty.function[0] if penalty.function[0] is not None else penalty.function[-1]
                size_x = casadi_function.size_in("x")[0]
                size_dt = casadi_function.size_in("dt")[0]
                size_u = casadi_function.size_in("u")[0]
                size_p = casadi_function.size_in("p")[0]
                size_a = casadi_function.size_in("a")[0]
                size_d = casadi_function.size_in("d")[0]
                _function = custom_plot.function(
                    0,  # t0
                    np.zeros(size_dt),  # phases_dt
                    custom_plot.node_idx[0],  # node_idx
                    np.zeros((size_x, 1)),  # states
                    np.zeros((size_u, 1)),  # controls
                    np.zeros((size_p, 1)),  # parameters
                    np.zeros((size_a, 1)),  # algebraic_states
                    np.zeros((size_d, 1)),  # numerical_timeseries
                    **custom_plot.parameters,  # parameters
                )

            else:
                raise NotImplementedError(f"Parameter {key} is not implemented in the serialization")

        return cls(
            function=_function,
            plot_type=custom_plot.type,
            phase_mappings=(
                None
                if custom_plot.phase_mappings is None
                else BiMappingSerializable.from_bimapping(custom_plot.phase_mappings)
            ),
            legend=custom_plot.legend,
            combine_to=custom_plot.combine_to,
            color=custom_plot.color,
            linestyle=custom_plot.linestyle,
            ylim=custom_plot.ylim,
            bounds=None if custom_plot.bounds is None else BoundsSerializable.from_bounds(custom_plot.bounds),
            node_idx=custom_plot.node_idx,
            label=custom_plot.label,
            compute_derivative=custom_plot.compute_derivative,
            integration_rule=custom_plot.integration_rule,
            parameters=parameters,
            all_variables_in_one_subplot=custom_plot.all_variables_in_one_subplot,
        )

    def serialize(self):
        return {
            "function": None if self._function is None else np.array(self._function)[:, 0].tolist(),
            "type": self.type.value,
            "phase_mappings": None if self.phase_mappings is None else self.phase_mappings.serialize(),
            "legend": self.legend,
            "combine_to": self.combine_to,
            "color": self.color,
            "linestyle": self.linestyle,
            "ylim": self.ylim,
            "bounds": None if self.bounds is None else self.bounds.serialize(),
            "node_idx": self.node_idx,
            "label": self.label,
            "compute_derivative": self.compute_derivative,
            "integration_rule": self.integration_rule.value,
            "parameters": {key: param.serialize() for key, param in self.parameters.items()},
            "all_variables_in_one_subplot": self.all_variables_in_one_subplot,
        }

    @classmethod
    def deserialize(cls, data):

        parameters = {}
        for key in data["parameters"].keys():
            if key == "penalty":
                parameters[key] = PenaltySerializable.deserialize(data["parameters"][key])
            else:
                raise NotImplementedError(f"Parameter {key} is not implemented in the serialization")

        return cls(
            function=None if data["function"] is None else DM(data["function"]),
            plot_type=PlotType(data["type"]),
            phase_mappings=(
                None if data["phase_mappings"] is None else BiMappingSerializable.deserialize(data["phase_mappings"])
            ),
            legend=data["legend"],
            combine_to=data["combine_to"],
            color=data["color"],
            linestyle=data["linestyle"],
            ylim=data["ylim"],
            bounds=None if data["bounds"] is None else BoundsSerializable.deserialize(data["bounds"]),
            node_idx=data["node_idx"],
            label=data["label"],
            compute_derivative=data["compute_derivative"],
            integration_rule=QuadratureRule(data["integration_rule"]),
            parameters=parameters,
            all_variables_in_one_subplot=data["all_variables_in_one_subplot"],
        )

    def function(self, *args, **kwargs):
        # This should not be called to get actual values, as it is evaluated at 0. This is solely to get the size of
        # the function
        return self._function


class OptimizationVariableContainerSerializable:
    node_index: int
    shape: tuple[int, int]

    def __init__(self, node_index: int, shape: tuple[int, int], len: int):
        self.node_index = node_index
        self.shape = shape
        self._len = len

    def __len__(self):
        return self._len

    @classmethod
    def from_container(cls, ovc):
        from ..optimization.optimization_variable import OptimizationVariableContainer

        ovc: OptimizationVariableContainer = ovc

        return cls(
            node_index=ovc.node_index,
            shape=ovc.shape,
            len=len(ovc),
        )

    def serialize(self):
        return {
            "node_index": self.node_index,
            "shape": self.shape,
            "len": self._len,
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            node_index=data["node_index"],
            shape=data["shape"],
            len=data["len"],
        )


class OdeSolverSerializable:
    polynomial_degree: int
    type: OdeSolver

    def __init__(self, polynomial_degree: int, type: OdeSolver):
        self.polynomial_degree = polynomial_degree
        self.type = type

    @classmethod
    def from_ode_solver(cls, ode_solver):
        from ..dynamics.ode_solver import OdeSolver

        ode_solver: OdeSolver = ode_solver

        return cls(
            polynomial_degree=5,
            type="ode",
        )

    def serialize(self):
        return {
            "polynomial_degree": self.polynomial_degree,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            polynomial_degree=data["polynomial_degree"],
            type=data["type"],
        )


class NlpSerializable:
    ns: int
    phase_idx: int

    n_states_nodes: int
    states: OptimizationVariableContainerSerializable
    states_dot: OptimizationVariableContainerSerializable
    controls: OptimizationVariableContainerSerializable
    algebraic_states: OptimizationVariableContainerSerializable
    parameters: OptimizationVariableContainerSerializable
    numerical_timeseries: OptimizationVariableContainerSerializable

    ode_solver: OdeSolverSerializable
    plot: dict[str, CustomPlotSerializable]

    def __init__(
        self,
        ns: int,
        phase_idx: int,
        n_states_nodes: int,
        states: OptimizationVariableContainerSerializable,
        states_dot: OptimizationVariableContainerSerializable,
        controls: OptimizationVariableContainerSerializable,
        algebraic_states: OptimizationVariableContainerSerializable,
        parameters: OptimizationVariableContainerSerializable,
        numerical_timeseries: OptimizationVariableContainerSerializable,
        ode_solver: OdeSolverSerializable,
        plot: dict[str, CustomPlotSerializable],
    ):
        self.ns = ns
        self.phase_idx = phase_idx
        self.n_states_nodes = n_states_nodes
        self.states = states
        self.states_dot = states_dot
        self.controls = controls
        self.algebraic_states = algebraic_states
        self.parameters = parameters
        self.numerical_timeseries = numerical_timeseries
        self.ode_solver = ode_solver
        self.plot = plot

    @classmethod
    def from_nlp(cls, nlp):
        from ..optimization.non_linear_program import NonLinearProgram

        nlp: NonLinearProgram = nlp

        return cls(
            ns=nlp.ns,
            phase_idx=nlp.phase_idx,
            n_states_nodes=nlp.n_states_nodes,
            states=OptimizationVariableContainerSerializable.from_container(nlp.states),
            states_dot=OptimizationVariableContainerSerializable.from_container(nlp.states_dot),
            controls=OptimizationVariableContainerSerializable.from_container(nlp.controls),
            algebraic_states=OptimizationVariableContainerSerializable.from_container(nlp.algebraic_states),
            parameters=OptimizationVariableContainerSerializable.from_container(nlp.parameters),
            numerical_timeseries=OptimizationVariableContainerSerializable.from_container(nlp.numerical_timeseries),
            ode_solver=OdeSolverSerializable.from_ode_solver(nlp.ode_solver),
            plot={key: CustomPlotSerializable.from_custom_plot(nlp.plot[key]) for key in nlp.plot},
        )

    def serialize(self):
        return {
            "ns": self.ns,
            "phase_idx": self.phase_idx,
            "n_states_nodes": self.n_states_nodes,
            "states": self.states.serialize(),
            "states_dot": self.states_dot.serialize(),
            "controls": self.controls.serialize(),
            "algebraic_states": self.algebraic_states.serialize(),
            "parameters": self.parameters.serialize(),
            "numerical_timeseries": self.numerical_timeseries.serialize(),
            "ode_solver": self.ode_solver.serialize(),
            "plot": {key: plot.serialize() for key, plot in self.plot.items()},
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            ns=data["ns"],
            phase_idx=data["phase_idx"],
            n_states_nodes=data["n_states_nodes"],
            states=OptimizationVariableContainerSerializable.deserialize(data["states"]),
            states_dot=OptimizationVariableContainerSerializable.deserialize(data["states_dot"]),
            controls=OptimizationVariableContainerSerializable.deserialize(data["controls"]),
            algebraic_states=OptimizationVariableContainerSerializable.deserialize(data["algebraic_states"]),
            parameters=OptimizationVariableContainerSerializable.deserialize(data["parameters"]),
            numerical_timeseries=OptimizationVariableContainerSerializable.deserialize(data["numerical_timeseries"]),
            ode_solver=OdeSolverSerializable.deserialize(data["ode_solver"]),
            plot={key: CustomPlotSerializable.deserialize(plot) for key, plot in data["plot"].items()},
        )


class SaveIterationsInfoSerializable:
    path_to_results: str
    result_file_name: str | list[str]
    nb_iter_save: int
    current_iter: int
    f_list: list[int]

    def __init__(
        self, path_to_results: str, result_file_name: str, nb_iter_save: int, current_iter: int, f_list: list[int]
    ):
        self.path_to_results = path_to_results
        self.result_file_name = result_file_name
        self.nb_iter_save = nb_iter_save
        self.current_iter = current_iter
        self.f_list = f_list

    @classmethod
    def from_save_iterations_info(cls, save_iterations_info):
        from .ipopt_output_plot import SaveIterationsInfo

        save_iterations_info: SaveIterationsInfo = save_iterations_info

        if save_iterations_info is None:
            return None

        return cls(
            path_to_results=save_iterations_info.path_to_results,
            result_file_name=save_iterations_info.result_file_name,
            nb_iter_save=save_iterations_info.nb_iter_save,
            current_iter=save_iterations_info.current_iter,
            f_list=save_iterations_info.f_list,
        )

    def serialize(self):
        return {
            "path_to_results": self.path_to_results,
            "result_file_name": self.result_file_name,
            "nb_iter_save": self.nb_iter_save,
            "current_iter": self.current_iter,
            "f_list": self.f_list,
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            path_to_results=data["path_to_results"],
            result_file_name=data["result_file_name"],
            nb_iter_save=data["nb_iter_save"],
            current_iter=data["current_iter"],
            f_list=data["f_list"],
        )


class OcpSerializable:
    n_phases: int
    nlp: list[NlpSerializable]

    time_phase_mapping: BiMappingSerializable

    plot_ipopt_outputs: bool
    plot_check_conditioning: bool
    save_ipopt_iterations_info: SaveIterationsInfoSerializable

    def __init__(
        self,
        n_phases: int,
        nlp: list[NlpSerializable],
        time_phase_mapping: BiMappingSerializable,
        plot_ipopt_outputs: bool,
        plot_check_conditioning: bool,
        save_ipopt_iterations_info: SaveIterationsInfoSerializable,
    ):
        self.n_phases = n_phases
        self.nlp = nlp

        self.time_phase_mapping = time_phase_mapping

        self.plot_ipopt_outputs = plot_ipopt_outputs
        self.plot_check_conditioning = plot_check_conditioning
        self.save_ipopt_iterations_info = save_ipopt_iterations_info

    @classmethod
    def from_ocp(cls, ocp):
        from ..optimization.optimal_control_program import OptimalControlProgram

        ocp: OptimalControlProgram = ocp

        return cls(
            n_phases=ocp.n_phases,
            nlp=[NlpSerializable.from_nlp(nlp) for nlp in ocp.nlp],
            time_phase_mapping=BiMappingSerializable.from_bimapping(ocp.time_phase_mapping),
            plot_ipopt_outputs=ocp.plot_ipopt_outputs,
            plot_check_conditioning=ocp.plot_check_conditioning,
            save_ipopt_iterations_info=SaveIterationsInfoSerializable.from_save_iterations_info(
                ocp.save_ipopt_iterations_info
            ),
        )

    def serialize(self):
        return {
            "n_phases": self.n_phases,
            "nlp": [nlp.serialize() for nlp in self.nlp],
            "time_phase_mapping": self.time_phase_mapping.serialize(),
            "plot_ipopt_outputs": self.plot_ipopt_outputs,
            "plot_check_conditioning": self.plot_check_conditioning,
            "save_ipopt_iterations_info": (
                None if self.save_ipopt_iterations_info is None else self.save_ipopt_iterations_info.serialize()
            ),
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            n_phases=data["n_phases"],
            nlp=[NlpSerializable.deserialize(nlp) for nlp in data["nlp"]],
            time_phase_mapping=BiMappingSerializable.deserialize(data["time_phase_mapping"]),
            plot_ipopt_outputs=data["plot_ipopt_outputs"],
            plot_check_conditioning=data["plot_check_conditioning"],
            save_ipopt_iterations_info=(
                None
                if data["save_ipopt_iterations_info"] is None
                else SaveIterationsInfoSerializable.deserialize(data["save_ipopt_iterations_info"])
            ),
        )
