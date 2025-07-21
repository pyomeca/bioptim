from typing import Any

from casadi import Function
import numpy as np

from ..dynamics.ode_solvers import OdeSolver
from ..limits.penalty_option import PenaltyOption
from ..limits.path_conditions import Bounds
from ..misc.mapping import BiMapping
from ..misc.enums import PlotType, QuadratureRule, InterpolationType
from ..misc.parameters_types import (
    IntDict,
    IntList,
    AnyDict,
    Bool,
    Str,
    Int,
    IntOptional,
    AnyList,
    AnyIterable,
    AnyIterableOrSlice,
    DoubleIntTuple,
    StrOrIterable,
    NpArray,
    FloatList,
    StrOptional,
    DoubleFloatTuple,
    IntIterableOptional,
    StrIterableOptional,
    StrListOptional,
)


class CasadiFunctionSerializable:
    _size_in: IntDict

    def __init__(self, size_in: IntDict):
        self._size_in = size_in

    @classmethod
    def from_casadi_function(cls, casadi_function: Function) -> "CasadiFunctionSerializable":
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

    def serialize(self) -> AnyDict:
        return {
            "size_in": self._size_in,
        }

    @classmethod
    def deserialize(cls, data: AnyDict) -> "CasadiFunctionSerializable":
        return cls(
            size_in=data["size_in"],
        )

    def size_in(self, key: Str) -> Int:
        return self._size_in[key]


class PenaltySerializable:
    function: list[CasadiFunctionSerializable | None]

    def __init__(self, function: list[CasadiFunctionSerializable | None]):
        self.function = function

    @classmethod
    def from_penalty(cls, penalty: PenaltyOption):
        penalty: PenaltyOption = penalty

        function = []
        for f in penalty.function:
            function.append(None if f is None else CasadiFunctionSerializable.from_casadi_function(f))
        return cls(
            function=function,
        )

    def serialize(self) -> AnyDict:
        return {
            "function": [None if f is None else f.serialize() for f in self.function],
        }

    @classmethod
    def deserialize(cls, data: AnyDict) -> "PenaltySerializable":
        return cls(
            function=[None if f is None else CasadiFunctionSerializable.deserialize(f) for f in data["function"]],
        )


class MappingSerializable:
    map_idx: IntList
    oppose: IntList

    def __init__(self, map_idx: IntList, oppose: IntList):
        self.map_idx: IntList = map_idx
        self.oppose: IntList = oppose

    def map(self, obj: Any):
        from ..misc.mapping import Mapping

        return Mapping.map(self, obj)

    @classmethod
    def from_mapping(cls, mapping) -> "MappingSerializable":
        return cls(
            map_idx=mapping.map_idx,
            oppose=mapping.oppose,
        )

    def serialize(self) -> AnyDict:
        return {
            "map_idx": list(self.map_idx),
            "oppose": self.oppose,
        }

    @classmethod
    def deserialize(cls, data: AnyDict):
        return cls(
            map_idx=data["map_idx"],
            oppose=data["oppose"],
        )


class BiMappingSerializable:
    to_first: MappingSerializable
    to_second: MappingSerializable

    def __init__(self, to_first: MappingSerializable, to_second: MappingSerializable):
        self.to_first: MappingSerializable = to_first
        self.to_second: MappingSerializable = to_second

    @classmethod
    def from_bimapping(cls, bimapping) -> "BiMappingSerializable":
        return cls(
            to_first=MappingSerializable.from_mapping(bimapping.to_first),
            to_second=MappingSerializable.from_mapping(bimapping.to_second),
        )

    def serialize(self) -> AnyDict:
        return {
            "to_first": self.to_first.serialize(),
            "to_second": self.to_second.serialize(),
        }

    @classmethod
    def deserialize(cls, data: AnyDict) -> "BiMappingSerializable":
        return cls(
            to_first=MappingSerializable.deserialize(data["to_first"]),
            to_second=MappingSerializable.deserialize(data["to_second"]),
        )


class BoundsSerializable:
    _bounds: Bounds

    def __init__(self, bounds: Bounds):
        self._bounds: Bounds = bounds

    @classmethod
    def from_bounds(cls, bounds: Bounds):
        return cls(bounds=bounds)

    def serialize(self) -> AnyDict:
        slice_list = self._bounds.min.slice_list  # min and max have the same slice_list
        slice_list_type = type(slice_list).__name__
        if isinstance(self._bounds.min.slice_list, slice):
            slice_list = [slice_list.start, slice_list.stop, slice_list.step]

        return {
            "key": self._bounds.key,
            "min": np.array(self._bounds.min).tolist(),
            "max": np.array(self._bounds.max).tolist(),
            "type": self._bounds.type.value,
            "slice_list_type": slice_list_type,
            "slice_list": slice_list,
        }

    @classmethod
    def deserialize(cls, data: AnyDict) -> "BoundsSerializable":
        return cls(
            bounds=Bounds(
                key=data["key"],
                min_bound=data["min"],
                max_bound=data["max"],
                interpolation=InterpolationType(data["type"]),
                slice_list=(
                    slice(data["slice_list"][0], data["slice_list"][1], data["slice_list"][2])
                    if data["slice_list_type"] == "slice"
                    else data["slice_list"]
                ),
            ),
        )

    def check_and_adjust_dimensions(self, n_elements: Int, n_shooting: Int) -> None:
        self._bounds.check_and_adjust_dimensions(n_elements, n_shooting)

    def type(self) -> InterpolationType:
        return self._bounds.type

    @property
    def min(self) -> NpArray:
        return self._bounds.min

    @property
    def max(self) -> NpArray:
        return self._bounds.max


class CustomPlotSerializable:
    type: PlotType
    phase_mappings: BiMappingSerializable
    legend: StrIterableOptional
    combine_to: StrOptional
    color: StrOptional
    linestyle: StrOptional
    ylim: DoubleFloatTuple | FloatList
    bounds: BoundsSerializable
    node_idx: IntIterableOptional
    label: StrListOptional
    compute_derivative: Bool
    integration_rule: QuadratureRule
    all_variables_in_one_subplot: Bool

    def __init__(
        self,
        plot_type: PlotType,
        phase_mappings: BiMappingSerializable,
        legend: StrIterableOptional,
        combine_to: StrOptional,
        color: StrOptional,
        linestyle: StrOptional,
        ylim: DoubleFloatTuple | FloatList,
        bounds: BoundsSerializable,
        node_idx: IntIterableOptional,
        label: StrListOptional,
        compute_derivative: Bool,
        integration_rule: QuadratureRule,
        all_variables_in_one_subplot: Bool,
    ):
        self.type: PlotType = plot_type
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
        self.all_variables_in_one_subplot = all_variables_in_one_subplot

    @classmethod
    def from_custom_plot(cls, custom_plot) -> "CustomPlotSerializable":
        from .plot import CustomPlot

        custom_plot: CustomPlot = custom_plot

        return cls(
            plot_type=custom_plot.type,
            phase_mappings=BiMappingSerializable.from_bimapping(custom_plot.phase_mappings),
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
            all_variables_in_one_subplot=custom_plot.all_variables_in_one_subplot,
        )

    def serialize(self) -> AnyDict:
        return {
            "type": self.type.value,
            "phase_mappings": self.phase_mappings.serialize(),
            "legend": self.legend,
            "combine_to": self.combine_to,
            "color": self.color,
            "linestyle": self.linestyle,
            "ylim": self.ylim,
            "bounds": None if self.bounds is None else self.bounds.serialize(),
            "node_idx": list(self.node_idx),
            "label": self.label,
            "compute_derivative": self.compute_derivative,
            "integration_rule": self.integration_rule.value,
            "all_variables_in_one_subplot": self.all_variables_in_one_subplot,
        }

    @classmethod
    def deserialize(cls, data: AnyDict) -> "CustomPlotSerializable":
        return cls(
            plot_type=PlotType(data["type"]),
            phase_mappings=BiMappingSerializable.deserialize(data["phase_mappings"]),
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
            all_variables_in_one_subplot=data["all_variables_in_one_subplot"],
        )

    def function(self, *args, **kwargs) -> callable:
        # This should not be called to get actual values, as it is evaluated at 0. This is solely to get the size of
        # the function
        return self._function


class OptimizationVariableContainerSerializable:
    node_index: Int
    shape: DoubleIntTuple

    def __init__(self, node_index: Int, shape: DoubleIntTuple, len: Int):
        self.node_index: Int = node_index
        self.shape: DoubleIntTuple = shape
        self._len: Int = len

    def __len__(self) -> Int:
        return self._len

    @classmethod
    def from_container(cls, ovc) -> "OptimizationVariableContainerSerializable":
        from ..optimization.optimization_variable import OptimizationVariableContainer

        ovc: OptimizationVariableContainer = ovc

        return cls(
            node_index=ovc.node_index,
            shape=ovc.shape,
            len=len(ovc),
        )

    def serialize(self) -> AnyDict:
        return {
            "node_index": self.node_index,
            "shape": self.shape,
            "len": self._len,
        }

    @classmethod
    def deserialize(cls, data) -> "OptimizationVariableContainerSerializable":
        return cls(
            node_index=data["node_index"],
            shape=data["shape"],
            len=data["len"],
        )


class OdeSolverSerializable:
    # TODO There are probably more parameters to serialize here, if the GUI fails, this is probably the reason
    polynomial_degree: IntOptional
    n_integration_steps: IntOptional
    type: OdeSolver

    def __init__(self, polynomial_degree: IntOptional, n_integration_steps: IntOptional, type: OdeSolver):
        self.polynomial_degree = polynomial_degree
        self.n_integration_steps = n_integration_steps
        self.type = type

    @classmethod
    def from_ode_solver(cls, ode_solver) -> "OdeSolverSerializable":
        from ..dynamics.ode_solvers import OdeSolver

        ode_solver: OdeSolver = ode_solver

        return cls(
            polynomial_degree=ode_solver.polynomial_degree if hasattr(ode_solver, "polynomial_degree") else None,
            n_integration_steps=ode_solver.n_integration_steps if hasattr(ode_solver, "n_integration_steps") else None,
            type="ode",
        )

    def serialize(self) -> AnyDict:
        return {
            "polynomial_degree": self.polynomial_degree,
            "n_integration_steps": self.n_integration_steps,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, data: AnyDict) -> "OdeSolverSerializable":
        return cls(
            polynomial_degree=data["polynomial_degree"],
            n_integration_steps=data["n_integration_steps"],
            type=data["type"],
        )


class SaveIterationsInfoSerializable:
    path_to_results: Str
    result_file_name: StrOrIterable
    nb_iter_save: Int
    current_iter: Int
    f_list: IntList

    def __init__(
        self, path_to_results: Str, result_file_name: Str, nb_iter_save: Int, current_iter: Int, f_list: IntList
    ):
        self.path_to_results = path_to_results
        self.result_file_name = result_file_name
        self.nb_iter_save = nb_iter_save
        self.current_iter = current_iter
        self.f_list = f_list

    @classmethod
    def from_save_iterations_info(cls, save_iterations_info) -> "SaveIterationsInfoSerializable":
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

    def serialize(self) -> AnyDict:
        return {
            "path_to_results": self.path_to_results,
            "result_file_name": self.result_file_name,
            "nb_iter_save": self.nb_iter_save,
            "current_iter": self.current_iter,
            "f_list": self.f_list,
        }

    @classmethod
    def deserialize(cls, data: AnyDict) -> "SaveIterationsInfoSerializable":
        return cls(
            path_to_results=data["path_to_results"],
            result_file_name=data["result_file_name"],
            nb_iter_save=data["nb_iter_save"],
            current_iter=data["current_iter"],
            f_list=data["f_list"],
        )


class NlpSerializable:
    ns: Int
    phase_idx: Int

    n_states_nodes: Int
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
        ns: Int,
        phase_idx: Int,
        n_states_nodes: Int,
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
    def from_nlp(cls, nlp) -> "NlpSerializable":

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
            plot={key: CustomPlotSerializable.from_custom_plot(nlp.plot[key]) for key in nlp.plot},
            ode_solver=OdeSolverSerializable.from_ode_solver(nlp.dynamics_type.ode_solver),
        )

    def serialize(self) -> AnyDict:
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
    def deserialize(cls, data: AnyDict) -> "NlpSerializable":
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


class OcpSerializable:
    n_phases: Int
    nlp: list[NlpSerializable]

    time_phase_mapping: BiMappingSerializable

    plot_ipopt_outputs: Bool
    plot_check_conditioning: Bool
    save_ipopt_iterations_info: SaveIterationsInfoSerializable

    def __init__(
        self,
        n_phases: Int,
        nlp: list[NlpSerializable],
        time_phase_mapping: BiMappingSerializable,
        plot_ipopt_outputs: Bool,
        plot_check_conditioning: Bool,
        save_ipopt_iterations_info: SaveIterationsInfoSerializable,
    ):
        self.n_phases = n_phases
        self.nlp = nlp

        self.time_phase_mapping = time_phase_mapping

        self.plot_ipopt_outputs = plot_ipopt_outputs
        self.plot_check_conditioning = plot_check_conditioning
        self.save_ipopt_iterations_info = save_ipopt_iterations_info

    @classmethod
    def from_ocp(cls, ocp) -> "OcpSerializable":
        from ..optimization.optimal_control_program import OptimalControlProgram

        ocp: OptimalControlProgram = ocp
        ocp.finalize_plot_phase_mappings()

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

    def serialize(self) -> AnyDict:
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
    def deserialize(cls, data: AnyDict) -> "OcpSerializable":
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

    def finalize_plot_phase_mappings(self) -> None:
        """
        This method can't be actually called from the serialized version, but we still can check if the work is done
        """

        for nlp in self.nlp:
            if not nlp.plot:
                continue

            for key in nlp.plot:
                if nlp.plot[key].phase_mappings is None:
                    raise RuntimeError("The phase mapping should be set on client side")
