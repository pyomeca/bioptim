import numpy as np

from ..limits.constraints import Constraint
from ..limits.objective_functions import ObjectiveFcn, ObjectiveList, Objective
from ..limits.path_conditions import Bounds
from ..misc.enums import Node, InterpolationType


class GraphAbstract:
    _return_line: ""
    _squared: ""
    """
    Methods
    -------
    _vector_layout_structure(self, vector: list | np.ndarray, decimal: int)
        Main structure of the method _vector_layout()
    _vector_layout(self, vector: list, size: int)
        Resize vector content for display task
    _add_dict_to_str(self, _dict: dict)
        Convert information contained in a dict to string
    _add_extra_parameters_to_str(self, list_params: list, string: str)
        Simple method to add extra-parameters to a string
    _lagrange_to_str(self, objective_list: ObjectiveList)
        Convert Lagrange objective into an easy-to-read string
    _mayer_to_str(self, objective_list: ObjectiveList)
        Convert Mayer objective into an easy-to-read string 
    _structure_scaling_parameter(self, el: PathCondition, parameter: Parameter)
        Main structure of the method _scaling_parameter()
    _scaling_parameter(self, parameter: Parameter)
        Take scaling into account for display task
    _get_parameter_function_name(self, parameter: Parameter)
        Get parameter function name (whether or not it is a custom function)
    _analyze_nodes(self, phase_idx: int, constraint: Constraint)
        Determine node index
    """

    def __init__(
        self,
        ocp,
    ):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        self.ocp = ocp

    def _vector_layout_structure(self, vector: list | np.ndarray, decimal: int):
        """
        Main structure of the next method _vector_layout(self, vector: list | np.ndarray, size: int, param: bool)

        Parameters
        ----------
        vector: list | np.ndarray
            The vector to be condensed
        decimal: int
            Number of decimals
        """
        condensed_vector = ""
        for i, var in enumerate(vector):
            condensed_vector += f"{round(float(var), decimal):.{decimal}f} "
            if i % 7 == 0 and i != 0:
                condensed_vector += f"... {self._return_line}... "
        return condensed_vector

    def _vector_layout(self, vector: list | np.ndarray | int):
        """
        Resize vector content for display task

        Parameters
        ----------
        vector: list | np.ndarray | int
            The vector to be condensed
        """

        condensed_vector = ""
        if isinstance(vector, int):
            condensed_vector = f"{vector}"
        else:
            vector = np.array(vector)
            if len(vector.shape) == 1:
                vector = vector[:, np.newaxis]

            if vector.shape[1] != 1:
                condensed_vector += f"{self._return_line}"
                condensed_vector += "["
            for i in range(vector.shape[1]):
                if i != 0:
                    condensed_vector += f"{self._return_line}"
                condensed_vector += "[" if vector.size > 1 else ""
                condensed_vector += self._vector_layout_structure(vector[:, i], 3)
                condensed_vector += "]" if vector.size > 1 else ""
            if vector.shape[1] != 1:
                condensed_vector += "]"

        return condensed_vector

    def _add_dict_to_str(self, _dict: dict):
        """
        Convert information contained in a dict to string

        Parameters
        ----------
        _dict: dict
            The dict to be converted
        """

        str_to_add = ""
        for d in _dict:
            str_to_add += f"<b>{d}</b>: {_dict[d]}{self._return_line}"
        return str_to_add

    def _add_extra_parameters_to_str(self, objective: Objective, string: str):
        """
        Simple method to add extra-parameters to a string

        Parameters
        ----------
        objective: Objective
            The current objective
        string: str
            The string to be completed
        """

        if hasattr(objective, "index"):
            string += f"<b>Index</b>: {objective.index}{self._return_line}" if objective.index is not None else ""
        if hasattr(objective, "weight"):
            string += f"<b>Weight</b>: {objective.weight}{self._return_line}"
        for param in objective.params:
            string += f"<b>{param}</b>: {objective.params[param]}{self._return_line}"
        string += f"<b>Index in list</b>: {objective.list_index}{self._return_line}"
        return string

    def _lagrange_to_str(self, objective_list: ObjectiveList):
        """
        Convert Lagrange objective into an easy-to-read string

        Parameters
        ----------
        objective_list: ObjectiveList
            The list of Lagrange objectives
        """

        objective_names = []
        lagrange_str = ""
        for i, obj in enumerate(objective_list):
            if not obj:
                continue

            if isinstance(obj.type, ObjectiveFcn.Lagrange):
                if obj.target is not None:
                    if obj.quadratic:
                        lagrange_str += (
                            f"({obj.name} - {self._vector_layout(obj.target[:, obj.node_idx.index(i)])})"
                            f"{self._squared}{self._return_line}"
                        )
                    else:
                        lagrange_str += (
                            f"{obj.name} - {self._vector_layout(obj.target[:, obj.node_idx.index(i)])}"
                            f"{self._return_line}"
                        )
                else:
                    if obj.quadratic:
                        lagrange_str += f"({obj.name}){self._squared}{self._return_line}"
                    else:
                        lagrange_str += f"{obj.name}{self._return_line}"
                lagrange_str = self._add_extra_parameters_to_str(obj, lagrange_str)
                lagrange_str += f"{self._return_line}"
                objective_names.append(obj.name)
        return lagrange_str, objective_names

    def _mayer_to_str(self, objective_list: ObjectiveList):
        """
        Convert Mayer objective into an easy-to-read string

        Parameters
        ----------
        objective_list: ObjectiveList
            The list of Mayer objectives
        """

        list_mayer_objectives = []
        for obj in objective_list:
            if not obj:
                continue

            for i in obj.node_idx:
                if isinstance(obj.type, ObjectiveFcn.Mayer):
                    mayer_str = ""
                    mayer_objective: list | tuple = [obj.node[0]] if isinstance(obj.node, (list, tuple)) else [obj.node]
                    if obj.target is not None:
                        if obj.quadratic:
                            mayer_str += (
                                f"({obj.name} - {self._vector_layout(obj.target[0][:, obj.node_idx.index(i)])})"
                                f"{self._squared}{self._return_line}"
                            )
                        else:
                            mayer_str += (
                                f"{obj.name} - {self._vector_layout(obj.target[0][:, obj.node_idx.index(i)])}"
                                f"{self._return_line}"
                            )
                    else:
                        if obj.quadratic:
                            mayer_str += f"({obj.name}){self._squared}{self._return_line}"
                        else:
                            mayer_str += f"{obj.name}{self._return_line}"
                    mayer_str = self._add_extra_parameters_to_str(obj, mayer_str)
                    found = False
                    for mayer in list_mayer_objectives:
                        if mayer[1] == mayer_str:
                            found = True
                    if not found:
                        mayer_objective.append(mayer_str)
                        list_mayer_objectives.append(mayer_objective)
        return list_mayer_objectives

    def _scaling_parameter(self, key: str):
        """
        Take scaling into account for display task

        Parameters
        ----------
        key: str
            The key of the parameter containing all the information to display
        """

        parameter = self.ocp.parameters[key]
        initial_guess_str = f"{self._vector_layout(self.ocp.parameter_init[key].init)}"
        min_bound_str = f"{self._vector_layout(self.ocp.parameter_bounds[key].min)}"
        max_bound_str = f"{self._vector_layout(self.ocp.parameter_bounds[key].max)}"

        scaling = [parameter.scaling[i][0] for i in range(parameter.size)]
        scaling_str = f"{self._vector_layout(scaling)}"

        return parameter, initial_guess_str, min_bound_str, max_bound_str, scaling_str

    def _analyze_nodes(self, phase_idx: int, constraint: Constraint):
        """
        Determine node index

        Parameters
        ----------
        phase_idx: int
            The index of the current phase
        constraint: Constraint
            The constraint to which the nodes to analyze is attached
        """

        if isinstance(constraint.node[0], Node):
            if constraint.node[0] != Node.ALL:
                node = self.ocp.nlp[phase_idx].ns if constraint.node[0] == Node.END else 0
            else:
                node = "all"
        else:
            node = constraint.node[0]
        return node


class OcpToConsole(GraphAbstract):
    _return_line = "\n"
    _squared = "²"
    """
    Methods
    -------
    print(self)
        Print ocp structure in the console
    print_bounds(self, phase_idx: int, bounds: Bounds, col_name: list[str])
        Print ocp bounds in the console
    print_bounds_table(bounds: Bounds, col_name: list[str], title: list[str]):
        Print bounds row

    """

    def print(self):
        """
        Print ocp structure in the console
        """
        for phase_idx in range(self.ocp.n_phases):
            # We only need to use the first index since the bounds are not depend on the dynamics
            self.ocp.nlp[phase_idx].states.node_index = 0
            self.ocp.nlp[phase_idx].states_dot.node_index = 0
            self.ocp.nlp[phase_idx].controls.node_index = 0
            self.ocp.nlp[phase_idx].stochastic_variables.node_index = 0

            print(f"PHASE: {phase_idx}")
            print(f"**********")
            print(f"BOUNDS:")
            print(f"STATES: InterpolationType.{self.ocp.nlp[phase_idx].x_bounds.type.name}")
            for key in self.ocp.nlp[phase_idx].states:
                self.print_bounds(
                    self.ocp.nlp[phase_idx].x_bounds[key],
                    [
                        self.ocp.nlp[phase_idx].states[key].cx_start[i].name()
                        for i in range(self.ocp.nlp[phase_idx].states[key].cx_start.shape[0])
                    ],
                )
            print(f"**********")
            print(f"CONTROLS: InterpolationType.{self.ocp.nlp[phase_idx].u_bounds.type.name}")
            for key in self.ocp.nlp[phase_idx].controls:
                self.print_bounds(
                    self.ocp.nlp[phase_idx].u_bounds[key],
                    [
                        self.ocp.nlp[phase_idx].controls[key].cx_start[i].name()
                        for i in range(self.ocp.nlp[phase_idx].controls[key].cx_start.shape[0])
                    ],
                )
            print(f"**********")
            print(f"PARAMETERS: ")
            print("")
            for parameter in self.ocp.nlp[phase_idx].parameters:
                parameter, initial_guess, min_bound, max_bound, scaling = self._scaling_parameter(parameter.name)
                print(f"Name: {parameter.name}")
                print(f"Size: {parameter.size}")
                print(f"Initial_guess: {initial_guess}")
                print(f"Scaling: {scaling}")
                print(f"Min_bound: {min_bound}")
                print(f"Max_bound: {max_bound}")
                print("")
            print("")
            print(f"**********")
            print(f"MODEL: {self.ocp.original_values['bio_model'][phase_idx]}")
            if isinstance(self.ocp.nlp[phase_idx].tf, (int, float)):
                print(f"PHASE DURATION: {round(self.ocp.nlp[phase_idx].tf, 2)} s")
            else:
                print(f"PHASE DURATION IS OPTIMIZED")
            print(f"SHOOTING NODES : {self.ocp.nlp[phase_idx].ns}")
            print(f"DYNAMICS: {self.ocp.nlp[phase_idx].dynamics_type.type.name}")
            print(f"ODE: {self.ocp.nlp[phase_idx].ode_solver.integrator.__name__}")
            print(f"**********")
            print("")

            mayer_objectives = self._mayer_to_str(self.ocp.nlp[phase_idx].J)
            lagrange_objectives = self._lagrange_to_str(self.ocp.nlp[phase_idx].J)[1]
            print(f"*** Lagrange: ")
            for name in lagrange_objectives:
                print(name)
            print("")
            for node_idx in range(self.ocp.nlp[phase_idx].ns):
                print(f"NODE {node_idx}")
                print(f"*** Mayer: ")
                for mayer in mayer_objectives:
                    if mayer[0] == node_idx:
                        print(mayer[1])
                for constraint in self.ocp.nlp[phase_idx].g:
                    if not constraint:
                        continue
                    node_index = self._analyze_nodes(phase_idx, constraint)
                    if node_index == node_idx:
                        print(f"*** Constraint: {constraint.name}")
                for constraint in self.ocp.nlp[phase_idx].g_implicit:
                    if not constraint:
                        continue
                    node_index = self._analyze_nodes(phase_idx, constraint)
                    if node_index == node_idx:
                        print(f"*** Implicit Constraint: {constraint.name}")
                print("")

    def print_bounds(self, bounds: Bounds, col_name: list[str]):
        """
        Print ocp bounds in the console

        Parameters
        ----------
        bounds: Bounds
            The controls or states bounds
        col_name: list[str]
            The list of controls or states name
        """

        if bounds.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            title = ["", "Beginning", "Middle", "End"]
        elif bounds.type == InterpolationType.CONSTANT:
            title = ["", "Bounds"]
        elif bounds.type == InterpolationType.LINEAR:
            title = ["", "Beginning", "End"]
        else:
            raise NotImplementedError(
                "Print bounds function has been implemented only with the following enums"
                ": InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT, "
                "InterpolationType.CONSTANT and InterpolationType.LINEAR."
            )
        self.print_bounds_table(bounds, col_name, title)

    @staticmethod
    def print_bounds_table(bounds: Bounds, col_name: list[str], title: list[str]):
        """
        Print bounds table

        Parameters
        ----------
        bounds: Bounds
            The controls or states bounds
        col_name: list[str]
            The list of states names
        title: list[str]
            The list of column's title
        """
        gap = 20  # length of a column printed in the console

        max_bounds = np.round(np.array(bounds.max.tolist()), 3)
        min_bounds = np.round(np.array(bounds.min.tolist()), 3)

        first_col_length = len(max(col_name, key=len))

        title_row = f"{title[0]}" + (first_col_length - len(title[0]) + 2) * " "
        for n in range(len(title) - 1):
            title_row += f"{title[n + 1]}" + (gap - len(title[n + 1])) * " "
        print(title_row)
        for h in range(bounds.shape[0]):
            table_row = col_name[h] + (first_col_length - len(col_name[h]) + 2) * " "
            for j in range(len(min_bounds[h])):
                row_element = f"[{min_bounds[h][j]}, {max_bounds[h][j]}]"
                row_element += (gap - row_element.__len__()) * " "
                table_row += row_element
            print(table_row)


class OcpToGraph(GraphAbstract):
    _return_line = "<br/>"
    _squared = "<sup>2</sup>"
    """
    Methods
    -------
    _constraint_to_str(self, constraint: Constraint)
        Convert constraint information into an easy-to-read string
    _global_objectives_to_str(self, objective_list: ObjectiveList)
        Convert global objectives of the ocp to string
    _draw_parameter_node(self, g: Digraph.subgraph, phase_idx: int, param_idx: int, parameter: Parameter)
        Draw the node which contains the information related to the parameters
    _draw_nlp_node(self, g: Digraph.subgraph, phase_idx: int)
        Draw the node which contains the information related to one of the phases (ie. the name of the model,
        the phase duration, the number of shooting nodes, the dynamics, the ODE)
    _draw_lagrange_node(self, g: Digraph.subgraph, phase_idx: int)
        Draw the node which contains the information related to the Lagrange objectives
    _draw_mayer_node(self, g: Digraph.subgraph, phase_idx: int)
        Draw the node which contains the information related to the Mayer objectives
    _draw_constraints_node(self, g: Digraph.subgraph, phase_idx: int)
        Draw the node which contains the information related to the constraints
    _draw_nlp_cluster(self, g: Digraph, phase_idx: int)
        Draw clusters for each nlp
    _draw_lagrange_to_mayer_edge(self, g: Digraph, phase_idx: int)
        Draw edge between Lagrange node and Mayer node
    _draw_mayer_to_constraints_edge(self, g: Digraph, phase_idx: int)
        Draw edge between Mayer node and constraints node
    _draw_nlp_to_parameters_edges(self, g: Digraph, phase_idx: int)
        Draw edges between nlp node and parameters
    _draw_phase_trans_to_phaseless_edge(self, g: Digraph, phaseless_objectives: str, nb_phase: int)
        Draw edge between phaseless objectives and phase transitions
    _draw_edges(self, g: Digraph, phase_idx: int)
        Draw edges between each node of a cluster
    _draw_phase_transitions(self, g: Digraph)
        Draw a cluster including all the information about the phase transitions of the problem
    _draw_phaseless_cluster(self, g:Digraph)
        Draw a cluster including all the information about the phaseless objectives of the problem
    """

    def _prepare_print(self):
        from graphviz import Digraph

        """
        Display ocp structure in a graph
        """

        # Initialize graph with graphviz
        g = Digraph("ocp_graph", node_attr={"shape": "plaintext"})

        # Draw OCP node
        g.node("OCP", shape="Mdiamond")

        # Draw nlp clusters and edges
        for phase_idx in range(self.ocp.n_phases):
            self._draw_nlp_cluster(g, phase_idx)
            self._draw_edges(g, phase_idx)

        # Draw phase_transitions
        self._draw_phase_transitions(g)

        # Draw phaseless objectives
        self._draw_phaseless_cluster(g)
        self._draw_phase_trans_to_phaseless_edge(g, self._global_objectives_to_str(self.ocp.J)[0], self.ocp.n_phases)
        return g

    def print(self):
        # Display graph
        self._prepare_print().view()

    def _constraint_to_str(self, constraint: Constraint):
        """
        Convert constraint information into an easy-to-read string

        Parameters
        ----------
        constraint: Constraint
            The constraint to be converted
        """

        constraint_str = ""
        constraint_str += f"{constraint.name}<br/>"
        constraint_str += f"<b>Min bound</b>: {constraint.min_bound}<br/>"
        constraint_str += f"<b>Max bound</b>: {constraint.max_bound}<br/>"
        constraint_str += (
            f"{f'<b>Target</b>: {self._vector_layout(constraint.target)} <br/>'}"
            if constraint.target is not None
            else ""
        )
        constraint_str += self._add_dict_to_str(constraint.params)
        constraint_str += f"<b>Index in list</b>: {constraint.list_index}<br/>"
        return constraint_str

    def _global_objectives_to_str(self, objective_list: ObjectiveList):
        """
        Convert global objectives of the ocp to string

        Parameters
        ----------
        objective_list: ObjectiveList
            The list of global objectives to be converted
        """

        global_objectives = ""
        global_objectives_names = []
        for objective in objective_list:
            if objective:
                name = objective.custom_function.__name__ if objective.custom_function else objective.name
                global_objectives += f"<b>Objective:</b> {name} <br/>"
                global_objectives += f"<b>Type:</b> {objective.type} <br/>"
                global_objectives_names += name
                global_objectives += (
                    f"{f'<b>Target</b>: {self._vector_layout(objective.target)} <br/>'}"
                    if objective.target is not None
                    else ""
                )
                global_objectives += f"<b>Quadratic</b>: {objective.quadratic} <br/><br/>"
        return global_objectives, global_objectives_names

    def _draw_parameter_node(self, g, phase_idx: int, param_idx: int, key: str):
        """
        Draw the node which contains the information related to the parameters

        Parameters
        ----------
        g: Digraph.subgraph
            The subgraph to which the node is attached
        phase_idx: int
            The index of the current phase
        param_idx: int
            The index of the parameter
        key: str
            The key of the parameter containing all the information to display
        """

        parameter, initial_guess, min_bound, max_bound, scaling = self._scaling_parameter(key)

        node_str = f"<u><b>{parameter.name[0].upper() + parameter.name[1:]}</b></u><br/>"
        node_str += f"<b>Size</b>: {parameter.size}<br/>"
        node_str += f"<b>Scaling</b>: {scaling}<br/>"
        node_str += f"<b>Initial guess</b>: {initial_guess}<br/>"
        node_str += f"<b>Min bound</b>: {min_bound} <br/>"
        node_str += f"<b>Max bound</b>: {max_bound} <br/><br/>"
        g.node(f"param_{phase_idx}{param_idx}", f"""<{node_str}>""")

    def _draw_nlp_node(self, g, phase_idx: int):
        """
        Draw the node which contains the information related to one of the phases (ie. the name of the model,
        the phase duration, the number of shooting nodes, the dynamics, the ODE)

        Parameters
        ----------
        g: Digraph.subgraph
            The subgraph to which the node is attached
        phase_idx: int
            The index of the current phase
        """

        node_str = f"<b>BioModel</b>: {type(self.ocp.nlp[phase_idx].model)}<br/>"
        if isinstance(self.ocp.nlp[phase_idx].tf, (int, float)):
            node_str += f"<b>Phase duration</b>: {round(self.ocp.nlp[phase_idx].tf, 2)} s<br/>"
        else:
            node_str += f"<b>Phase duration</b>: optimized<br/>"
        node_str += f"<b>Shooting nodes</b>: {self.ocp.nlp[phase_idx].ns}<br/>"
        node_str += f"<b>Dynamics</b>: {self.ocp.nlp[phase_idx].dynamics_type.type.name}<br/>"
        node_str += f"<b>ODE</b>: {self.ocp.nlp[phase_idx].ode_solver.rk_integrator.__name__}<br/>"
        node_str += f"<b>Control type</b>: {self.ocp.nlp[phase_idx].control_type.name}"
        g.node(f"nlp_node_{phase_idx}", f"""<{node_str}>""")

    def _draw_lagrange_node(self, g, phase_idx: int):
        """
        Draw the node which contains the information related to the Lagrange objectives

        Parameters
        ----------
        g: Digraph.subgraph
            The subgraph to which the node is attached
        phase_idx: int
            The index of the current phase
        """

        lagrange_str = self._lagrange_to_str(self.ocp.nlp[phase_idx].J)[0]
        node_str = f"<u><b>Lagrange</b></u><br/>{lagrange_str}"
        g.node(f"lagrange_{phase_idx}", f"""<{node_str}>""")

    def _draw_mayer_node(self, g, phase_idx: int):
        """
        Draw the node which contains the information related to the Mayer objectives

        Parameters
        ----------
        g: Digraph.subgraph
            The subgraph to which the node is attached
        phase_idx: int
            The index of the current phase
        """

        list_mayer_objectives = self._mayer_to_str(self.ocp.nlp[phase_idx].J)
        all_mayer_str = "<u><b>Mayer</b></u><br/>"
        if len(list_mayer_objectives) != 0:
            for objective in list_mayer_objectives:
                all_mayer_str += objective[1]
                all_mayer_str += f"<b>Shooting nodes index</b>: {objective[0]}<br/><br/>"
        else:
            all_mayer_str += "No Mayer set"
        g.node(f"mayer_node_{phase_idx}", f"""<{all_mayer_str}>""")

    def _draw_constraints_node(self, g, phase_idx: int):
        """
        Draw the node which contains the information related to the constraints

        Parameters
        ----------
        g: Digraph.subgraph
            The subgraph to which the node is attached
        phase_idx: int
            The index of the current phase
        """

        list_constraints = []

        for constraint in self.ocp.nlp[phase_idx].g:
            if not constraint:
                continue
            constraints_str = ""
            node_index = self._analyze_nodes(phase_idx, constraint)
            constraints_str += self._constraint_to_str(constraint)
            list_constraints.append([constraints_str, node_index])

        all_constraints_str = "<u><b>Constraints</b></u><br/>"
        if len(list_constraints) != 0:
            for constraint in list_constraints:
                all_constraints_str += constraint[0]
                all_constraints_str += f"<b>Shooting nodes index</b>: {constraint[1]}<br/><br/>"
        else:
            all_constraints_str += "No constraint set"
        g.node(f"constraints_node_{phase_idx}", f"""<{all_constraints_str}>""")

    def _draw_nlp_cluster(self, g, phase_idx: int):
        """
        Draw clusters for each nlp

        Parameters
        ----------
        phase_idx: int
            The index of the current phase
        g: Digraph
            The graph to be completed
        """

        with g.subgraph(name=f"cluster_{phase_idx}") as g:
            g.attr(style="filled", color="lightgrey")
            nlp_title = f"<u><b>Phase #{phase_idx}</b></u>"
            g.attr(label=f"<{nlp_title}>")
            g.node_attr.update(style="filled", color="white")

            self._draw_nlp_node(g, phase_idx)

            if len(self.ocp.nlp[phase_idx].parameters) > 0:
                param_idx = 0
                for param in self.ocp.nlp[phase_idx].parameters:
                    self._draw_parameter_node(g, phase_idx, param_idx, param.name)
                    param_idx += 1
            else:
                node_str = "<u><b>Parameters</b></u><br/> No parameter set"
                g.node(f"param_{phase_idx}0", f"""<{node_str}>""")

            only_mayer = True
            for objective in self.ocp.nlp[phase_idx].J:
                if not objective:
                    continue
                if isinstance(objective.type, ObjectiveFcn.Lagrange):
                    only_mayer = False

            if len(self.ocp.nlp[phase_idx].J) > 0 and not only_mayer:
                self._draw_lagrange_node(g, phase_idx)
            else:
                node_str = "<u><b>Lagrange</b></u><br/> No Lagrange set"
                g.node(f"lagrange_{phase_idx}", f"""<{node_str}>""")

            self._draw_mayer_node(g, phase_idx)
            self._draw_constraints_node(g, phase_idx)

    @staticmethod
    def _draw_lagrange_to_mayer_edge(g, phase_idx: int):
        """
        Draw edge between Lagrange node and Mayer node

        Parameters
        ----------
        g: Digraph
            The graph to be modified
        phase_idx: int
            The index of the current phase
        """

        g.edge(f"lagrange_{phase_idx}", f"mayer_node_{phase_idx}", color="lightgrey")

    @staticmethod
    def _draw_mayer_to_constraints_edge(g, phase_idx: int):
        """
        Draw edge between Mayer node and constraints node

        Parameters
        ----------
        g: Digraph
            The graph to be modified
        phase_idx: int
            The index of the current phase
        """
        g.edge(f"mayer_node_{phase_idx}", f"constraints_node_{phase_idx}", color="lightgrey")

    def _draw_nlp_to_parameters_edges(self, g, phase_idx: int):
        """
        Draw edges between nlp node and parameters

        Parameters
        ----------
        g: Digraph
            The graph to be modified
        phase_idx: int
            The index of the current phase
        """

        nb_parameters = len(self.ocp.nlp[phase_idx].parameters)
        g.edge(f"nlp_node_{phase_idx}", f"param_{phase_idx}0", color="lightgrey")
        for param_idx in range(nb_parameters):
            if param_idx >= 1:
                g.edge(f"param_{phase_idx}{param_idx - 1}", f"param_{phase_idx}{param_idx}", color="lightgrey")
        if nb_parameters > 1:
            g.edge(f"param_{phase_idx}{nb_parameters - 1}", f"lagrange_{phase_idx}", color="lightgrey")
        else:
            g.edge(f"param_{phase_idx}0", f"lagrange_{phase_idx}", color="lightgrey")

    @staticmethod
    def _draw_phase_trans_to_phaseless_edge(g, phaseless_objectives: str, nb_phase: int):
        """
        Draw edge between phaseless objectives and phase transitions

        Parameters
        ----------
        g: Digraph
            The graph to be modified
        phaseless_objectives: str
            The phaseless objectives converted to string
        nb_phase: int
            The number of phases
        """

        if nb_phase > 1 and phaseless_objectives != "":
            g.edge(f"phaseless_objectives", f"Phase #0", color="invis")

    def _draw_edges(self, g, phase_idx: int):
        """
        Draw edges between each node of a cluster

        Parameters
        ----------
        phase_idx: int
            The index of the current phase
        g: Digraph
            The graph to be completed
        """

        # Draw edges between OCP node and each nlp cluster
        g.edge("OCP", f"nlp_node_{phase_idx}")

        self._draw_nlp_to_parameters_edges(g, phase_idx)
        self._draw_lagrange_to_mayer_edge(g, phase_idx)
        self._draw_mayer_to_constraints_edge(g, phase_idx)

    def _draw_phase_transitions(self, g):
        """
        Draw a cluster including all the information about the phase transitions of the problem

        Parameters
        ----------
        g: Digraph
            The graph to be completed
        """

        with g.subgraph(name=f"cluster_phase_transitions") as g:
            g.attr(style="", color="invis")
            g.node_attr.update(style="filled", color="grey")
            for phase_idx in range(self.ocp.n_phases):
                if phase_idx != self.ocp.n_phases - 1:
                    g.node(f"Phase #{phase_idx}")
                    g.node(f"Phase #{phase_idx + 1}")
                    g.edge(
                        f"Phase #{phase_idx}",
                        f"Phase #{phase_idx + 1}",
                        label=self.ocp.phase_transitions[phase_idx].type.name,
                    )
            title = f"<u><b>Phase transitions</b></u>"
            g.attr(label=f"<{title}>")

    def _draw_phaseless_cluster(self, g):
        """
        Draw a cluster including all the information about the phaseless objectives of the problem

        Parameters
        ----------
        g: Digraph
            The graph to be completed
        """

        phaseless_objectives = self._global_objectives_to_str(self.ocp.J)[0]
        if phaseless_objectives != "":
            with g.subgraph(name=f"cluster_phaseless_objectives") as g:
                g.attr(style="filled", color="lightgrey")
                nlp_title = f"<u><b>Phaseless objectives</b></u>"
                g.attr(label=f"<{nlp_title}>")
                g.node_attr.update(style="filled", color="white")
                g.node(f"phaseless_objectives", f"""<{phaseless_objectives}>""")
            g.edge("OCP", f"phaseless_objectives", color="invis")
