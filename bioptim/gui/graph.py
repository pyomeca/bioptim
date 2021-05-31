from graphviz import Digraph

from ..limits.constraints import Constraint
from ..limits.objective_functions import ObjectiveFcn, ObjectiveList
from ..optimization.parameters import Parameter


class GraphAbstract:
    _return_line: ""
    _squared: ""

    def __init__(
        self,
        ocp,
    ):
        self.ocp = ocp

    def _add_dict_to_str(self, _dict: dict):
        str_to_add = ""
        for d in _dict:
            str_to_add += f"{d}: {_dict[d]}{self._return_line}"
        return str_to_add

    def _constraint_to_str(self, constraint: Constraint):
        """
        Convert constraint information into an easy-to-read string

        Parameters
        ----------
        constraint: Constraint
            The constraint to be converted
        """

        constraint_str = ""
        target_str = "" if constraint.sliced_target is None else f"{constraint.sliced_target}"
        if constraint.quadratic:
            constraint_str += f"{constraint.min_bound} ≤ "
            constraint_str += f"({constraint.name}" if target_str != "" else f"{constraint.name}"
            constraint_str += f" - {target_str}){self._squared}" if target_str != "" else ""
            constraint_str += f" ≤ {constraint.max_bound}{self._return_line}"
        else:
            constraint_str += f"{constraint.min_bound} ≤ {constraint.name}"
            constraint_str += f" - {target_str}" if target_str != "" else ""
            constraint_str += f" ≤ {constraint.max_bound}{self._return_line}"
        constraint_str += self._add_dict_to_str(constraint.params)
        return constraint_str

    def _add_extra_parameters_to_str(self, list_params: list, string: str):
        """
        Simple method to add extra-parameters to a string

        Parameters
        ----------
        list_params: list
            The list of parameters to add to the string
        string: str
            The string to be completed
        """

        for param in list_params:
            string += f"{param}: {list_params[param]}{self._return_line}"
        string += f"{self._return_line}"
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
        for objective in objective_list:
            if len(objective) > 0:
                obj = objective[0]["objective"]
                if isinstance(obj.type, ObjectiveFcn.Lagrange):
                    if obj.sliced_target is not None:
                        if obj.quadratic:
                            lagrange_str += f"({obj.name} - {obj.sliced_target}){self._squared}{self._return_line}"
                        else:
                            lagrange_str += f"{obj.name} - {obj.sliced_target}{self._return_line}"
                    else:
                        if obj.quadratic:
                            lagrange_str += f"({obj.name}){self._squared}{self._return_line}"
                        else:
                            lagrange_str += f"{obj.name}{self._return_line}"
                    lagrange_str = self._add_extra_parameters_to_str(obj.params, lagrange_str)
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
        for objective in objective_list:
            for obj_index in objective:
                obj = obj_index["objective"]
                if isinstance(obj.type, ObjectiveFcn.Mayer):
                    mayer_str = ""
                    mayer_objective = [obj.node[0]]
                    if obj.sliced_target is not None:
                        if obj.quadratic:
                            mayer_str += f"({obj.name} - {obj.sliced_target}){self._squared}"
                        else:
                            mayer_str += f"{obj.name} - {obj.sliced_target}"
                    else:
                        if obj.quadratic:
                            mayer_str += f"({obj.name}){self._squared}"
                        else:
                            mayer_str += f"{obj.name}"
                    mayer_str = self._add_extra_parameters_to_str(obj.params, mayer_str)
                    found = False
                    for mayer in list_mayer_objectives:
                        if mayer[1] == mayer_str:
                            found = True
                    if not found:
                        mayer_objective.append(mayer_str)
                        list_mayer_objectives.append(mayer_objective)
        return list_mayer_objectives

    def _scaling_parameter(self, parameter: Parameter):
        """
        Take scaling into account for display task

        Parameters
        ----------
        parameter: Parameter
            The unscaled parameter
        """

        size_initial_guess = len(parameter.initial_guess.init[0])
        initial_guess = [
            parameter.initial_guess.init[i][j] * parameter.scaling[i]
            for i in range(parameter.size)
            for j in range(size_initial_guess)
        ]
        size_min_bound = len(parameter.bounds.min[0])
        min_bound = [
            parameter.bounds.min[i][j] * parameter.scaling[i]
            for i in range(parameter.size)
            for j in range(size_min_bound)
        ]
        size_max_bound = len(parameter.bounds.max[0])
        max_bound = [
            parameter.bounds.max[i][j] * parameter.scaling[i]
            for i in range(parameter.size)
            for j in range(size_max_bound)
        ]
        return initial_guess, min_bound, max_bound

    def _get_parameter_function_name(self, parameter: Parameter):
        """
        Get parameter function name (whether or not it is a custom function)

        Parameters
        ----------
        parameter: Parameter
            The parameter to which the function is linked
        """
        name = ""
        if parameter.penalty_list is not None:
            if parameter.penalty_list.type.name == "CUSTOM":
                name = parameter.penalty_list.custom_function.__name__
            else:
                name = parameter.penalty_list.name
        return name

    def _analyze_nodes(self, phase_idx: int, constraint: Constraint):
        if constraint["constraint"].node[0].value != "all":
            node = self.ocp.nlp[phase_idx].ns if constraint["constraint"].node[0].value == "end" else 0
        else:
            node = "all"
        return node


class OcpToConsole(GraphAbstract):
    _return_line = "\n"
    _squared = "²"
    """
    Methods
    -------
    print_to_console(self)
        Print ocp structure in the console
    constraint_to_str(constraint: Constraint)
        Convert constraint information into an easy-to-read string
    add_extra_parameters_to_str(list_constraints: list, string: str)
        Simple method to add extra-parameters to a string
    lagrange_to_str(objective_list: ObjectiveList)
        Convert Lagrange objective into an easy-to-read string
    mayer_to_str(objective_list: ObjectiveList)
        Convert Mayer objective into an easy-to-read string
    vector_layout(vector: list, size: int)
        Resize vector content for display task
    scaling_parameter(parameter: Parameter)
        Take scaling into account for display task
    get_parameter_function_name(parameter: Parameter)
        Get parameter function name (whether or not it is a custom function)
    """

    def print(self):
        """
        Print ocp structure in the console
        """

        for phase_idx in range(self.ocp.n_phases):
            print(f"PHASE {phase_idx}")
            print(f"**********")
            print(f"PARAMETERS: ")
            print("")
            for parameter in self.ocp.nlp[phase_idx].parameters:
                initial_guess, min_bound, max_bound = self._scaling_parameter(parameter)
                objective_name = self._get_parameter_function_name(parameter)
                print(f"Name: {parameter.name}")
                print(f"Size: {parameter.size}")
                print(f"Initial_guess: {initial_guess}")
                print(f"Min_bound: {min_bound}")
                print(f"Max_bound: {max_bound}")
                print(f"Objectives: {objective_name}")
                print("")
            print("")
            print(f"**********")
            print(f"MODEL: {self.ocp.original_values['biorbd_model'][phase_idx]}")
            print(f"PHASE DURATION: {round(self.ocp.nlp[phase_idx].t_initial_guess, 2)} s")
            print(f"SHOOTING NODES : {self.ocp.nlp[phase_idx].ns}")
            print(f"DYNAMICS: {self.ocp.nlp[phase_idx].dynamics_type.type.name}")
            print(f"ODE: {self.ocp.nlp[phase_idx].ode_solver.rk_integrator.__name__}")
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
                    if constraint[0]["node_index"] == -1:
                        node_index = self._analyze_nodes(phase_idx, constraint[0])
                    else:
                        node_index = constraint["node_index"]
                    if node_index == node_idx:
                        print(f"*** Constraint: {constraint[0]['constraint'].name}")
                print("")


class OcpToGraph(GraphAbstract):
    _return_line = "<br/>"
    _squared = "<sup>2</sup>"
    """
    Methods
    -------
    print_to_graph(self)
        Display ocp structure in a graph
    draw_nlp_cluster(self, phase_idx: int, G: Digraph)
        Draw clusters for each nlp
    draw_edges(self, phase_idx: int, G:Digraph)
        Draw edges between each node of a cluster
    display_phase_transitions(self, G:Digraph)
        Draw a cluster including all the information about the phase transitions of the problem
    """

    def print(self):
        """
        Display ocp structure in a graph
        """

        # Initialize graph with graphviv
        G = Digraph("ocp_graph", node_attr={"shape": "plaintext"})

        # Draw OCP node
        G.node("OCP", shape="Mdiamond")

        # Draw nlp clusters and edges
        for phase_idx in range(self.ocp.n_phases):
            self._draw_nlp_cluster(G, phase_idx)
            self._draw_edges(G, phase_idx)

        # Draw phase_transitions
        self._draw_phase_transitions(G)

        # Display graph
        G.view()

    def _vector_layout(self, vector: list, size: int):
        """
        Resize vector content for display task

        Parameters
        ----------
        vector: list
            The vector to be condensed
        size: int
            The size of the vector
        """

        if size > 1:
            condensed_vector = "[ "
            count = 0
            for var in vector:
                count += 1
                condensed_vector += f"{float(var)} "
                if count == 5:
                    condensed_vector += f"... <br/>... "
                    count = 0
            condensed_vector += "]<sup>T</sup>"
        else:
            condensed_vector = f"{float(vector[0])}"
        return condensed_vector

    def _draw_parameter_node(self, g: Digraph.subgraph, phase_idx: int, param_idx: int, parameter: Parameter):
        initial_guess, min_bound, max_bound = self._scaling_parameter(parameter)
        node_str = f"<u><b>{parameter.name[0].upper() + parameter.name[1:]}</b></u><br/>"
        node_str += f"<b>Size</b>: {parameter.size}<br/>"
        node_str += f"<b>Initial guesses</b>: {self._vector_layout(initial_guess, parameter.size)}<br/><br/>"
        node_str += f"{self._vector_layout(min_bound, parameter.size)} ≤<br/>"
        node_str += f"{'(' if parameter.penalty_list is not None and parameter.penalty_list.quadratic else ''}{self._get_parameter_function_name(parameter)} -<br/>"
        node_str += f"{parameter.penalty_list.sliced_target if parameter.penalty_list is not None else ''}{f'){self._squared}' if parameter.penalty_list is not None and parameter.penalty_list.quadratic else ''} ≤<br/>"
        node_str += f"{self._vector_layout(max_bound, parameter.size)}"
        g.node(f"param_{phase_idx}{param_idx}", f"""<{node_str}>""")

    def _draw_nlp_node(self, g: Digraph.subgraph, phase_idx: int):
        node_str = f"<b>Model</b>: {self.ocp.nlp[phase_idx].model.path().filename().to_string()}.{self.ocp.nlp[phase_idx].model.path().extension().to_string()}<br/>"
        node_str += f"<b>Phase duration</b>: {round(self.ocp.nlp[phase_idx].t_initial_guess, 2)} s<br/>"
        node_str += f"<b>Shooting nodes</b>: {self.ocp.nlp[phase_idx].ns}<br/>"
        node_str += f"<b>Dynamics</b>: {self.ocp.nlp[phase_idx].dynamics_type.type.name}<br/>"
        node_str += f"<b>ODE</b>: {self.ocp.nlp[phase_idx].ode_solver.rk_integrator.__name__}"
        g.node(f"nlp_node_{phase_idx}", f"""<{node_str}>""")

    def _draw_lagrange_node(self, g: Digraph.subgraph, phase_idx: int):
        lagrange_str = self._lagrange_to_str(self.ocp.nlp[phase_idx].J)[0]
        node_str = f"<b>Lagrange</b><br/>{lagrange_str}"
        g.node(f"lagrange_{phase_idx}", f"""<{node_str}>""")

    def _draw_mayer_node(self, g: Digraph.subgraph, phase_idx: int):
        list_mayer_objectives = self._mayer_to_str(self.ocp.nlp[phase_idx].J)
        all_mayer_str = "<b>Mayer</b><br/>"
        if len(list_mayer_objectives) != 0:
            for objective in list_mayer_objectives:
                all_mayer_str += objective[1]
                all_mayer_str += f"Shooting nodes index: {objective[0]}<br/><br/>"
        else:
            all_mayer_str = "<b>Mayer</b><br/> No Mayer set"
        g.node(f"mayer_node_{phase_idx}", f"""<{all_mayer_str}>""")

    def _draw_constraints_node(self, g: Digraph.subgraph, phase_idx: int):
        list_constraints = []

        for constraint in self.ocp.nlp[phase_idx].g:
            constraints_str = ""
            if constraint[0]["node_index"] == -1:
                node_index = self._analyze_nodes(phase_idx, constraint[0])
            else:
                node_index = constraint[0]["node_index"]
            constraints_str += self._constraint_to_str(constraint[0]["constraint"])
            list_constraints.append([constraints_str, node_index])

        all_constraints_str = "<b>Constraints</b><br/>"
        if len(list_constraints) != 0:
            for constraint in list_constraints:
                all_constraints_str += constraint[0]
                all_constraints_str += f"Shooting nodes index: {constraint[1]}<br/><br/>"
        else:
            all_constraints_str = "<b>Constraints</b><br/> No constraint set"
        g.node(f"constraints_node_{phase_idx}", f"""<{all_constraints_str}>""")

    def _draw_nlp_cluster(self, G: Digraph, phase_idx: int):
        """
        Draw clusters for each nlp

        Parameters
        ----------
        phase_idx: int
            The index of the current phase
        G: Digraph
            The graph to be completed
        """

        with G.subgraph(name=f"cluster_{phase_idx}") as g:
            g.attr(style="filled", color="lightgrey")
            nlp_title = f"<u><b>Phase #{phase_idx}</b></u>"
            g.attr(label=f"<{nlp_title}>")
            g.node_attr.update(style="filled", color="white")

            self._draw_nlp_node(g, phase_idx)

            if len(self.ocp.nlp[phase_idx].parameters) > 0:
                param_idx = 0
                for param in self.ocp.nlp[phase_idx].parameters:
                    self._draw_parameter_node(g, phase_idx, param_idx, param)
                    param_idx += 1
            else:
                node_str = "<b>Parameters</b><br/> No parameter set"
                g.node(f"param_{phase_idx}0", f"""<{node_str}>""")

            if len(self.ocp.nlp[phase_idx].J) > 0:
                self._draw_lagrange_node(g, phase_idx)
            else:
                node_str = "<b>Lagrange</b><br/> No Lagrange set"
                g.node(f"lagrange_{phase_idx}", f"""<{node_str}>""")

            self._draw_mayer_node(g, phase_idx)
            self._draw_constraints_node(g, phase_idx)

    # Draw edge between Lagrange node and Mayer node
    def _draw_lagrange_to_mayer_edge(self, G: Digraph, phase_idx: int):
        G.edge(f"lagrange_{phase_idx}", f"mayer_node_{phase_idx}", color="lightgrey")

    # Draw edge between Mayer node and constraints node
    def _draw_mayer_to_constraints_edge(self, G: Digraph, phase_idx: int):
        G.edge(f"mayer_node_{phase_idx}", f"constraints_node_{phase_idx}", color="lightgrey")

    # Draw edges between nlp node and parameters
    def _draw_nlp_to_parameters_edges(self, G: Digraph, phase_idx: int):
        nb_parameters = len(self.ocp.nlp[phase_idx].parameters)
        G.edge(f"nlp_node_{phase_idx}", f"param_{phase_idx}0", color="lightgrey")
        for param_idx in range(nb_parameters):
            if param_idx >= 1:
                G.edge(f"param_{phase_idx}{param_idx - 1}", f"param_{phase_idx}{param_idx}", color="lightgrey")
        if nb_parameters > 1:
            G.edge(f"param_{phase_idx}{nb_parameters - 1}", f"lagrange_{phase_idx}", color="lightgrey")
        else:
            G.edge(f"param_{phase_idx}0", f"lagrange_{phase_idx}", color="lightgrey")

    def _draw_edges(self, G: Digraph, phase_idx: int):
        """
        Draw edges between each node of a cluster

        Parameters
        ----------
        phase_idx: int
            The index of the current phase
        G: Digraph
            The graph to be completed
        """

        # Draw edges between OCP node and each nlp cluster
        G.edge("OCP", f"nlp_node_{phase_idx}")

        self._draw_nlp_to_parameters_edges(G, phase_idx)
        self._draw_lagrange_to_mayer_edge(G, phase_idx)
        self._draw_mayer_to_constraints_edge(G, phase_idx)

    # Display phase transitions
    def _draw_phase_transitions(self, G: Digraph):
        """
        Draw a cluster including all the information about the phase transitions of the problem

        Parameters
        ----------
        G: Digraph
            The graph to be completed
        """

        with G.subgraph(name=f"cluster_phase_transitions") as g:
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
