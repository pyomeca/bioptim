from graphviz import Digraph

from ..limits.constraints import Constraint
from ..limits.objective_functions import ObjectiveFcn, ObjectiveList
from ..optimization.parameters import Parameter

class OCPToConsole:

    def __init__(
        self,
        ocp,
    ):

        self.ocp = ocp

    def print_to_console(self):
        for phase_idx in range(self.ocp.n_phases):
            print(f"PHASE {phase_idx}")
            print(f"**********")
            print(f"PARAMETERS: ")
            print("")
            for parameter in self.ocp.nlp[phase_idx].parameters:
                initial_guess, min_bound, max_bound = self.scaling_parameter(parameter)
                objective_name = self.get_parameter_function_name(parameter)
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

            mayer_objectives = self.mayer_to_str(self.ocp.nlp[phase_idx].J)
            lagrange_objectives = self.lagrange_to_str(self.ocp.nlp[phase_idx].J)[1]
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
                for i in range(self.ocp.nlp[phase_idx].g.__len__()):
                    if self.ocp.nlp[phase_idx].g[i][0]['node_index'] == node_idx:
                        print(f"*** Constraint {i}: {self.ocp.nlp[phase_idx].g[phase_idx][i]['constraint'].name}")
                print("")

    @staticmethod
    def constraint_to_str(constraint: Constraint):

        def add_param_constraint_to_str(param_dict: dict):
            str_to_add = ""
            for param in param_dict:
                str_to_add += f"{param}: {param_dict[param]}<br/>"
            return str_to_add

        constraint_str = ""
        target_str = "" if constraint.sliced_target is None else \
            f"{constraint.sliced_target}"
        if constraint.quadratic:
            constraint_str += f"{constraint.min_bound} ≤ "
            constraint_str += f"({constraint.name}" if target_str is not "" else f"{constraint.name}"
            constraint_str += f" - {target_str})<sup>2</sup>" if target_str is not "" else ""
            constraint_str += f" ≤ {constraint.max_bound}<br/>"
        else:
            constraint_str += f"{constraint.min_bound} ≤ {constraint.name}"
            constraint_str += f" - {target_str}" if target_str is not "" else ""
            constraint_str += f" ≤ {constraint.max_bound}<br/>"
        constraint_str += add_param_constraint_to_str(constraint.params)
        return constraint_str

    @staticmethod
    def add_extra_parameters_to_str(list_constraints: list, string: str):
        for param in list_constraints:
            string += f"{param}: {list_constraints[param]}<br/>"
        string += f"<br/>"
        return string

    @staticmethod
    def lagrange_to_str(objective_list: ObjectiveList):
        objective_names = []
        lagrange_str = ""
        for objective in objective_list:
            if len(objective)>0:
                obj = objective[0]['objective']
                if isinstance(obj.type, ObjectiveFcn.Lagrange):
                    if obj.sliced_target is not None:
                        if obj.quadratic:
                            lagrange_str += f"({obj.name} - {obj.sliced_target})<sup>2</sup><br/>"
                        else:
                            lagrange_str += f"{obj.name} - {obj.sliced_target}<br/>"
                    else:
                        if obj.quadratic:
                            lagrange_str += f"({obj.name})<sup>2</sup><br/>"
                        else:
                            lagrange_str += f"{obj.name}<br/>"
                    lagrange_str = OCPToConsole.add_extra_parameters_to_str(obj.params, lagrange_str)
                    objective_names.append(obj.name)
        return lagrange_str, objective_names

    @staticmethod
    def mayer_to_str(objective_list: ObjectiveList):
        list_mayer_objectives = []
        for objective in objective_list:
            for obj_index in objective:
                obj = obj_index['objective']
                if isinstance(obj.type, ObjectiveFcn.Mayer):
                    mayer_str = ""
                    mayer_objective = [obj.node[0]]
                    if obj.sliced_target is not None:
                        if obj.quadratic:
                            mayer_str += f"({obj.name} - {obj.sliced_target})<sup>2</sup>"
                        else:
                            mayer_str += f"{obj.name} - {obj.sliced_target}"
                    else:
                        if obj.quadratic:
                            mayer_str += f"({obj.name})<sup>2</sup>"
                        else:
                            mayer_str += f"{obj.name}"
                    mayer_str = OCPToConsole.add_extra_parameters_to_str(obj.params, mayer_str)
                    mayer_objective.append(mayer_str)
                    list_mayer_objectives.append(mayer_objective)
        return list_mayer_objectives

    @staticmethod
    def vector_layout(vector: list, size: int):
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

    @staticmethod
    def scaling_parameter(parameter: Parameter):
        initial_guess = [parameter.initial_guess.init[i][j] * parameter.scaling[i] for i in
                         range(parameter.size) for j in
                         range(len(parameter.initial_guess.init[0]))]
        min_bound = [parameter.bounds.min[i][j] * parameter.scaling[i] for i in
                     range(parameter.size) for j in
                     range(len(parameter.bounds.min[0]))]
        max_bound = [parameter.bounds.max[i][j] * parameter.scaling[i] for i in
                     range(parameter.size) for j in
                     range(len(parameter.bounds.max[0]))]
        return initial_guess, min_bound, max_bound

    @staticmethod
    def get_parameter_function_name(parameter: Parameter):
        name = ""
        if parameter.penalty_list is not None:
            if parameter.penalty_list.type.name == "CUSTOM":
                name = parameter.penalty_list.custom_function.__name__
            else:
                name = parameter.penalty_list.name
        return name

class OCPToGraph:

    def __init__(
            self,
            ocp,
    ):

        self.ocp = ocp

    def print_to_graph(self):
        # Initialize graph with graphviv
        G = Digraph('ocp_graph', node_attr={'shape': 'plaintext'})

        # Draw OCP node
        G.node('OCP', shape='Mdiamond')

        # Draw nlp clusters and edges
        for phase_idx in range(self.ocp.n_phases):
            self.draw_nlp_cluster(phase_idx, G)
            self.draw_edges(phase_idx, G)

        # Draw phase_transitions
        self.display_phase_transitions(G)

        # Display graph
        G.view()

    # Draw nlp clusters composed of different nodes
    def draw_nlp_cluster(self, phase_idx: int, G: Digraph):

        def draw_parameter_node(param_idx: int, parameter: Parameter):
            initial_guess, min_bound, max_bound = OCPToConsole.scaling_parameter(parameter)
            node_str = f"<u><b>{parameter.name}</b></u><br/>"
            node_str += f"<b>Size</b>: {parameter.size}<br/>"
            node_str += f"<b>Initial guesses</b>: {OCPToConsole.vector_layout(initial_guess, parameter.size)}<br/><br/>"
            node_str += f"{OCPToConsole.vector_layout(min_bound, parameter.size)} ≤<br/>"
            node_str += f"{'(' if parameter.penalty_list is not None and parameter.penalty_list.quadratic else ''}{OCPToConsole.get_parameter_function_name(parameter)} -<br/>"
            node_str += f"{parameter.penalty_list.sliced_target if parameter.penalty_list is not None else ''}{')<sup>2</sup>' if parameter.penalty_list is not None and parameter.penalty_list.quadratic else ''} ≤<br/>"
            node_str += f"{OCPToConsole.vector_layout(max_bound, parameter.size)}"
            g.node(f"param_{phase_idx}{param_idx}", f'''<{node_str}>''')

        def draw_nlp_node():
            node_str = f"<b>Model</b>: {self.ocp.nlp[phase_idx].model.path().filename().to_string()}.{self.ocp.nlp[phase_idx].model.path().extension().to_string()}<br/>"
            node_str += f"<b>Phase duration</b>: {round(self.ocp.nlp[phase_idx].t_initial_guess, 2)} s<br/>"
            node_str += f"<b>Shooting nodes</b>: {self.ocp.nlp[phase_idx].ns}<br/>"
            node_str += f"<b>Dynamics</b>: {self.ocp.nlp[phase_idx].dynamics_type.type.name}<br/>"
            node_str += f"<b>ODE</b>: {self.ocp.nlp[phase_idx].ode_solver.rk_integrator.__name__}"
            g.node(f'nlp_node_{phase_idx}', f'''<{node_str}>''')

        def draw_lagrange_node():
            lagrange_str = OCPToConsole.lagrange_to_str(self.ocp.nlp[phase_idx].J)[0]
            node_str = f"<b>Lagrange</b>:<br/>{lagrange_str}"
            g.node(f'lagrange_{phase_idx}', f'''<{node_str}>''')

        def draw_mayer_node():

            list_mayer_objectives = OCPToConsole.mayer_to_str(self.ocp.nlp[phase_idx].J)
            all_mayer_str = "<b>Mayer:</b><br/>"
            if len(list_mayer_objectives) != 0:
                for objective in list_mayer_objectives:
                    all_mayer_str += objective[1]
                    all_mayer_str += f"Shooting nodes index: {objective[0]}<br/><br/>"
            else:
                all_mayer_str = "No Mayer set"
            g.node(f'mayer_node_{phase_idx}', f'''<{all_mayer_str}>''')

        def draw_constraints_node():
            list_constraints = []
            for node_idx in range(self.ocp.nlp[phase_idx].ns + 1):
                constraints_str = ""
                for constraint in self.ocp.nlp[phase_idx].g:
                    nb_constraint_nodes = len(constraint)
                    for i in range(nb_constraint_nodes):
                        if constraint[i]['node_index'] == node_idx:
                            constraints_str += OCPToConsole.constraint_to_str(constraint[0]['constraint'])

                if constraints_str != "":
                    found = False
                    for constraint in list_constraints:
                        if constraints_str == constraint[0]:
                            found = True
                            constraint[1].append(node_idx)
                    if not found:
                        list_constraints.append([constraints_str, [node_idx]])

            all_constraints_str = "<b>Constraints:</b><br/>"
            if len(list_constraints) != 0:
                for constraint in list_constraints:
                    all_constraints_str += constraint[0]
                    if len(constraint[1]) == self.ocp.nlp[phase_idx].ns:
                        constraint[1] = 'all'
                    all_constraints_str += f"Shooting nodes index: {constraint[1]}<br/><br/>"
            else:
                all_constraints_str = "No constraint set"
            g.node(f'constraints_node_{phase_idx}', f'''<{all_constraints_str}>''')

        with G.subgraph(name=f'cluster_{phase_idx}') as g:
            g.attr(style='filled', color='lightgrey')
            g.attr(label=f"Phase #{phase_idx}")
            g.node_attr.update(style='filled', color='white')

            draw_nlp_node()

            if len(self.ocp.nlp[phase_idx].parameters) > 0:
                param_idx = 0
                for param in self.ocp.nlp[phase_idx].parameters:
                    draw_parameter_node(param_idx, param)
                    param_idx += 1
            else:
                g.node(name=f'param_{phase_idx}0', label=f"No parameter set")

            if len(self.ocp.nlp[phase_idx].J) > 0:
                draw_lagrange_node()
            else:
                g.node(name=f'lagrange_{phase_idx}', label=f"No Lagrange set")

            draw_mayer_node()
            draw_constraints_node()

    def draw_edges(self, phase_idx: int, G:Digraph):

        # Draw edge between Lagrange node and Mayer node
        def draw_lagrange_to_mayer_edge():
            G.edge(f'lagrange_{phase_idx}',
                   f'mayer_node_{phase_idx}',
                   color='black')

        # Draw edge between Mayer node and constraints node
        def draw_mayer_to_constraints_edge():
            G.edge(f'mayer_node_{phase_idx}',
                   f'constraints_node_{phase_idx}',
                   color='black')

        # Draw edges between nlp node and parameters
        def draw_nlp_to_parameters_edges():
            nb_parameters = len(self.ocp.nlp[phase_idx].parameters)
            G.edge(f'nlp_node_{phase_idx}',
                   f'param_{phase_idx}0',
                   color='black')
            for param_idx in range(nb_parameters):
                if param_idx >= 1:
                    G.edge(f'param_{phase_idx}{param_idx - 1}',
                           f'param_{phase_idx}{param_idx}',
                           color='black')
            if nb_parameters > 1:
                G.edge(f'param_{phase_idx}{nb_parameters-1}',
                       f'lagrange_{phase_idx}',
                       color='black')
            else:
                G.edge(f'param_{phase_idx}0',
                       f'lagrange_{phase_idx}',
                       color='black')

        # Draw edges between OCP node and each nlp cluster
        G.edge('OCP', f'nlp_node_{phase_idx}')

        draw_nlp_to_parameters_edges()
        draw_lagrange_to_mayer_edge()
        draw_mayer_to_constraints_edge()

    # Display phase transitions
    def display_phase_transitions(self, G:Digraph):
        with G.subgraph(name=f'cluster_phase_transitions') as g:
            g.attr(style='', color='black')
            g.node_attr.update(style='filled', color='grey')
            for phase_idx in range(self.ocp.n_phases):
                if phase_idx != self.ocp.n_phases - 1:
                    g.node(f'Phase #{phase_idx}')
                    g.node(f'Phase #{phase_idx + 1}')
                    g.edge(f'Phase #{phase_idx}',
                           f'Phase #{phase_idx + 1}',
                           label=self.ocp.phase_transitions[phase_idx].type.name)
            g.attr(label=f"Phase transitions")