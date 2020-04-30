import numpy as np

class Data:
    def __init__(self, type):
        self.phase = []
        self.type = type

    class Phase:
        def __init__(self, nodes, nlp):
            self.nlp = nlp
            self.node = []
            for i in range (len(nodes[0])):
                self.node.append(Data.Phase.Node(nodes[:, i]))

        class Node:
            def __init__(self, dofs):
                self.dof = []
                for i in range (len(dofs)):
                    self.dof.append(Data.Phase.Node.Dof(dofs[i]))

            class Dof:
                def __init__(self, x0):
                    self.x0 = x0


    def append_phase(self, phase, nlp):
        self.phase.append(Data.Phase(phase, nlp))

    def get_data(self, phases=(), nodes=(), dofs=(), integrated=False):
        if self.phase == []:
            raise RuntimeError("Data empty, please append phase before trying to get data.")

        phases = phases if isinstance(phases, (list, tuple)) else [phases]
        nodes = nodes if isinstance(nodes, (list, tuple)) else [nodes]
        dofs = dofs if isinstance(dofs, (list, tuple)) else [dofs]

        range_phases = range(len(self.phase)) if phases == () else phases
        range_dofs = range(self.phase[range_phases[0]].nlp[f"nb{self.type}"]) if dofs == () else dofs
        data = np.ndarray((len(range_dofs), 1))

        for idx_phase in range_phases:
            range_nodes = range(self.phase[idx_phase].nlp["ns"] + 1) if nodes == () else nodes
            for idx_node in range_nodes:
                if integrated:
                    node = np.ndarray((len(range_dofs), len(self.phase[idx_phase].node[idx_node].dof[0].xf)))
                else:
                    node = np.ndarray((len(range_dofs), 1))
                cmp = 0
                for idx_dof in range_dofs:
                    if integrated:
                        node[cmp] = self.phase[idx_phase].node[idx_node].dof[idx_dof].xf
                    else:
                        node[cmp] = self.phase[idx_phase].node[idx_node].dof[idx_dof].x0
                    cmp += 1
                data = np.concatenate((data, node), axis=1)
        data = np.delete(data, 0, axis=1)
        return data

    def give_integrated(self, xf, idx_phase, idx_node):
        dof = self.phase[idx_phase].node[idx_node].dof
        if len(dof) != len(xf):
            raise RuntimeError(f"xf length must be ({len(dof)}), not ({len(xf)}).")

        for idx_dof in range(len(xf)):
            if isinstance(xf[idx_dof], (tuple, list)):
                dof[idx_dof].xf = np.concatenate((np.array([dof[idx_dof].x0]), xf[idx_dof]))
            else:
                dof[idx_dof].xf = np.array([dof[idx_dof].x0, xf[idx_dof]])

    @staticmethod
    def concat_dof(first_data, second_data, phases=(), nodes=()):
        first_array = first_data.get_data(**{"phases": phases, "nodes": nodes})
        second_array = second_data.get_data(**{"phases": phases, "nodes": nodes})
        return np.concatenate((first_array, second_array))

    @staticmethod
    def get_phase(V_phase, var_size, nb_nodes, offset, nb_variables, duplicate_last_column):
        """
        Extracts variables from V.
        :param V_phase: numpy array : Extract of V for a phase.
        """
        array = np.ndarray((var_size, nb_nodes))
        for dof in range(var_size):
            array[dof] = V_phase[offset + dof :: nb_variables]

        if duplicate_last_column:
            return np.c_[array, array[:, -1]]
        else:
            return array

    @staticmethod
    def get_data_from_V(ocp, V, num_phase=None):
        V_array = np.array(V).squeeze()

        if num_phase is None:
            num_phase = range(len(ocp.nlp))
        elif isinstance(num_phase, int):
            num_phase = [num_phase]
        offsets = [0]
        for i, nlp in enumerate(ocp.nlp):
            offsets.append(offsets[i] + nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * (nlp["ns"]))

        data = {"q": Data("Q"), "q_dot": Data("Qdot"), "tau": Data("Tau"), "muscle": Data("Muscle")}

        for i in num_phase:
            nlp = ocp.nlp[i]
            V_phase = np.array(V_array[offsets[i] : offsets[i + 1]])
            nb_var = nlp["nx"] + nlp["nu"]

            data["q"].append_phase(Data.get_phase(V_phase, nlp["nbQ"], nlp["ns"] + 1, 0, nb_var, False), nlp)
            data["q_dot"].append_phase(Data.get_phase(V_phase, nlp["nbQdot"], nlp["ns"] + 1, nlp["nbQ"], nb_var, False), nlp)
            data["tau"].append_phase(Data.get_phase(V_phase, nlp["nbTau"], nlp["ns"], nlp["nx"], nb_var, True), nlp)
            if (nlp["has_muscles"]):
                data["muscle"].append_phase(Data.get_phase(V_phase, nlp["nbMuscle"], nlp["ns"], nlp["nx"] + nlp["nbTau"], nb_var, True,), nlp)

        return data

    @staticmethod
    def get_data_integrated_from_V(ocp, V):
        data = Data.get_data_from_V(ocp, V)

        for idx_phase in range(ocp.nb_phases):
            for idx_node in range(ocp.nlp[idx_phase]["ns"]):
                if ocp.nlp[idx_phase]["has_muscles"]:
                    p = Data.concat_dof(data["tau"], data["muscle"], idx_phase, idx_node)
                else:
                    p = data["q"].get_data(**{"phases": idx_phase, "nodes": idx_node})

                xf_dof = np.reshape(ocp.nlp[idx_phase]["dynamics"].call(
                        {"x0": Data.concat_dof(data["q"], data["q_dot"], idx_phase, idx_node), "p": p})["xf"],
                    ocp.nlp[idx_phase]["nx"],
                )

                data["q"].give_integrated(xf_dof[:ocp.nlp[idx_phase]["nbQ"]], idx_phase, idx_node)
                data["q_dot"].give_integrated(xf_dof[ocp.nlp[idx_phase]["nbQ"]:], idx_phase, idx_node)
            data["q"].give_integrated(xf_dof[:ocp.nlp[idx_phase]["nbQ"]], idx_phase, ocp.nlp[idx_phase]["ns"])
            data["q_dot"].give_integrated(xf_dof[ocp.nlp[idx_phase]["nbQ"]:], idx_phase, ocp.nlp[idx_phase]["ns"])
        data["q"].get_data(**{"integrated": True})
        return data
