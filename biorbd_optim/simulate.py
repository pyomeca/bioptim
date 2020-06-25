import numpy as np


class Simulate:
    @staticmethod
    def _concat_variables(variables, offset_phases, idx_nodes):
        var = np.ndarray(0)
        for key in variables.keys():
            var = np.append(var, variables[key][:, offset_phases + idx_nodes])
        return var

    @staticmethod
    def from_sol(ocp, sol):
        v = np.array(sol["x"]).squeeze()

        offset = 0
        for nlp in ocp.nlp:
            # TODO adds StateTransitionFunctions between phases
            for idx_nodes in range(nlp["ns"]):
                v[offset + nlp["nx"] + nlp["nu"] : offset + 2 * nlp["nx"] + nlp["nu"]] = np.array(
                    nlp["dynamics"][idx_nodes](
                        x0=v[offset : offset + nlp["nx"]], p=v[offset + nlp["nx"] : offset + nlp["nx"] + nlp["nu"]]
                    )["xf"]
                ).squeeze()
                offset += nlp["nx"] + nlp["nu"]
        sol["x"] = v
        return sol

    @staticmethod
    def from_data(ocp, data):
        states = data[0]
        controls = data[1]
        v = np.ndarray(0)

        offset_phases = 0
        for idx_phases, nlp in enumerate(ocp.nlp):
            offset = 0
            v_phase = np.ndarray((nlp["ns"] + 1) * nlp["nx"] + nlp["ns"] * nlp["nu"])
            v_phase[offset : offset + nlp["nx"]] = Simulate._concat_variables(states, offset_phases, 0)
            for idx_nodes in range(nlp["ns"]):
                v_phase[offset + nlp["nx"] + nlp["nu"] : offset + 2 * nlp["nx"] + nlp["nu"]] = np.array(
                    nlp["dynamics"][idx_nodes](
                        x0=Simulate._concat_variables(states, offset_phases, idx_nodes),
                        p=Simulate._concat_variables(controls, offset_phases, idx_nodes),
                    )["xf"]
                ).squeeze()
                offset += nlp["nx"] + nlp["nu"]
            v = np.append(v, v_phase)
            offset_phases += nlp["ns"]
        return {"x": v}


