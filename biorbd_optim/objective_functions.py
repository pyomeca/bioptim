import casadi


class ObjectiveFunction:
    @staticmethod
    def minimize_torque(ocp, nlp, weight=1):
        n_tau = nlp["nbTau"]

        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(nlp["U"][i][:n_tau], nlp["U"][i][:n_tau],)
                * nlp["dt"]
                * nlp["dt"]
                * weight
            )

    @staticmethod
    def minimize_states(ocp, nlp, weight=1):
        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(nlp["X"][i], nlp["X"][i]) * nlp["dt"] * nlp["dt"] * weight
            )

    @staticmethod
    def minimize_muscle(ocp, nlp, weight=1):
        n_tau = nlp["nbTau"]
        nb_muscle = nlp["nbMuscle"]
        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(
                    nlp["U"][i][n_tau : n_tau + nb_muscle],
                    nlp["U"][i][n_tau : n_tau + nb_muscle],
                )
                * nlp["dt"]
                * nlp["dt"]
                * weight
            )

    @staticmethod
    def minimize_all_controls(ocp, nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")

    @staticmethod
    def cyclic(ocp, nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")

    @staticmethod
    def minimize_final_distance_between_two_markers(ocp, nlp, weight=1, markers=()):
        if not isinstance(markers, (list, tuple)) or len(markers) != 2:
            raise RuntimeError("minimize_distance_between_two_markers expect markers to be a list of 2 marker indices")
        q = nlp["dof_mapping"].expand(nlp["X"][nlp["ns"]][:nlp["nbQ"]])
        marker0 = nlp["model"].marker(q, markers[0]).to_mx()
        marker1 = nlp["model"].marker(q, markers[1]).to_mx()

        ocp.J += (
            casadi.dot(marker0 - marker1, marker0 - marker1) * nlp["dt"] * nlp["dt"] * weight
        )