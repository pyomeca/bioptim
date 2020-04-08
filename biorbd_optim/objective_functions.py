import casadi


class ObjectiveFunction:
    @staticmethod
    def minimize_torque(ocp, nlp, weight=1):
        n_tau = nlp["nbTau"]

        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(
                    nlp["U"][i][: n_tau], nlp["U"][i][: n_tau],
                )
                * nlp["dt"]
                * nlp["dt"]
                * weight
            )

    @staticmethod
    def minimize_states(ocp, nlp, weight=1):
        for i in range(nlp["ns"]):
            ocp.J += casadi.dot(nlp["X"][i], nlp["X"][i]) * nlp["dt"] * nlp["dt"] * weight

    @staticmethod
    def minimize_muscle(ocp, nlp, weight=1):
        n_tau = nlp["nbTau"]
        nb_muscle = nlp["nbMuscle"]
        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(
                    nlp["U"][i][n_tau : n_tau],
                    nlp["U"][i][n_tau : n_tau + nb_muscle],
                )
                * nlp["dt"]
                * nlp["dt"]
                * weight
            )

    @staticmethod
    def minimize_all_controls(nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")

    @staticmethod
    def cyclic(nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")
