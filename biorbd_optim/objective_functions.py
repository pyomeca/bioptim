import casadi


class ObjectiveFunction:
    @staticmethod
    def minimize_torque(nlp, weight=1):
        n_tau = nlp.dof_mapping.nb_reduced
        n_mus = nlp.model.nbMuscleTotal()
        for i in range(nlp.ns):
            nlp.J += (
                casadi.dot(
                    nlp.U[i][n_mus : n_mus + n_tau], nlp.U[i][n_mus : n_mus + n_tau],
                )
                * nlp.dt
                * nlp.dt
                * weight
            )

    @staticmethod
    def minimize_states(nlp, weight=1):
        for i in range(nlp.ns):
            nlp.J += casadi.dot(nlp.X[i], nlp.X[i]) * nlp.dt * nlp.dt * weight

    @staticmethod
    def minimize_muscle(nlp, weight=1):
        for i in range(nlp.ns):
            nlp.J += (
                casadi.dot(
                    nlp.U[i][: nlp.model.nbMuscleTotal()],
                    nlp.U[i][: nlp.model.nbMuscleTotal()],
                )
                * nlp.dt
                * nlp.dt
                * weight
            )

    @staticmethod
    def minimize_all_controls(nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")

    @staticmethod
    def cyclic(nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")
