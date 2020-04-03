from casadi import MX, vertcat


class Variable:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu)
        :param nlp: An OptimalControlProgram class.
        """
        dof_names = nlp.model.nameDof()
        q = MX()
        q_dot = MX()
        for i in nlp.dof_mapping.reduce_idx:
            q = vertcat(q, MX.sym("Q_" + dof_names[i].to_string()))
        for i in nlp.dof_mapping.reduce_idx:
            q_dot = vertcat(q_dot, MX.sym("Qdot_" + dof_names[i].to_string()))
        nlp.x = vertcat(q, q_dot)

        for i in nlp.dof_mapping.reduce_idx:
            nlp.u = vertcat(nlp.u, MX.sym("Tau_" + dof_names[i].to_string()))

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()

    @staticmethod
    def muscles_and_torque_driven(nlp):
        dof_names = nlp.model.nameDof()
        muscle_names = nlp.model.muscleNames()
        q = MX()
        q_dot = MX()
        for i in range(nlp.model.nbQ()):
            q = vertcat(q, MX.sym("Q_" + dof_names[i].to_string()))
        for i in range(nlp.model.nbQdot()):
            q_dot = vertcat(q_dot, MX.sym("Qdot_" + dof_names[i].to_string()))
        nlp.x = vertcat(q, q_dot)

        for i in range(nlp.model.nbMuscleTotal()):
            nlp.u = vertcat(
                nlp.u, MX.sym("Tau_for_muscle_" + muscle_names[i].to_string())
            )
        for i in range(nlp.model.nbGeneralizedTorque()):
            nlp.u = vertcat(nlp.u, MX.sym("Tau_" + dof_names[i].to_string()))

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()
