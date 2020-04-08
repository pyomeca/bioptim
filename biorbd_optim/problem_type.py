from casadi import MX, vertcat

from .dynamics import Dynamics
from .mapping import Mapping


class ProblemType:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu)
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_driven

        if nlp["dof_mapping"] is None:
            nlp["dof_mapping"] = Mapping(range(nlp.model.nbQ()), range(nlp.model.nbQ()))

        dof_names = nlp["model"].nameDof()
        q = MX()
        q_dot = MX()
        for i in nlp["dof_mapping"].reduce_idx:
            q = vertcat(q, MX.sym("Q_" + dof_names[i].to_string()))
        for i in nlp["dof_mapping"].reduce_idx:
            q_dot = vertcat(q_dot, MX.sym("Qdot_" + dof_names[i].to_string()))
        nlp["x"] = vertcat(q, q_dot)

        u = MX()
        for i in nlp["dof_mapping"].reduce_idx:
            u = vertcat(u, MX.sym("Tau_" + dof_names[i].to_string()))
        nlp["u"] = u

        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

        nlp["nbQ"] = nlp["dof_mapping"].nb_reduced
        nlp["nbQdot"] = nlp["dof_mapping"].nb_reduced
        nlp["nbQtau"] = nlp["dof_mapping"].nb_reduced
        nlp["nbMuscle"] = 0

    @staticmethod
    def muscles_and_torque_driven(nlp):
        ProblemType.torque_driven(nlp)
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_muscle_driven

        u = MX()
        muscle_names = nlp.model.muscleNames()
        for i in range(nlp.model.nbMuscleTotal()):
            u = vertcat(
                u, MX.sym("Tau_for_muscle_" + muscle_names[i].to_string())
            )
        nlp["u"] = vertcat(u, nlp["u"])

        nlp["nbMuscle"] = nlp.model.nbMuscleTotal()
