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
            nlp["dof_mapping"] = Mapping(
                range(nlp["model"].nbQ()), range(nlp["model"].nbQ())
            )

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
        nlp["nbTau"] = nlp["dof_mapping"].nb_reduced
        nlp["nbMuscle"] = 0

    @staticmethod
    def muscles_and_torque_driven(nlp):
        ProblemType.torque_driven(nlp)
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_muscle_driven

        u = MX()
        muscle_names = nlp.model.muscleNames()
        for i in range(nlp.model.nbMuscleTotal()):
            u = vertcat(u, MX.sym("Tau_for_muscle_" + muscle_names[i].to_string()))
        nlp["u"] = vertcat(nlp["u"], u)

        nlp["nbMuscle"] = nlp.model.nbMuscleTotal()

    @staticmethod
    def get_data_from_V(ocp, V, num_phase=None):
        if num_phase is None:
            num_phase = range(ocp.nlp)
        elif isinstance(num_phase, int):
            num_phase = [num_phase]

        for i in num_phase:
            nlp = ocp.nlp[i]
            if (
                nlp["problem_type"] == ProblemType.torque_driven
                or nlp["problem_type"] == ProblemType.muscles_and_torque_driven
            ):
                q.append(np.ndarray((self.ns, self.nbQ)))
                q_dot = np.ndarray((self.ns, self.nbQdot))
                tau = np.ndarray((self.ns, self.nbTau))
                for idx in range(self.nbQ):
                    q[:, idx] = np.array(V[idx :: self.nx + self.nu]).squeeze()
                    q_dot[:, idx] = np.array(
                        V[self.nbQ + idx :: self.nx + self.nu]
                    ).squeeze()
                    tau[: self.ns, idx] = np.array(
                        V[self.ns + idx :: self.nx + self.nu]
                    )
                tau[-1, :] = tau[-2, :]
                if self.problem_type == ProblemType.muscles_and_torque_driven:
                    muscle = np.ndarray((self.ns + self.ocp.nb_phases, self.nbMuscle))
                    for idx in range(self.nbMuscle):
                        muscle[: self.ns, :] = np.array(
                            V[self.ns + self.nbTau + idx :: self.nx + self.nu]
                        )
                    muscle[-1, :] = muscle[-2, :]
                    return q, q_dot, tau, muscle
                else:
                    return q, q_dot, tau
            else:
                raise RuntimeError(
                    "plot.__get_data not implemented yet for this problem_type"
                )
