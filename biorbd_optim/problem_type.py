from casadi import MX, vertcat
import numpy as np

from .dynamics import Dynamics
from .mapping import BidirectionalMapping, Mapping


class ProblemType:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques but without muscles, must be used with dynamics without contacts.
        :param nlp: An instance of the OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_driven
        ProblemType.__configure_torque_driven(nlp)

    @staticmethod
    def torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques, without muscles, must be used with dynamics with contacts.
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_driven_with_contact
        ProblemType.__configure_torque_driven(nlp)

    @staticmethod
    def __configure_torque_driven(nlp):
        """
        Configures common settings for torque driven problems with and without contacts.
        :param nlp: An OptimalControlProgram class.
        """
        if nlp["q_mapping"] is None:
            nlp["q_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbQ())), Mapping(range(nlp["model"].nbQ()))
            )
        if nlp["q_dot_mapping"] is None:
            nlp["q_dot_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbQdot())), Mapping(range(nlp["model"].nbQdot()))
            )
        if nlp["tau_mapping"] is None:
            nlp["tau_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbGeneralizedTorque())), Mapping(range(nlp["model"].nbGeneralizedTorque()))
            )

        dof_names = nlp["model"].nameDof()
        q = MX()
        q_dot = MX()
        for i in nlp["q_mapping"].reduce.map_idx:
            q = vertcat(q, MX.sym("Q_" + dof_names[i].to_string()))
        for i in nlp["q_dot_mapping"].reduce.map_idx:
            q_dot = vertcat(q_dot, MX.sym("Qdot_" + dof_names[i].to_string()))
        nlp["x"] = vertcat(q, q_dot)

        u = MX()
        for i in nlp["tau_mapping"].reduce.map_idx:
            u = vertcat(u, MX.sym("Tau_" + dof_names[i].to_string()))
        nlp["u"] = u

        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

        nlp["nbQ"] = nlp["q_mapping"].reduce.len
        nlp["nbQdot"] = nlp["q_dot_mapping"].reduce.len
        nlp["nbTau"] = nlp["tau_mapping"].reduce.len
        nlp["nbMuscle"] = 0

    @staticmethod
    def muscle_activations_and_torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_muscle_driven
        ProblemType.__configure_torque_driven(nlp)

        nlp["nbMuscle"] = nlp["model"].nbMuscleTotal()

        u = MX()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["nu"] = nlp["u"].rows()

    @staticmethod
    def muscle_excitations_and_torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_muscle_excitations_and_torque_driven
        ProblemType.__configure_torque_driven(nlp)

        nlp["nbMuscle"] = nlp["model"].nbMuscleTotal()

        u = MX()
        x = MX()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_excitation"))
            x = vertcat(x, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["x"] = vertcat(nlp["x"], x)

        nlp["nu"] = nlp["u"].rows()
        nlp["nx"] = nlp["x"].rows()

    @staticmethod
    def muscles_and_torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_muscle_driven_with_contact
        ProblemType.__configure_torque_driven(nlp)

        u = MX()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["model"].nbMuscleTotal()):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)

        nlp["nu"] = nlp["u"].rows()

        nlp["nbMuscle"] = nlp["model"].nbMuscleTotal()

    @staticmethod
    def get_data_from_V_phase(V_phase, var_size, nb_nodes, offset, nb_variables, duplicate_last_column):
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
        has_muscles = False

        if num_phase is None:
            num_phase = range(len(ocp.nlp))
        elif isinstance(num_phase, int):
            num_phase = [num_phase]
        offsets = [0]
        for i, nlp in enumerate(ocp.nlp):
            offsets.append(offsets[i] + nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * (nlp["ns"]))

        q, q_dot, tau, muscle = [], [], [], []

        for i in num_phase:
            nlp = ocp.nlp[i]

            V_phase = np.array(V_array[offsets[i] : offsets[i + 1]])
            nb_var = nlp["nx"] + nlp["nu"]

            if (
                nlp["problem_type"] == ProblemType.torque_driven
                or nlp["problem_type"] == ProblemType.torque_driven_with_contact
                or nlp["problem_type"] == ProblemType.muscle_activations_and_torque_driven
                or nlp["problem_type"] == ProblemType.muscles_and_torque_driven_with_contact
            ):
                q.append(ProblemType.get_data_from_V_phase(V_phase, nlp["nbQ"], nlp["ns"] + 1, 0, nb_var, False))
                q_dot.append(
                    ProblemType.get_data_from_V_phase(V_phase, nlp["nbQdot"], nlp["ns"] + 1, nlp["nbQ"], nb_var, False)
                )
                tau.append(ProblemType.get_data_from_V_phase(V_phase, nlp["nbTau"], nlp["ns"], nlp["nx"], nb_var, True))

                if (
                    nlp["problem_type"] == ProblemType.muscle_activations_and_torque_driven
                    or nlp["problem_type"] == ProblemType.muscles_and_torque_driven_with_contact
                ):
                    has_muscles = True
                    muscle.append(
                        ProblemType.get_data_from_V_phase(
                            V_phase, nlp["nbMuscle"], nlp["ns"], nlp["nx"] + nlp["nbTau"], nb_var, True,
                        )
                    )
                else:
                    muscle.append([])

            else:
                raise RuntimeError(f"{nlp['problem_type'].__name__} not implemented yet in get_data_from_V")

        if len(num_phase) == 1:
            q = q[0]
            q_dot = q_dot[0]
            tau = tau[0]
            muscle = muscle[0]
        if has_muscles:
            return q, q_dot, tau, muscle
        else:
            return q, q_dot, tau

    @staticmethod
    def get_states_integrated_from_V(ocp, V, number_elements=1, concatenate_phases=False):
        v = ProblemType.get_data_from_V(ocp, V)[:3]

        if ocp.nb_phases == 1:
            v = [v]

        t = [np.linspace(0, ocp.nlp[i]["tf"], ocp.nlp[i]["ns"] * number_elements) for i in range(ocp.nb_phases)]
        integrated_state = []
        for idx_phase, v_phase in enumerate(v):

            integrated_state_phase = np.ndarray(
                (ocp.nlp[idx_phase]["nbQ"] + ocp.nlp[idx_phase]["nbQdot"], ocp.nlp[idx_phase]["ns"] + 1)
            )
            for j in range(ocp.nlp[idx_phase]["ns"] + 1):
                integrated_state_phase[:, j] = np.reshape(
                    ocp.nlp[idx_phase]["dynamics"].call(
                        {"x0": np.concatenate((v_phase[:2][0][:, j], v_phase[:2][1][:, j])), "p": v_phase[2][:, j]}
                    )["xf"],
                    ocp.nlp[idx_phase]["nbQ"] + ocp.nlp[idx_phase]["nbQdot"]
                )

            integrated_state.append(integrated_state_phase)

        if concatenate_phases:
            same_dof = True
            for i in range(self.ocp.nb_phases):
                for k in range(self.ocp.nlp[0]["model"].nbDof()):
                    if (
                            self.ocp.nlp[i]["model"].nameDof()[k].to_string()
                            != self.ocp.nlp[i - 1]["model"].nameDof()[k].to_string()
                    ):
                        same_dof = False
            if same_dof:
                t_concat = t[0]
                state_concat = integrated_state[0]
                for i in range(1, self.ocp.nb_phases):
                    state_concat = np.concatenate((state_concat, integrated_state[i][:, 1:]), axis=1)
                    t_concat = np.concatenate((t_concat, t[0][i][1:] + t[-1]))
        return integrated_state

    @staticmethod
    def __interpolate(self, idx_phase, x_phase, t, nb_frames):
        x_interpolate = np.ndarray((self.ocp.nlp[idx_phase]["nbQ"], nb_frames))
        for j in range(self.ocp.nlp[idx_phase]["nbQ"]):
            x_interpolate[j] = interpolate.splev(
                np.linspace(0, t[idx_phase][-1], nb_frames), interpolate.splrep(t[idx_phase], x_phase[j], s=0)
            )
        return x_interpolate

    @staticmethod
    def get_q_from_V(ocp, V, num_phase=None):
        if ocp.nlp[0]["problem_type"] == ProblemType.torque_driven:
            x, _, _ = ProblemType.get_data_from_V(ocp, V, num_phase)

        elif (
                ocp.nlp[0]["problem_type"] == ProblemType.muscle_activations_and_torque_driven
                or ocp.nlp[0]["problem_type"] == ProblemType.muscles_and_torque_driven_with_contact
        ):
            x, _, _, _ = ProblemType.get_data_from_V(ocp, V, num_phase)

        else:
            raise RuntimeError(f"{ocp.nlp[0]['problem_type']} is not implemented for this type of OCP")
        return x
