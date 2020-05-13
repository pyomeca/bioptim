from casadi import MX, vertcat, Function

from .dynamics import Dynamics
from .mapping import BidirectionalMapping, Mapping
from .plot import CustomPlot


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
        ProblemType.__configure_torque_driven(nlp)

        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [Dynamics.forward_dynamics_torque_driven(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()  # .map(nlp["ns"], "thread", 2)

    @staticmethod
    def torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques, without muscles, must be used with dynamics with contacts.
        :param nlp: An OptimalControlProgram class.
        """

        ProblemType.__configure_torque_driven(nlp)

        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [Dynamics.forward_dynamics_torque_driven_with_contact(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()  # .map(nlp["ns"], "thread", 2)

        nlp["contact_forces_func"] = Function(
            "contact_forces_func",
            [symbolic_states, symbolic_controls],
            [Dynamics.forces_from_forward_dynamics_with_contact(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["contact_forces"],
        ).expand()
        ProblemType.__configure_contact(nlp)

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
            q = vertcat(q, MX.sym("Q_" + dof_names[i].to_string(), 1, 1))
        for i in nlp["q_dot_mapping"].reduce.map_idx:
            q_dot = vertcat(q_dot, MX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))
        nlp["x"] = vertcat(q, q_dot)

        u = MX()
        for i in nlp["tau_mapping"].reduce.map_idx:
            u = vertcat(u, MX.sym("Tau_" + dof_names[i].to_string(), 1, 1))
        nlp["u"] = u
        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

        nlp["nbQ"] = nlp["q_mapping"].reduce.len
        nlp["nbQdot"] = nlp["q_dot_mapping"].reduce.len
        nlp["nbTau"] = nlp["tau_mapping"].reduce.len

        nlp["var_states"] = {"q": nlp["q_mapping"].reduce.len, "q_dot": nlp["q_dot_mapping"].reduce.len}
        nlp["var_controls"] = {"tau": nlp["tau_mapping"].reduce.len}

    @staticmethod
    def __configure_contact(nlp):
        nlp["nbContact"] = nlp["model"].nbContacts()
        contact_names = [n.to_string() for n in nlp["model"].contactNames()]
        plot_mappings = nlp["plot_mappings"]["contact_forces"] if "contact_forces" in nlp["plot_mappings"] else None
        nlp["custom_plots"] = {"contact_forces": CustomPlot(nlp["nbContact"], nlp["contact_forces_func"], legend=contact_names, phase_mappings=plot_mappings)}

    @staticmethod
    def muscle_activations_and_torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)

        u = MX()
        nlp["nbMuscle"] = nlp["model"].nbMuscles()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["nu"] = nlp["u"].rows()

        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [Dynamics.forward_dynamics_torque_muscle_driven(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()  # .map(nlp["ns"], "thread", 2)

    @staticmethod
    def muscle_excitations_and_torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)

        u = MX()
        x = MX()
        nlp["nbMuscle"] = nlp["model"].nbMuscles()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_excitation"))
            x = vertcat(x, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["x"] = vertcat(nlp["x"], x)
        nlp["nu"] = nlp["u"].rows()
        nlp["nx"] = nlp["x"].rows()

        nlp["var_states"]["muscles"] = nlp["nbMuscle"]
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [Dynamics.forward_dynamics_muscle_excitations_and_torque_driven(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()  # .map(nlp["ns"], "thread", 2)

    @staticmethod
    def muscles_and_torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)

        u = MX()
        nlp["nbMuscle"] = nlp["model"].nbMuscles()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["nu"] = nlp["u"].rows()

        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [Dynamics.forward_dynamics_torque_muscle_driven_with_contact(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()  # .map(nlp["ns"], "thread", 2)

        nlp["contact_forces_func"] = Function(
            "contact_forces_func",
            [symbolic_states, symbolic_controls],
            [
                Dynamics.forces_from_forward_dynamics_torque_muscle_driven_with_contact(
                    symbolic_states, symbolic_controls, nlp
                )
            ],
            ["x", "u"],
            ["contact_forces"],
        ).expand()
        ProblemType.__configure_contact(nlp)
