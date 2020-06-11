from casadi import MX, SX, vertcat, Function

from .dynamics import Dynamics
from .mapping import BidirectionalMapping, Mapping
from .plot import CustomPlot
from .enums import PlotType


class ProblemType:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques but without muscles, must be used with dynamics without contacts.
        :param nlp: An instance of the OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        ProblemType.__configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_driven)

    @staticmethod
    def torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques, without muscles, must be used with dynamics with contacts.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        ProblemType.__configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_driven_with_contact)
        ProblemType.__configure_contact(ocp, nlp, Dynamics.forces_from_forward_dynamics_with_contact)

    @staticmethod
    def torque_activations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Controls u are torques and torques activations.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        nlp["nbActuators"] = nlp["nbTau"]
        ProblemType.__configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_activations_driven)

    @staticmethod
    def torque_activations_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Controls u are torques and torques activations.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        nlp["nbActuators"] = nlp["nbTau"]
        ProblemType.__configure_forward_dyn_func(
            ocp, nlp, Dynamics.forward_dynamics_torque_activations_driven_with_contact
        )
        ProblemType.__configure_contact(ocp, nlp, Dynamics.forces_from_forward_dynamics_with_contact)

    @staticmethod
    def muscle_activations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_muscles(nlp, False, True)

        u_mx = MX()
        u_sx = SX()
        for i in range(nlp["nbMuscle"]):
            u_mx = vertcat(u_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            u_sx = vertcat(u_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u_MX"] = vertcat(nlp["u_MX"], u_mx)
        nlp["u_SX"] = vertcat(nlp["u_SX"], u_sx)
        nlp["var_controls"] = {"muscles": nlp["nbMuscle"]}

        ProblemType.__configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_muscle_activations_driven)

    @staticmethod
    def muscle_activations_and_torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        ProblemType.__configure_muscles(nlp, False, True)

        u_mx = MX()
        u_sx = SX()
        for i in range(nlp["nbMuscle"]):
            u_mx = vertcat(u_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            u_sx = vertcat(u_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u_MX"] = vertcat(nlp["u_MX"], u_mx)
        nlp["u_SX"] = vertcat(nlp["u_SX"], u_sx)
        nlp["nu"] = nlp["u_MX"].rows()
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_muscle_driven)

    @staticmethod
    def muscle_excitations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_muscles(nlp, True, True)

        u_mx = MX()
        x_mx = MX()
        x_sx = SX()
        u_sx = SX()
        for i in range(nlp["nbMuscle"]):
            u_mx = vertcat(u_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x_mx = vertcat(x_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            u_sx = vertcat(u_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x_sx = vertcat(x_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u_MX"] = vertcat(nlp["u_MX"], u_mx)
        nlp["x_MX"] = vertcat(nlp["x_MX"], x_mx)
        nlp["u_SX"] = vertcat(nlp["u_SX"], u_sx)
        nlp["x_SX"] = vertcat(nlp["x_SX"], x_sx)
        nlp["var_states"]["muscles"] = nlp["nbMuscle"]
        nlp["var_controls"] = {"muscles": nlp["nbMuscle"]}

        ProblemType.__configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_muscle_excitations_driven)

    @staticmethod
    def muscle_excitations_and_torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        ProblemType.__configure_muscles(nlp, True, True)

        u_mx = MX()
        x_mx = MX()
        x_sx = SX()
        u_sx = SX()
        for i in range(nlp["nbMuscle"]):
            u_mx = vertcat(u_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x_mx = vertcat(x_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            u_sx = vertcat(u_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x_sx = vertcat(x_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u_MX"] = vertcat(nlp["u_MX"], u_mx)
        nlp["x_MX"] = vertcat(nlp["x_MX"], x_mx)
        nlp["u_SX"] = vertcat(nlp["u_SX"], u_sx)
        nlp["x_SX"] = vertcat(nlp["x_SX"], x_sx)
        nlp["var_states"]["muscles"] = nlp["nbMuscle"]
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(
            ocp, nlp, Dynamics.forward_dynamics_muscle_excitations_and_torque_driven
        )

    @staticmethod
    def muscles_activations_and_torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        ProblemType.__configure_muscles(nlp, False, True)

        u_mx = MX()
        u_sx = SX()
        for i in range(nlp["nbMuscle"]):
            u_mx = vertcat(u_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            u_sx = vertcat(u_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u_MX"] = vertcat(nlp["u_MX"], u_mx)
        nlp["u_SX"] = vertcat(nlp["u_SX"], u_sx)
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(
            ocp, nlp, Dynamics.forward_dynamics_muscle_activations_and_torque_driven_with_contact
        )
        ProblemType.__configure_contact(
            ocp, nlp, Dynamics.forces_from_forward_dynamics_muscle_activations_and_torque_driven_with_contact
        )

    @staticmethod
    def muscle_excitations_and_torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_q_qdot(nlp, True, False)
        ProblemType.__configure_tau(nlp, False, True)
        ProblemType.__configure_muscles(nlp, True, True)

        u_mx = MX()
        x_mx = MX()
        x_sx = SX()
        u_sx = SX()
        for i in range(nlp["nbMuscle"]):
            u_mx = vertcat(u_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x_mx = vertcat(x_mx, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            u_sx = vertcat(u_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x_sx = vertcat(x_sx, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u_MX"] = vertcat(nlp["u_MX"], u_mx)
        nlp["x_MX"] = vertcat(nlp["x_MX"], x_mx)
        nlp["u_SX"] = vertcat(nlp["u_SX"], u_sx)
        nlp["x_SX"] = vertcat(nlp["x_SX"], x_sx)
        nlp["var_states"]["muscles"] = nlp["nbMuscle"]
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(
            ocp, nlp, Dynamics.forward_dynamics_muscle_excitations_and_torque_driven_with_contact
        )
        ProblemType.__configure_contact(
            ocp, nlp, Dynamics.forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact
        )

    @staticmethod
    def __configure_q_qdot(nlp, as_states, as_controls):
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

        dof_names = nlp["model"].nameDof()
        q_mx = MX()
        q_dot_mx = MX()
        q_sx = SX()
        q_dot_sx = SX()
        for i in nlp["q_mapping"].reduce.map_idx:
            q_mx = vertcat(q_mx, MX.sym("Q_" + dof_names[i].to_string(), 1, 1))
            q_sx = vertcat(q_sx, SX.sym("Q_" + dof_names[i].to_string(), 1, 1))
        for i in nlp["q_dot_mapping"].reduce.map_idx:
            q_dot_mx = vertcat(q_dot_mx, MX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))
            q_dot_sx = vertcat(q_dot_sx, SX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))

        nlp["nbQ"] = nlp["q_mapping"].reduce.len
        nlp["nbQdot"] = nlp["q_dot_mapping"].reduce.len

        legend_q = ["q_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_mapping"].reduce.map_idx]
        legend_qdot = ["qdot_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_dot_mapping"].reduce.map_idx]

        if as_states:
            nlp["x_MX"] = vertcat(q_mx, q_dot_mx)
            nlp["x_SX"] = vertcat(q_sx, q_dot_sx)
            nlp["var_states"] = {"q": nlp["nbQ"], "q_dot": nlp["nbQdot"]}
            nlp["plot"]["q"] = CustomPlot(
                lambda x, u, p: x[: nlp["nbQ"]], plot_type=PlotType.INTEGRATED, legend=legend_q
            )
            nlp["plot"]["q_dot"] = CustomPlot(
                lambda x, u, p: x[nlp["nbQ"] : nlp["nbQ"] + nlp["nbQdot"]],
                plot_type=PlotType.INTEGRATED,
                legend=legend_qdot,
            )
        if as_controls:
            nlp["u_MX"] = vertcat(q_mx, q_dot_mx)
            nlp["u_SX"] = vertcat(q_sx, q_dot_sx)
            nlp["var_controls"] = {"q": nlp["nbQ"], "q_dot": nlp["nbQdot"]}
            # Add plot if it happens

    @staticmethod
    def __configure_tau(nlp, as_states, as_controls):
        """
        Configures common settings for torque driven problems with and without contacts.
        :param nlp: An OptimalControlProgram class.
        """
        if nlp["tau_mapping"] is None:
            nlp["tau_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbGeneralizedTorque())), Mapping(range(nlp["model"].nbGeneralizedTorque()))
            )

        dof_names = nlp["model"].nameDof()
        u_mx = MX()
        u_sx = SX()
        for i in nlp["tau_mapping"].reduce.map_idx:
            u_mx = vertcat(u_mx, MX.sym("Tau_" + dof_names[i].to_string(), 1, 1))
            u_sx = vertcat(u_sx, SX.sym("Tau_" + dof_names[i].to_string(), 1, 1))

        nlp["nbTau"] = nlp["tau_mapping"].reduce.len
        legend_tau = ["tau_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["tau_mapping"].reduce.map_idx]

        if as_states:
            nlp["x_MX"] = u_mx
            nlp["x_SX"] = u_sx
            nlp["var_states"] = {"tau": nlp["nbTau"]}
            # Add plot if it happens
        if as_controls:
            nlp["u_MX"] = u_mx
            nlp["u_SX"] = u_sx
            nlp["var_controls"] = {"tau": nlp["nbTau"]}
            nlp["plot"]["tau"] = CustomPlot(
                lambda x, u, p: u[: nlp["nbTau"]], plot_type=PlotType.STEP, legend=legend_tau
            )

    @staticmethod
    def __configure_contact(ocp, nlp, dyn_func):
        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        symbolic_param = nlp["p"]
        nlp["contact_forces_func"] = Function(
            "contact_forces_func",
            [symbolic_states, symbolic_controls, symbolic_param],
            [dyn_func(symbolic_states, symbolic_controls, symbolic_param, nlp)],
            ["x", "u", "p"],
            ["contact_forces"],
        ).expand()

        nlp["nbContact"] = nlp["model"].nbContacts()
        contact_names = [n.to_string() for n in nlp["model"].contactNames()]
        phase_mappings = nlp["plot_mappings"]["contact_forces"] if "contact_forces" in nlp["plot_mappings"] else None
        nlp["plot"]["contact_forces"] = CustomPlot(
            nlp["contact_forces_func"], axes_idx=phase_mappings, legend=contact_names
        )

    @staticmethod
    def __configure_muscles(nlp, as_states, as_controls):
        nlp["nbMuscle"] = nlp["model"].nbMuscles()
        nlp["muscleNames"] = [names.to_string() for names in nlp["model"].muscleNames()]

        combine = None
        if as_states:
            nx_q = nlp["nbQ"] + nlp["nbQdot"]
            nlp["plot"]["muscles_states"] = CustomPlot(
                lambda x, u, p: x[nx_q : nx_q + nlp["nbMuscle"]],
                plot_type=PlotType.INTEGRATED,
                legend=nlp["muscleNames"],
                ylim=[0, 1],
            )
            combine = "muscles_states"
        if as_controls:
            nlp["plot"]["muscles_control"] = CustomPlot(
                lambda x, u, p: u[nlp["nbTau"] : nlp["nbTau"] + nlp["nbMuscle"]],
                plot_type=PlotType.STEP,
                legend=nlp["muscleNames"],
                combine_to=combine,
                ylim=[0, 1],
            )

    @staticmethod
    def __configure_forward_dyn_func(ocp, nlp, dyn_func):
        nlp["nu"] = nlp["u_MX"].rows()
        nlp["nx"] = nlp["x_MX"].rows()

        symbolic_params = SX()
        nlp["parameters_to_optimize"] = ocp.param_to_optimize
        for key in nlp["parameters_to_optimize"]:
            symbolic_params = vertcat(symbolic_params, nlp["parameters_to_optimize"][key]["mx"])
        nlp["p_SX"] = symbolic_params
        nlp["np"] = symbolic_params.rows()

        MX_symbolic_states = MX.sym("x", nlp["nx"], 1)
        MX_symbolic_controls = MX.sym("u", nlp["nu"], 1)
        MX_symbolic_params = MX.sym("p", nlp["np"], 1)
        nlp["p_MX"] = MX_symbolic_params


        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [MX_symbolic_states, MX_symbolic_controls, MX_symbolic_params],
            [dyn_func(MX_symbolic_states, MX_symbolic_controls, MX_symbolic_params, nlp)],
            ["x", "u", "p"],
            ["xdot"],
        ).expand()  # .map(nlp["ns"], "thread", 2)
