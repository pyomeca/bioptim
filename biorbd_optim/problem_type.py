from casadi import MX, SX, vertcat, Function
from enum import Enum

from .dynamics import Dynamics
from .mapping import BidirectionalMapping, Mapping
from .plot import CustomPlot
from .enums import PlotType


class Problem:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def initialize(ocp, nlp):
        nlp["problem_type"]["type"](ocp, nlp)

    @staticmethod
    def custom(ocp, nlp):
        nlp["problem_type"]["configure"](ocp, nlp)

    @staticmethod
    def torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques but without muscles, must be used with dynamics without contacts.
        :param nlp: An instance of the OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_driven)

    @staticmethod
    def torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques, without muscles, must be used with dynamics with contacts.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_driven_with_contact)
        Problem.configure_contact(ocp, nlp, Dynamics.forces_from_forward_dynamics_with_contact)

    @staticmethod
    def torque_activations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Controls u are torques and torques activations.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        nlp["nbActuators"] = nlp["nbTau"]
        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_activations_driven)

    @staticmethod
    def torque_activations_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Controls u are torques and torques activations.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        nlp["nbActuators"] = nlp["nbTau"]
        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(
                ocp, nlp, Dynamics.forward_dynamics_torque_activations_driven_with_contact
            )
        Problem.configure_contact(ocp, nlp, Dynamics.forces_from_forward_dynamics_with_contact)

    @staticmethod
    def muscle_activations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_muscles(ocp, nlp, False, True)

        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_muscle_activations_driven)

    @staticmethod
    def muscle_activations_and_torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        Problem.configure_muscles(ocp, nlp, False, True)

        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, nlp["problem_type"]["dynamic"])
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_torque_muscle_driven)

    @staticmethod
    def muscle_excitations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_muscles(ocp, nlp, True, True)

        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_muscle_excitations_driven)

    @staticmethod
    def muscle_excitations_and_torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        Problem.configure_muscles(ocp, nlp, True, True)

        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.forward_dynamics_muscle_excitations_and_torque_driven)

    @staticmethod
    def muscle_activations_and_torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        Problem.configure_muscles(ocp, nlp, False, True)

        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(
                ocp, nlp, Dynamics.forward_dynamics_muscle_activations_and_torque_driven_with_contact
            )
        Problem.configure_contact(
            ocp, nlp, Dynamics.forces_from_forward_dynamics_muscle_activations_and_torque_driven_with_contact
        )

    @staticmethod
    def muscle_excitations_and_torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(ocp, nlp, True, False)
        Problem.configure_tau(ocp, nlp, False, True)
        Problem.configure_muscles(ocp, nlp, True, True)

        if "dynamic" in nlp["problem_type"]:
            Problem.configure_forward_dyn_func(ocp, nlp, Dynamics.custom)
        else:
            Problem.configure_forward_dyn_func(
                ocp, nlp, Dynamics.forward_dynamics_muscle_excitations_and_torque_driven_with_contact
            )
        Problem.configure_contact(
            ocp, nlp, Dynamics.forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact
        )

    @staticmethod
    def configure_q_qdot(ocp, nlp, as_states, as_controls):
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
        q_mx_expand = MX()
        q_dot_mx = MX()
        q_dot_mx_expand = MX()
        q_sx = SX()
        q_dot_sx = SX()

        for i in nlp["q_mapping"].reduce.map_idx:
            q_mx = vertcat(q_mx, MX.sym("Q_" + dof_names[i].to_string(), 1, 1))
            q_sx = vertcat(q_sx, SX.sym("Q_" + dof_names[i].to_string(), 1, 1))
        for i in nlp["q_dot_mapping"].reduce.map_idx:
            q_dot_mx = vertcat(q_dot_mx, MX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))
            q_dot_sx = vertcat(q_dot_sx, SX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))
        for i in range(len(nlp["q_mapping"].expand.map_idx)):
            q_mx_expand = vertcat(q_mx_expand, MX.sym("Q_expand_" + dof_names[i].to_string(), 1, 1))
        for i in range(len(nlp["q_dot_mapping"].expand.map_idx)):
            q_dot_mx_expand = vertcat(q_dot_mx_expand, MX.sym("Qdot_expand_" + dof_names[i].to_string(), 1, 1))

        nlp["nbQ"] = nlp["q_mapping"].reduce.len
        nlp["nbQdot"] = nlp["q_dot_mapping"].reduce.len

        legend_q = ["q_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_mapping"].reduce.map_idx]
        legend_qdot = ["qdot_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_dot_mapping"].reduce.map_idx]

        if as_states:
            nlp["q_MX"] = q_mx
            nlp["q_MX_expand"] = q_mx_expand
            nlp["qdot_MX"] = q_dot_mx
            nlp["qdot_MX_expand"] = q_dot_mx_expand
            if ocp.with_SX:
                nlp["x"] = vertcat(nlp["x"], q_sx, q_dot_sx)
            else:
                nlp["x"] = vertcat(nlp["x"], q_mx, q_dot_mx)
            nlp["var_states"]["q"] = nlp["nbQ"]
            nlp["var_states"]["q_dot"] = nlp["nbQdot"]

            nlp["plot"]["q"] = CustomPlot(
                lambda x, u, p: x[: nlp["nbQ"]], plot_type=PlotType.INTEGRATED, legend=legend_q
            )
            nlp["plot"]["q_dot"] = CustomPlot(
                lambda x, u, p: x[nlp["nbQ"] : nlp["nbQ"] + nlp["nbQdot"]],
                plot_type=PlotType.INTEGRATED,
                legend=legend_qdot,
            )
        if as_controls:
            if ocp.with_SX:
                nlp["u"] = vertcat(nlp["u"], q_sx, q_dot_sx)
            else:
                nlp["u"] = vertcat(nlp["u"], q_mx, q_dot_mx)
            nlp["var_controls"]["q"] = nlp["nbQ"]
            nlp["var_controls"]["q_dot"] = nlp["nbQdot"]
            # Add plot if it happens

        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

    @staticmethod
    def configure_tau(ocp, nlp, as_states, as_controls):
        """
        Configures common settings for torque driven problems with and without contacts.
        :param nlp: An OptimalControlProgram class.
        """
        if nlp["tau_mapping"] is None:
            nlp["tau_mapping"] = BidirectionalMapping(
                # Mapping(range(nlp["model"].nbGeneralizedTorque())), Mapping(range(nlp["model"].nbGeneralizedTorque()))
                Mapping(range(nlp["model"].nbQdot())),
                Mapping(
                    range(nlp["model"].nbQdot())
                ),  # To change when nlp["model"].nbGeneralizedTorque() will return the proper number
            )

        dof_names = nlp["model"].nameDof()
        if ocp.with_SX:
            tau = SX()
            for i in nlp["tau_mapping"].reduce.map_idx:
                tau = vertcat(tau, SX.sym("Tau_" + dof_names[i].to_string(), 1, 1))
        else:
            tau = MX()
            for i in nlp["tau_mapping"].reduce.map_idx:
                tau = vertcat(tau, MX.sym("Tau_" + dof_names[i].to_string(), 1, 1))

        nlp["nbTau"] = nlp["tau_mapping"].reduce.len
        legend_tau = ["tau_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["tau_mapping"].reduce.map_idx]

        if as_states:
            nlp["x"] = vertcat(nlp["x"], tau)
            nlp["var_states"]["tau"] = nlp["nbTau"]

            # Add plot if it happens
        if as_controls:
            nlp["u"] = vertcat(nlp["u"], tau)
            nlp["var_controls"]["tau"] = nlp["nbTau"]

            nlp["plot"]["tau"] = CustomPlot(
                lambda x, u, p: u[: nlp["nbTau"]], plot_type=PlotType.STEP, legend=legend_tau
            )

        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

    @staticmethod
    def configure_contact(ocp, nlp, dyn_func):
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

        all_contact_names = []
        for elt in ocp.nlp:
            all_contact_names.extend(
                [name.to_string() for name in elt["model"].contactNames() if name.to_string() not in all_contact_names]
            )

        if "contact_forces" in nlp["plot_mappings"]:
            phase_mappings = nlp["plot_mappings"]["contact_forces"]
        else:
            contact_names_in_phase = [name.to_string() for name in nlp["model"].contactNames()]
            phase_mappings = Mapping([i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase])

        nlp["plot"]["contact_forces"] = CustomPlot(
            nlp["contact_forces_func"], axes_idx=phase_mappings, legend=all_contact_names
        )

    @staticmethod
    def configure_muscles(ocp, nlp, as_states, as_controls):
        nlp["nbMuscle"] = nlp["model"].nbMuscles()
        nlp["muscleNames"] = [names.to_string() for names in nlp["model"].muscleNames()]

        combine = None
        if as_states:
            if ocp.with_SX:
                muscles = SX()
                for i in range(nlp["nbMuscle"]):
                    muscles = vertcat(muscles, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            else:
                muscles = MX()
                for i in range(nlp["nbMuscle"]):
                    muscles = vertcat(muscles, SX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
            nlp["x"] = vertcat(nlp["x"], muscles)
            nlp["var_states"]["muscles"] = nlp["nbMuscle"]

            nx_q = nlp["nbQ"] + nlp["nbQdot"]
            nlp["plot"]["muscles_states"] = CustomPlot(
                lambda x, u, p: x[nx_q : nx_q + nlp["nbMuscle"]],
                plot_type=PlotType.INTEGRATED,
                legend=nlp["muscleNames"],
                ylim=[0, 1],
            )
            combine = "muscles_states"

        if as_controls:
            if ocp.with_SX:
                muscles = SX()
                for i in range(nlp["nbMuscle"]):
                    muscles = vertcat(muscles, SX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            else:
                muscles = MX()
                for i in range(nlp["nbMuscle"]):
                    muscles = vertcat(muscles, MX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            nlp["u"] = vertcat(nlp["u"], muscles)
            nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

            nlp["plot"]["muscles_control"] = CustomPlot(
                lambda x, u, p: u[nlp["nbTau"] : nlp["nbTau"] + nlp["nbMuscle"]],
                plot_type=PlotType.STEP,
                legend=nlp["muscleNames"],
                combine_to=combine,
                ylim=[0, 1],
            )

        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

    @staticmethod
    def configure_forward_dyn_func(ocp, nlp, dyn_func):
        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()
        MX_symbolic_states = MX.sym("x", nlp["nx"], 1)
        MX_symbolic_controls = MX.sym("u", nlp["nu"], 1)

        if ocp.with_SX:
            symbolic_params = SX()
        else:
            symbolic_params = MX()
        nlp["p"] = symbolic_params
        nlp["parameters_to_optimize"] = ocp.param_to_optimize
        for key in nlp["parameters_to_optimize"]:
            symbolic_params = vertcat(symbolic_params, nlp["parameters_to_optimize"][key]["sym_var"])
        nlp["np"] = symbolic_params.rows()
        MX_symbolic_params = MX.sym("p", nlp["np"], 1)

        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [MX_symbolic_states, MX_symbolic_controls, MX_symbolic_params],
            [dyn_func(MX_symbolic_states, MX_symbolic_controls, MX_symbolic_params, nlp)],
            ["x", "u", "p"],
            ["xdot"],
        ).expand()


class ProblemType(Enum):
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN = Problem.muscle_excitations_and_torque_driven
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN = Problem.muscle_activations_and_torque_driven
    MUSCLE_ACTIVATIONS_DRIVEN = Problem.muscle_activations_driven
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = Problem.muscle_excitations_and_torque_driven_with_contact
    MUSCLE_EXCITATIONS_DRIVEN = Problem.muscle_excitations_driven
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = Problem.muscle_activations_and_torque_driven_with_contact

    TORQUE_DRIVEN = Problem.torque_driven
    TORQUE_ACTIVATIONS_DRIVEN = Problem.torque_activations_driven
    TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT = Problem.torque_activations_driven_with_contact
    TORQUE_DRIVEN_WITH_CONTACT = Problem.torque_driven_with_contact

    CUSTOM = Problem.custom
