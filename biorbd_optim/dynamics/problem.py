from casadi import MX, vertcat, horzcat, Function

from .dynamics_functions import DynamicsFunctions
from ..misc.enums import PlotType, ControlType
from ..misc.mapping import BidirectionalMapping, Mapping
from ..gui.plot import CustomPlot


class Problem:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def initialize(ocp, nlp):
        nlp["dynamics_type"].type.value[0](ocp, nlp)

    @staticmethod
    def custom(ocp, nlp):
        nlp["dynamics_type"].configure(ocp, nlp)

    @staticmethod
    def torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques but without muscles, must be used with dynamics without contacts.
        :param nlp: An instance of the OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_driven)

    @staticmethod
    def torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques, without muscles, must be used with dynamics with contacts.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_driven_with_contact)
        Problem.configure_contact(ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_with_contact)

    @staticmethod
    def torque_activations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Controls u are torques and torques activations.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        nlp["nbActuators"] = nlp["nbTau"]
        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_activations_driven)

    @staticmethod
    def torque_activations_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Controls u are torques and torques activations.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        nlp["nbActuators"] = nlp["nbTau"]
        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(
                ocp, nlp, DynamicsFunctions.forward_dynamics_torque_activations_driven_with_contact
            )
        Problem.configure_contact(ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_with_contact)

    @staticmethod
    def muscle_activations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_muscles(nlp, False, True)

        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_activations_driven)

    @staticmethod
    def muscle_activations_and_torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, False, True)

        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, nlp["dynamics_type"].dynamics)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_muscle_driven)

    @staticmethod
    def muscle_excitations_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_muscles(nlp, True, True)

        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_driven)

    @staticmethod
    def muscle_excitations_and_torque_driven(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, True, True)

        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_and_torque_driven
            )

    @staticmethod
    def muscle_activations_and_torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, False, True)

        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_activations_and_torque_driven_with_contact
            )
        Problem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_muscle_activations_and_torque_driven_with_contact
        )

    @staticmethod
    def muscle_excitations_and_torque_driven_with_contact(ocp, nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, True, True)

        if nlp["dynamics_type"].dynamics:
            Problem.configure_forward_dyn_func(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_forward_dyn_func(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_and_torque_driven_with_contact
            )
        Problem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact
        )

    @staticmethod
    def configure_q_qdot(nlp, as_states, as_controls):
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
        q = nlp["CX"]()
        q_dot = nlp["CX"]()

        for i in nlp["q_mapping"].reduce.map_idx:
            q = vertcat(q, nlp["CX"].sym("Q_" + dof_names[i].to_string(), 1, 1))
        for i in nlp["q_dot_mapping"].reduce.map_idx:
            q_dot = vertcat(q_dot, nlp["CX"].sym("Qdot_" + dof_names[i].to_string(), 1, 1))
        for i in nlp["q_mapping"].expand.map_idx:
            q_mx = vertcat(q_mx, MX.sym("Q_" + dof_names[i].to_string(), 1, 1))
        for i in nlp["q_dot_mapping"].expand.map_idx:
            q_dot_mx = vertcat(q_dot_mx, MX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))

        nlp["nbQ"] = nlp["q_mapping"].reduce.len
        nlp["nbQdot"] = nlp["q_dot_mapping"].reduce.len

        legend_q = ["q_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_mapping"].reduce.map_idx]
        legend_qdot = ["qdot_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_dot_mapping"].reduce.map_idx]

        nlp["q"] = q_mx
        nlp["qdot"] = q_dot_mx
        if as_states:
            nlp["x"] = vertcat(nlp["x"], q, q_dot)
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
            nlp["u"] = vertcat(nlp["u"], q, q_dot)
            nlp["var_controls"]["q"] = nlp["nbQ"]
            nlp["var_controls"]["q_dot"] = nlp["nbQdot"]
            # Add plot if it happens

        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

    @staticmethod
    def configure_tau(nlp, as_states, as_controls):
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

        n_col = nlp["control_type"].value
        tau_mx = MX()
        all_tau = [nlp["CX"]() for _ in range(n_col)]

        for i in nlp["tau_mapping"].reduce.map_idx:
            for j in range(len(all_tau)):
                all_tau[j] = vertcat(all_tau[j], nlp["CX"].sym(f"Tau_{dof_names[i].to_string()}_{j}", 1, 1))
        for i in nlp["q_mapping"].expand.map_idx:
            tau_mx = vertcat(tau_mx, MX.sym("Tau_" + dof_names[i].to_string(), 1, 1))

        nlp["nbTau"] = nlp["tau_mapping"].reduce.len
        legend_tau = ["tau_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["tau_mapping"].reduce.map_idx]
        nlp["tau"] = tau_mx

        if as_states:
            nlp["x"] = vertcat(nlp["x"], all_tau[0])
            nlp["var_states"]["tau"] = nlp["nbTau"]
            # Add plot if it happens

        if as_controls:
            nlp["u"] = vertcat(nlp["u"], horzcat(*all_tau))
            nlp["var_controls"]["tau"] = nlp["nbTau"]

            if nlp["control_type"] == ControlType.LINEAR_CONTINUOUS:
                plot_type = PlotType.PLOT
            else:
                plot_type = PlotType.STEP
            nlp["plot"]["tau"] = CustomPlot(lambda x, u, p: u[: nlp["nbTau"]], plot_type=plot_type, legend=legend_tau)

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
    def configure_muscles(nlp, as_states, as_controls):
        nlp["nbMuscle"] = nlp["model"].nbMuscles()
        nlp["muscleNames"] = [names.to_string() for names in nlp["model"].muscleNames()]

        muscles_mx = MX()
        for name in nlp["muscleNames"]:
            muscles_mx = vertcat(muscles_mx, MX.sym(f"Muscle_{name}", 1, 1))
        nlp["muscles"] = muscles_mx

        combine = None
        if as_states:
            muscles = nlp["CX"]()
            for name in nlp["muscleNames"]:
                muscles = vertcat(muscles, nlp["CX"].sym(f"Muscle_{name}_activation"))

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
            n_col = nlp["control_type"].value
            all_muscles = [nlp["CX"]() for _ in range(n_col)]
            for j in range(len(all_muscles)):
                for name in nlp["muscleNames"]:
                    all_muscles[j] = vertcat(all_muscles[j], nlp["CX"].sym(f"Muscle_{name}_excitation_{j}", 1, 1))

            nlp["u"] = vertcat(nlp["u"], horzcat(*all_muscles))
            nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

            if nlp["control_type"] == ControlType.LINEAR_CONTINUOUS:
                plot_type = PlotType.LINEAR
            else:
                plot_type = PlotType.STEP
            nlp["plot"]["muscles_control"] = CustomPlot(
                lambda x, u, p: u[nlp["nbTau"] : nlp["nbTau"] + nlp["nbMuscle"]],
                plot_type=plot_type,
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

        symbolic_params = nlp["CX"]()
        nlp["parameters_to_optimize"] = ocp.param_to_optimize
        for key in nlp["parameters_to_optimize"]:
            symbolic_params = vertcat(symbolic_params, nlp["parameters_to_optimize"][key]["cx"])
        nlp["p"] = symbolic_params
        nlp["np"] = symbolic_params.rows()
        MX_symbolic_params = MX.sym("p", nlp["np"], 1)

        dynamics = dyn_func(MX_symbolic_states, MX_symbolic_controls, MX_symbolic_params, nlp)
        if isinstance(dynamics, (list, tuple)):
            dynamics = vertcat(*dynamics)
        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [MX_symbolic_states, MX_symbolic_controls, MX_symbolic_params],
            [dynamics],
            ["x", "u", "p"],
            ["xdot"],
        ).expand()
