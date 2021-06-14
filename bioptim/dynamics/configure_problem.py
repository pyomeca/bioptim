from typing import Callable, Any, Union
from enum import Enum

from casadi import MX, vertcat, horzcat, Function

from .dynamics_functions import DynamicsFunctions
from ..misc.enums import PlotType, ControlType
from ..misc.mapping import BiMapping, Mapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..gui.plot import CustomPlot


class ConfigureProblem:
    """
    Dynamics configuration for the most common ocp

    Methods
    -------

    """

    @staticmethod
    def initialize(ocp, nlp):
        """
        Call the dynamics a first time

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.type.value[0](ocp, nlp)

    @staticmethod
    def custom(ocp, nlp):
        """
        Call the user-defined dynamics configuration function

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.configure(ocp, nlp)

    @staticmethod
    def torque_driven(ocp, nlp):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_driven)

    @staticmethod
    def torque_derivative_driven(ocp, nlp):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, True, False)
        ConfigureProblem.configure_taudot(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_derivative_driven)

    @staticmethod
    def torque_driven_with_contact(ocp, nlp):
        """
        Configure the dynamics for a torque driven with contact program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_driven_with_contact)
        ConfigureProblem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_with_contact_for_torque_driven_problem
        )

    @staticmethod
    def torque_derivative_driven_with_contact(ocp, nlp):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, True, False)
        ConfigureProblem.configure_taudot(nlp, False, True)
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_torque_derivative_driven_with_contact
            )
        ConfigureProblem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_with_contact_for_torque_derivative_driven_problem
        )

    @staticmethod
    def torque_activations_driven(ocp, nlp):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau activations).
        The tau activations are bounded between -1 and 1 and actual tau is computed from torque-position-velocity
        relationship

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_activations_driven)

    @staticmethod
    def torque_activations_driven_with_contact(ocp, nlp):
        """
        Configure the dynamics for a torque with contact driven program (states are q and qdot,
        controls are tau activations). The tau activations are bounded between -1 and 1 and actual tau is computed
        from torque-position-velocity relationship

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_torque_activations_driven_with_contact
            )
        ConfigureProblem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_with_contact_for_torque_activation_driven_problem
        )

    @staticmethod
    def muscle_activations_driven(ocp, nlp):
        """
        Configure the dynamics for a muscle driven program (states are q and qdot, controls are the muscle activations).
        The muscle activations are bounded between 0 and 1 and actual tau is computed from muscle force resulting from
        the activations

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_muscles(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_activations_driven)

    @staticmethod
    def muscle_activations_and_torque_driven(ocp, nlp):
        """
        Configure the dynamics for a muscle and torque driven program (states are q and qdot, controls are tau and the
        muscle activations). The tau are used as supplementary force in the case muscles are too weak. The muscle
        activations are bounded between 0 and 1 and actual tau is computed from muscle force resulting from the
        activations and added to the tau control

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        ConfigureProblem.configure_muscles(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, nlp.dynamics_type.dynamics)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_activations_and_torque_driven
            )

    @staticmethod
    def muscle_activations_and_torque_driven_with_contact(ocp, nlp):
        """
        Configure the dynamics for a muscle and torque driven with contact program (states are q and qdot, controls are
        tau and the muscle activations). The tau are used as supplementary force in the case muscles are too weak. The
        muscle activations are bounded between 0 and 1 and actual tau is computed from muscle force resulting from the
        activations and added to the tau control

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        ConfigureProblem.configure_muscles(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_activations_and_torque_driven_with_contact
            )
        ConfigureProblem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_muscle_activations_and_torque_driven_with_contact
        )

    @staticmethod
    def muscle_excitations_driven(ocp, nlp):
        """
        Configure the dynamics for a muscle driven program (states are q, qdot and muscle activations, controls are the
        muscle excitations (EMG)). The muscle activations are computed from the muscle dynamics. The muscle excitations
        are bounded between 0 and 1 and actual tau is computed from muscle force resulting from the activations

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_muscles(nlp, True, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_driven)

    @staticmethod
    def muscle_excitations_and_torque_driven(ocp, nlp):
        """
        Configure the dynamics for a muscle and torque driven program (states are q, qdot and muscle activations,
        controls are tau and the muscle excitations (EMG)). The tau are used as supplementary force in the case muscles
        are too weak. The muscle activations are computed from the muscle dynamics. The muscle excitations
        are bounded between 0 and 1 and actual tau is computed from muscle force resulting from the activations added
        to the tau control

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        ConfigureProblem.configure_muscles(nlp, True, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_and_torque_driven
            )

    @staticmethod
    def muscle_excitations_and_torque_driven_with_contact(ocp, nlp):
        """
        Configure the dynamics for a muscle and torque driven with contact program (states are q, qdot and muscle
        activations, controls are tau and the muscle excitations (EMG)). The tau are used as supplementary force in the
        case muscles are too weak. The muscle activations are computed from the muscle dynamics. The muscle excitations
        are bounded between 0 and 1 and actual tau is computed from muscle force resulting from the activations added
        to the tau control

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        ConfigureProblem.configure_muscles(nlp, True, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_and_torque_driven_with_contact
            )
        ConfigureProblem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact
        )

    @staticmethod
    def configure_q(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized coordinates

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        """

        if nlp.mapping["q"] is None:
            nlp.mapping["q"] = BiMapping(range(nlp.model.nbQ()), range(nlp.model.nbQ()))

        dof_names = nlp.model.nameDof()
        q_mx = MX()
        q = nlp.cx()

        for i in nlp.mapping["q"].to_first.map_idx:
            q = vertcat(q, nlp.cx.sym("Q_" + dof_names[i].to_string(), 1, 1))
        for i, _ in enumerate(nlp.mapping["q"].to_second.map_idx):
            q_mx = vertcat(q_mx, MX.sym("Q_" + dof_names[i].to_string(), 1, 1))

        legend_q = ["q_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["q"].to_first.map_idx]
        if as_states:
            nlp.states.append("q", q, q_mx)
            nlp.plot["q"] = CustomPlot(
                lambda x, u, p: x[nlp.states["q"].index, :],
                plot_type=PlotType.INTEGRATED,
                legend=legend_q,
                bounds=nlp.x_bounds[nlp.states["q"].index],  # TODO This is empty (this is a bug)
            )

        if as_controls:
            nlp.controls.append("q", q, q_mx)
            # Add plot (and retrieving bounds if plots of bounds) if this problem is ever added

    @staticmethod
    def configure_qdot(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized velocities

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        """

        if nlp.mapping["qdot"] is None:
            nlp.mapping["qdot"] = BiMapping(range(nlp.model.nbQdot()), range(nlp.model.nbQdot()))

        dof_names = nlp.model.nameDof()
        qdot_mx = MX()
        qdot = nlp.cx()

        for i in nlp.mapping["qdot"].to_first.map_idx:
            qdot = vertcat(qdot, nlp.cx.sym("Qdot_" + dof_names[i].to_string(), 1, 1))
        for i, _ in enumerate(nlp.mapping["qdot"].to_second.map_idx):
            qdot_mx = vertcat(qdot_mx, MX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))

        legend_qdot = ["qdot_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["qdot"].to_first.map_idx]

        if as_states:
            nlp.states.append("qdot", qdot, qdot_mx)
            nlp.plot["qdot"] = CustomPlot(
                lambda x, u, p: x[nlp.states["qdot"].index, :],
                plot_type=PlotType.INTEGRATED,
                legend=legend_qdot,
                bounds=nlp.x_bounds[nlp.states["qdot"].index],  # TODO This is empty (this is a bug)
            )

        if as_controls:
            nlp.controls.append("qdot", qdot, qdot_mx)
            # Add plot (and retrieving bounds if plots of bounds) if this problem is ever added

    @staticmethod
    def configure_q_qdot(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized coordinates and generalized velocities

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates and generalized velocities should be states
        as_controls: bool
            If the generalized coordinates and generalized velocities should be controls
        """

        ConfigureProblem.configure_q(nlp, as_states, as_controls)
        ConfigureProblem.configure_qdot(nlp, as_states, as_controls)

    @staticmethod
    def configure_tau(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized forces should be a state
        as_controls: bool
            If the generalized forces should be a control
        """

        if nlp.mapping["tau"] is None:
            nlp.mapping["tau"] = BiMapping(
                range(nlp.model.nbGeneralizedTorque()), range(nlp.model.nbGeneralizedTorque())
            )

        dof_names = nlp.model.nameDof()

        n_col = nlp.control_type.value
        tau_mx = MX()
        all_tau = [nlp.cx() for _ in range(n_col)]

        for i in nlp.mapping["tau"].to_first.map_idx:
            for j in range(len(all_tau)):
                all_tau[j] = vertcat(all_tau[j], nlp.cx.sym(f"Tau_{dof_names[i].to_string()}_{j}", 1, 1))
        for i, _ in enumerate(nlp.mapping["tau"].to_second.map_idx):
            tau_mx = vertcat(tau_mx, MX.sym("Tau_" + dof_names[i].to_string(), 1, 1))

        legend_tau = ["tau_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["tau"].to_first.map_idx]

        if as_states:
            nlp.states.append("tau", horzcat(*all_tau), tau_mx)

            nlp.plot["tau"] = (
                CustomPlot(
                    lambda x, u, p: x[nlp.controls["tau"].index, :],
                    plot_type=PlotType.INTEGRATED,
                    legend=legend_tau,
                    bounds=nlp.u_bounds[nlp.controls["tau"].index],
                ),
            )

        if as_controls:
            nlp.controls.append("tau", horzcat(*all_tau), tau_mx)

            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                plot_type = PlotType.PLOT
            else:
                plot_type = PlotType.STEP
            nlp.plot["tau"] = (
                CustomPlot(
                    lambda x, u, p: u[nlp.controls["tau"].index, :],
                    plot_type=plot_type,
                    legend=legend_tau,
                    bounds=nlp.u_bounds[nlp.controls["tau"].index],
                ),
            )

    @staticmethod
    def configure_taudot(nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """

        if nlp.mapping["taudot"] is None:
            nlp.mapping["taudot"] = BiMapping(
                range(nlp.model.nbGeneralizedTorque()), range(nlp.model.nbGeneralizedTorque())
            )

        dof_names = nlp.model.nameDof()

        n_col = nlp.control_type.value
        taudot_mx = MX()
        all_taudot = [nlp.cx() for _ in range(n_col)]  # cx ?

        for i in nlp.mapping["taudot"].to_first.map_idx:
            for j in range(len(all_taudot)):
                all_taudot[j] = vertcat(all_taudot[j], nlp.cx.sym(f"Taudot_{dof_names[i].to_string()}_{j}", 1, 1))
        for i, _ in enumerate(nlp.mapping["q"].to_second.map_idx):
            taudot_mx = vertcat(taudot_mx, MX.sym("Taudot_" + dof_names[i].to_string(), 1, 1))

        nlp.shape["taudot"] = len(nlp.mapping["taudot"].to_first)
        legend_taudot = [
            "taudot_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["taudot"].to_first.map_idx
        ]

        if as_states:
            nlp.var_states.append("taudot", nlp.shape["taudot"], list(range(0, nlp.shape["taudot"])))
            nlp.x = vertcat(nlp.x, all_taudot[0])
            # Add plot if it happens, not sure it would

        if as_controls:
            nlp.var_controls.append("taudot", nlp.shape["taudot"], list(range(0, nlp.shape["taudot"])))
            nlp.u = vertcat(nlp.u, horzcat(*all_taudot))
            taudot_bounds = nlp.u_bounds[: nlp.shape["taudot"]]  # taudot as the only control.

            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                plot_type = PlotType.PLOT
            else:
                plot_type = PlotType.STEP
            nlp.plot["taudot"] = (
                CustomPlot(
                    lambda x, u, p: u[: nlp.shape["taudot"]],
                    plot_type=plot_type,
                    legend=legend_taudot,
                    bounds=taudot_bounds,
                ),
            )

    @staticmethod
    def configure_muscles(nlp, as_states: bool, as_controls: bool):
        """
        Configure the muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the muscles should be a state
        as_controls: bool
            If the muscles should be a control
        """

        muscle_names = [names.to_string() for names in nlp.model.muscleNames()]

        muscles_mx = MX()
        for name in muscle_names:
            muscles_mx = vertcat(muscles_mx, MX.sym(f"Muscle_{name}_{nlp.phase_idx}", 1, 1))

        combine = None
        if as_states:

            muscles = nlp.cx()
            for name in muscle_names:
                muscles = vertcat(muscles, nlp.cx.sym(f"Muscle_{name}_activation_{nlp.phase_idx}"))

            nlp.states.append("muscles", muscles, muscles_mx)

            muscles_bounds = nlp.x_bounds[nlp.states["muscles"].index]
            nlp.plot["muscles_states"] = CustomPlot(
                lambda x, u, p: x[nlp.states["muscles"].index, :],
                plot_type=PlotType.INTEGRATED,
                legend=muscle_names,
                ylim=[0, 1],
                bounds=muscles_bounds,
            )
            combine = "muscles_states"

        if as_controls:
            n_col = nlp.control_type.value
            all_muscles = [nlp.cx() for _ in range(n_col)]
            for j in range(len(all_muscles)):
                for name in muscle_names:
                    all_muscles[j] = vertcat(
                        all_muscles[j], nlp.cx.sym(f"Muscle_{name}_excitation_{j}_{nlp.phase_idx}", 1, 1)
                    )

            nlp.controls.append("muscles", horzcat(*all_muscles), muscles_mx)

            muscles_bounds = nlp.u_bounds[nlp.controls["muscles"].index]

            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                plot_type = PlotType.LINEAR
            else:
                plot_type = PlotType.STEP
            nlp.plot["muscles_control"] = CustomPlot(
                lambda x, u, p: u[nlp.controls["muscles"].index, :],
                plot_type=plot_type,
                legend=muscle_names,
                combine_to=combine,
                ylim=[0, 1],
                bounds=muscles_bounds,
            )

    @staticmethod
    def configure_dynamics_function(ocp, nlp, dyn_func):
        """
        Configure the forward dynamics

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[states, controls, param]
            The function to get the derivative of the states
        """

        mx_symbolic_states = MX.sym("x", nlp.states.shape, 1)
        mx_symbolic_controls = MX.sym("u", nlp.controls.shape, 1)

        nlp.parameters = ocp.v.parameters_in_list
        mx_symbolic_params = MX.sym("p", nlp.parameters.shape, 1)

        dynamics = dyn_func(mx_symbolic_states, mx_symbolic_controls, mx_symbolic_params, nlp)
        if isinstance(dynamics, (list, tuple)):
            dynamics = vertcat(*dynamics)
        nlp.dynamics_func = Function(
            "ForwardDyn",
            [mx_symbolic_states, mx_symbolic_controls, mx_symbolic_params],
            [dynamics],
            ["x", "u", "p"],
            ["xdot"],
        ).expand()

    @staticmethod
    def configure_contact(ocp, nlp, dyn_func: Callable):
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[states, controls, param]
            The function to get the values of contact forces from the dynamics
        """

        symbolic_states = MX.sym("x", nlp.states.shape, 1)
        symbolic_controls = MX.sym("u", nlp.controls.shape, 1)
        symbolic_param = MX.sym("p", nlp.parameters.shape, 1)
        nlp.contact_forces_func = Function(
            "contact_forces_func",
            [symbolic_states, symbolic_controls, symbolic_param],
            [dyn_func(symbolic_states, symbolic_controls, symbolic_param, nlp)],
            ["x", "u", "p"],
            ["contact_forces"],
        ).expand()

        all_contact_names = []
        for elt in ocp.nlp:
            all_contact_names.extend(
                [name.to_string() for name in elt.model.contactNames() if name.to_string() not in all_contact_names]
            )

        if "contact_forces" in nlp.mapping["plot"]:
            phase_mappings = nlp.mapping["plot"]["contact_forces"]
        else:
            contact_names_in_phase = [name.to_string() for name in nlp.model.contactNames()]
            phase_mappings = Mapping([i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase])

        nlp.plot["contact_forces"] = CustomPlot(
            nlp.contact_forces_func, plot_type=PlotType.INTEGRATED, axes_idx=phase_mappings, legend=all_contact_names
        )


class DynamicsFcn(Enum):
    """
    Selection of valid dynamics functions
    """

    TORQUE_DRIVEN = (ConfigureProblem.torque_driven,)
    TORQUE_DRIVEN_WITH_CONTACT = (ConfigureProblem.torque_driven_with_contact,)

    TORQUE_DERIVATIVE_DRIVEN = (ConfigureProblem.torque_derivative_driven,)
    TORQUE_DERIVATIVE_DRIVEN_WITH_CONTACT = (ConfigureProblem.torque_derivative_driven_with_contact,)

    TORQUE_ACTIVATIONS_DRIVEN = (ConfigureProblem.torque_activations_driven,)
    TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT = (ConfigureProblem.torque_activations_driven_with_contact,)

    MUSCLE_ACTIVATIONS_DRIVEN = (ConfigureProblem.muscle_activations_driven,)
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN = (ConfigureProblem.muscle_activations_and_torque_driven,)
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = (ConfigureProblem.muscle_activations_and_torque_driven_with_contact,)

    # TODO MUSCLE_ACTIVATIONS_AND_TORQUE_DERIVATIVE_DRIVEN = (ConfigureProblem.muscle_activations_and_torque_derivative_driven,)
    # TODO MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_DERIVATIVE_WITH_CONTACT = (ConfigureProblem.muscle_activations_and_torque_derivative_driven_with_contact,)

    MUSCLE_EXCITATIONS_DRIVEN = (ConfigureProblem.muscle_excitations_driven,)
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN = (ConfigureProblem.muscle_excitations_and_torque_driven,)
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = (ConfigureProblem.muscle_excitations_and_torque_driven_with_contact,)

    # TODO MUSCLE_EXCITATIONS_AND_TORQUE_DERIVATIVE_DRIVEN = (ConfigureProblem.muscle_excitations_and_torque_derivative_driven,)
    # TODO MUSCLE_EXCITATIONS_AND_TORQUE_DERIVATIVE_DRIVEN_WITH_CONTACT = (ConfigureProblem.muscle_excitations_and_torque_derivative_driven_with_contact,)

    CUSTOM = (ConfigureProblem.custom,)

class Dynamics(OptionGeneric):
    """
    A placeholder for the chosen dynamics by the user

    Attributes
    ----------
    dynamic_function: Callable
        The custom dynamic function provided by the user
    configure: Callable
        The configuration function provided by the user that declares the NLP (states and controls),
        usually only necessary when defining custom functions

    """

    def __init__(
        self,
        dynamics_type: Union[Callable, DynamicsFcn],
        configure: Callable = None,
        dynamic_function: Callable = None,
        **params
    ):
        """
        Parameters
        ----------
        dynamics_type: Union[Callable, DynamicsFcn]
            The chosen dynamic functions
        configure: Callable
            The configuration function provided by the user that declares the NLP (states and controls),
            usually only necessary when defining custom functions
        dynamic_function: Callable
            The custom dynamic function provided by the user
        params: dict
            Any parameters to pass to the dynamic and configure functions
        """

        if not isinstance(dynamics_type, DynamicsFcn):
            configure = dynamics_type
            dynamics_type = DynamicsFcn.CUSTOM

        super(Dynamics, self).__init__(type=dynamics_type, **params)
        self.dynamic_function = dynamic_function
        self.configure = configure


class DynamicsList(UniquePerPhaseOptionList):
    """
    A list of Dynamics if more than one is required, typically when more than one phases are declared

    Methods
    -------
    add(dynamics: DynamicsFcn, **extra_parameters)
        Add a new Dynamics to the list
    print(self)
        Print the DynamicsList to the console
    """

    def add(self, dynamics_type: Union[Callable, Dynamics, DynamicsFcn], **extra_parameters: Any):
        """
        Add a new Dynamics to the list

        Parameters
        ----------
        dynamics_type: Union[Callable, Dynamics, DynamicsFcn]
            The chosen dynamic functions
        extra_parameters: dict
            Any parameters to pass to Dynamics
        """

        if isinstance(dynamics_type, Dynamics):
            self.copy(dynamics_type)

        else:
            super(DynamicsList, self)._add(dynamics_type=dynamics_type, option_type=Dynamics, **extra_parameters)

    def print(self):
        """
        Print the DynamicsList to the console
        """
        raise NotImplementedError("Printing of DynamicsList is not ready yet")
