from typing import Callable

from casadi import MX, vertcat, horzcat, Function
import numpy as np

from .dynamics_functions import DynamicsFunctions
from ..misc.enums import PlotType, ControlType
from ..misc.mapping import BiMapping, Mapping
from ..gui.plot import CustomPlot


class Problem:
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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_driven)

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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_driven_with_contact)
        Problem.configure_contact(
            ocp, nlp, DynamicsFunctions.forces_from_forward_dynamics_with_contact_for_torque_driven_problem
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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        nlp.shape["actuators"] = nlp.shape["tau"]
        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_torque_activations_driven)

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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        nlp.shape["actuators"] = nlp.shape["tau"]
        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_torque_activations_driven_with_contact
            )
        Problem.configure_contact(
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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_muscles(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_activations_driven)

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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, nlp.dynamics_type.dynamics)
        else:
            Problem.configure_dynamics_function(
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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_activations_and_torque_driven_with_contact
            )
        Problem.configure_contact(
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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_muscles(nlp, True, True)

        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_driven)

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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, True, True)

        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(
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

        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, True, True)

        if nlp.dynamics_type.dynamic_function:
            Problem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            Problem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.forward_dynamics_muscle_excitations_and_torque_driven_with_contact
            )
        Problem.configure_contact(
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

        nlp.shape["q"] = nlp.mapping["q"].to_first.len

        legend_q = ["q_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["q"].to_first.map_idx]

        nlp.q = q_mx
        if as_states:
            nlp.x = vertcat(nlp.x, q)
            nlp.var_states["q"] = nlp.shape["q"]
            q_bounds = nlp.x_bounds[: nlp.shape["q"]]

            nlp.plot["q"] = CustomPlot(
                lambda x, u, p: x[: nlp.shape["q"]],
                plot_type=PlotType.INTEGRATED,
                legend=legend_q,
                bounds=q_bounds,
            )

        if as_controls:
            nlp.u = vertcat(nlp.u, q)
            nlp.var_controls["q"] = nlp.shape["q"]
            # Add plot (and retrieving bounds if plots of bounds) if this problem is ever added

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()

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

        nlp.shape["qdot"] = nlp.mapping["qdot"].to_first.len

        legend_qdot = ["qdot_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["qdot"].to_first.map_idx]

        nlp.qdot = qdot_mx
        if as_states:
            nlp.x = vertcat(nlp.x, qdot)
            nlp.var_states["qdot"] = nlp.shape["qdot"]
            qdot_bounds = nlp.x_bounds[nlp.shape["q"] :]

            nlp.plot["qdot"] = CustomPlot(
                lambda x, u, p: x[nlp.shape["q"] : nlp.shape["q"] + nlp.shape["qdot"]],
                plot_type=PlotType.INTEGRATED,
                legend=legend_qdot,
                bounds=qdot_bounds,
            )

        if as_controls:
            nlp.u = vertcat(nlp.u, qdot)
            nlp.var_controls["qdot"] = nlp.shape["qdot"]
            # Add plot (and retrieving bounds if plots of bounds) if this problem is ever added

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()

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

        Problem.configure_q(nlp, as_states, as_controls)
        Problem.configure_qdot(nlp, as_states, as_controls)

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
        for i, _ in enumerate(nlp.mapping["q"].to_second.map_idx):
            tau_mx = vertcat(tau_mx, MX.sym("Tau_" + dof_names[i].to_string(), 1, 1))

        nlp.shape["tau"] = nlp.mapping["tau"].to_first.len
        legend_tau = ["tau_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["tau"].to_first.map_idx]
        nlp.tau = tau_mx

        if as_states:
            nlp.x = vertcat(nlp.x, all_tau[0])
            nlp.var_states["tau"] = nlp.shape["tau"]
            # Add plot if it happens

        if as_controls:
            nlp.u = vertcat(nlp.u, horzcat(*all_tau))
            nlp.var_controls["tau"] = nlp.shape["tau"]
            tau_bounds = nlp.u_bounds[: nlp.shape["tau"]]

            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                plot_type = PlotType.PLOT
            else:
                plot_type = PlotType.STEP
            nlp.plot["tau"] = (
                CustomPlot(
                    lambda x, u, p: u[: nlp.shape["tau"]], plot_type=plot_type, legend=legend_tau, bounds=tau_bounds
                ),
            )

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()

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

        nlp.shape["muscle"] = nlp.model.nbMuscles()
        nlp.muscleNames = [names.to_string() for names in nlp.model.muscleNames()]

        muscles_mx = MX()
        for name in nlp.muscleNames:
            muscles_mx = vertcat(muscles_mx, MX.sym(f"Muscle_{name}_{nlp.phase_idx}", 1, 1))
        nlp.muscles = muscles_mx

        combine = None
        if as_states:
            muscles = nlp.cx()
            for name in nlp.muscleNames:
                muscles = vertcat(muscles, nlp.cx.sym(f"Muscle_{name}_activation_{nlp.phase_idx}"))

            nlp.x = vertcat(nlp.x, muscles)
            nlp.var_states["muscles"] = nlp.shape["muscle"]

            nx_q = nlp.shape["q"] + nlp.shape["qdot"]
            muscles_bounds = nlp.x_bounds[nx_q : nx_q + nlp.shape["muscle"]]
            nlp.plot["muscles_states"] = CustomPlot(
                lambda x, u, p: x[nx_q : nx_q + nlp.shape["muscle"]],
                plot_type=PlotType.INTEGRATED,
                legend=nlp.muscleNames,
                ylim=[0, 1],
                bounds=muscles_bounds,
            )
            combine = "muscles_states"

        if as_controls:
            n_col = nlp.control_type.value
            all_muscles = [nlp.cx() for _ in range(n_col)]
            for j in range(len(all_muscles)):
                for name in nlp.muscleNames:
                    all_muscles[j] = vertcat(
                        all_muscles[j], nlp.cx.sym(f"Muscle_{name}_excitation_{j}_{nlp.phase_idx}", 1, 1)
                    )

            nlp.u = vertcat(nlp.u, horzcat(*all_muscles))
            nlp.var_controls["muscles"] = nlp.shape["muscle"]
            muscles_bounds = nlp.u_bounds[nlp.shape["tau"] : nlp.shape["tau"] + nlp.shape["muscle"]]

            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                plot_type = PlotType.LINEAR
            else:
                plot_type = PlotType.STEP
            nlp.plot["muscles_control"] = CustomPlot(
                lambda x, u, p: u[nlp.shape["tau"] : nlp.shape["tau"] + nlp.shape["muscle"]],
                plot_type=plot_type,
                legend=nlp.muscleNames,
                combine_to=combine,
                ylim=[0, 1],
                bounds=muscles_bounds,
            )

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()

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

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()
        mx_symbolic_states = MX.sym("x", nlp.nx, 1)
        mx_symbolic_controls = MX.sym("u", nlp.nu, 1)

        nlp.parameters = ocp.v.parameters_in_list
        nlp.p = ocp.v.parameters.cx

        if len(nlp.parameters):
            nlp.p_scaling = np.vstack([p.scaling for p in nlp.parameters])
        else:
            nlp.p_scaling = np.array([[1.]])
        nlp.np = sum([p.size for p in nlp.parameters])
        mx_symbolic_params = MX.sym("p", nlp.np, 1)

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

        symbolic_states = MX.sym("x", nlp.nx, 1)
        symbolic_controls = MX.sym("u", nlp.nu, 1)
        symbolic_param = MX.sym("p", nlp.np, 1)
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
            nlp.contact_forces_func, axes_idx=phase_mappings, legend=all_contact_names
        )
