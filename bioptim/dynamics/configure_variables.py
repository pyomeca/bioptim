from typing import Callable, Any
import numpy as np
from casadi import DM, vertcat, Function

from .configure_new_variable import NewVariableConfiguration
from .fatigue.fatigue_dynamics import FatigueList
from ..misc.enums import PlotType
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping, Mapping
from ..misc.options import OptionGeneric
from ..models.protocols.stochastic_biomodel import StochasticBioModel
from ..dynamics.ode_solvers import OdeSolver
from ..gui.plot import CustomPlot

class ConfigureVariables:

    @staticmethod
    def configure_new_variable(
            name: str,
            name_elements: list,
            ocp,
            nlp,
            as_states: bool,
            as_controls: bool,
            as_algebraic_states: bool = False,
            fatigue: FatigueList = None,
            combine_name: str = None,
            combine_state_control_plot: bool = False,
            skip_plot: bool = False,
            axes_idx: BiMapping = None,
    ):
        """
        Add a new variable to the states/controls pool

        Parameters
        ----------
        name: str
            The name of the new variable to add
        name_elements: list[str]
            The name of each element of the vector
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the new variable should be added to the state variable set
        as_controls: bool
            If the new variable should be added to the control variable set
        as_algebraic_states: bool
            If the new variable should be added to the algebraic states variable set
        fatigue: FatigueList
            The list of fatigable item
        combine_name: str
            The name of a previously added plot to combine to
        combine_state_control_plot: bool
            If states and controls plot should be combined. Only effective if as_states and as_controls are both True
        skip_plot: bool
            If no plot should be automatically added
        axes_idx: BiMapping
            The axes index to use for the plot
        """
        NewVariableConfiguration(
            name,
            name_elements,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_algebraic_states,
            fatigue,
            combine_name,
            combine_state_control_plot,
            skip_plot,
            axes_idx,
        )


    @staticmethod
    def configure_integrated_value(
            name: str,
            name_elements: list,
            ocp,
            nlp,
            initial_matrix: DM,
    ):
        """
        Add a new integrated value. This creates an MX (not an optimization variable) that is integrated using the
        integrated_value_functions function provided. This integrated_value can be used in the constraints and objectives
        without having to recompute them over and over again.

        Parameters
        ----------
        name: str
            The name of the new variable to add
        name_elements: list[str]
            The name of each element of the vector
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        initial_matrix: DM
            The initial value of the integrated value
        """

        # TODO: compute values at collocation points
        # but for now only cx_start can be used
        n_cx = (
            nlp.dynamics_type.ode_solver.n_cx - 1
            if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 3
        )
        if n_cx < 3:
            n_cx = 3

        dummy_mapping = Mapping(list(range(len(name_elements))))
        initial_vector = StochasticBioModel.reshape_to_vector(initial_matrix)
        cx_scaled_next_formatted = [initial_vector for _ in range(n_cx)]
        nlp.integrated_values.append(
            name=name,
            cx=cx_scaled_next_formatted,
            cx_scaled=cx_scaled_next_formatted,  # Only the first value is used
            mapping=dummy_mapping,
            node_index=0,
        )
        for node_index in range(1, nlp.ns + 1):  # cannot use phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
            cx_scaled_next = [nlp.integrated_value_functions[name](nlp, node_index) for _ in range(n_cx)]
            nlp.integrated_values.append(
                name,
                cx_scaled_next_formatted,
                cx_scaled_next,
                dummy_mapping,
                node_index,
            )


    @staticmethod
    def configure_q(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
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
        as_algebraic_states: bool
            If the generalized coordinates should be an algebraic state
        """
        name = "q"
        name_q = nlp.model.name_dof
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(name, name_q, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states, axes_idx=axes_idx)


    @staticmethod
    def configure_qdot(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
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
        as_algebraic_states: bool
            If the generalized velocities should be an algebraic state
        """

        name = "qdot"
        name_qdot = ConfigureVariables._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(name, name_qdot, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states, axes_idx=axes_idx)


    @staticmethod
    def configure_qddot(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized accelerations

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        as_algebraic_states: bool
            If the generalized velocities should be an algebraic state
        """

        name = "qddot"
        name_qddot = ConfigureVariables._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(name, name_qddot, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states, axes_idx=axes_idx)


    @staticmethod
    def configure_qdddot(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized accelerations

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        as_algebraic_states: bool
            If the generalized velocities should be an algebraic state
        """

        name = "qdddot"
        name_qdddot = ConfigureVariables._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(name, name_qdddot, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states, axes_idx=axes_idx)


    @staticmethod
    def configure_stochastic_k(ocp, nlp, n_noised_controls: int, n_references: int):
        """
        Configure the optimal feedback gain matrix K.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "k"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_k = []
        control_names = [f"control_{i}" for i in range(n_noised_controls)]
        ref_names = [f"feedback_{i}" for i in range(n_references)]
        for name_1 in control_names:
            for name_2 in ref_names:
                name_k += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(
            list(range(len(control_names) * len(ref_names))), list(range(len(control_names) * len(ref_names)))
        )
        ConfigureVariables.configure_new_variable(
            name,
            name_k,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
        )


    @staticmethod
    def configure_stochastic_c(ocp, nlp, n_noised_states: int, n_noise: int):
        """
        Configure the stochastic variable matrix C representing the injection of motor noise (df/dw).
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "c"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states variables and mapping cannot be use together for now.")

        name_c = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noise)]:
                name_c += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(
            list(range(n_noised_states * n_noise)), list(range(n_noised_states * n_noise))
        )

        ConfigureVariables.configure_new_variable(
            name,
            name_c,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
            skip_plot=True,
        )


    @staticmethod
    def configure_stochastic_a(ocp, nlp, n_noised_states: int):
        """
        Configure the stochastic variable matrix A representing the propagation of motor noise (df/dx).
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "a"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_a = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_a += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states ** 2)), list(range(n_noised_states ** 2)))

        ConfigureVariables.configure_new_variable(
            name,
            name_a,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
            skip_plot=True,
        )


    @staticmethod
    def configure_stochastic_cov_explicit(ocp, nlp, n_noised_states: int, initial_matrix: DM):
        """
        Configure the covariance matrix P representing the motor noise.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "cov"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_cov = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_cov += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states ** 2)), list(range(n_noised_states ** 2)))
        ConfigureVariables.configure_integrated_value(
            name,
            name_cov,
            ocp,
            nlp,
            initial_matrix=initial_matrix,
        )


    @staticmethod
    def configure_stochastic_cov_implicit(ocp, nlp, n_noised_states: int):
        """
        Configure the covariance matrix P representing the motor noise.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "cov"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_cov = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_cov += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states ** 2)), list(range(n_noised_states ** 2)))
        ConfigureVariables.configure_new_variable(
            name,
            name_cov,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
        )


    @staticmethod
    def configure_stochastic_cholesky_cov(ocp, nlp, n_noised_states: int):
        """
        Configure the diagonal matrix needed to reconstruct the covariance matrix using L @ L.T.
        This formulation allows insuring that the covariance matrix is always positive semi-definite.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "cholesky_cov"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_cov = []
        for nb_1, name_1 in enumerate([f"X_{i}" for i in range(n_noised_states)]):
            for name_2 in [f"X_{i}" for i in range(nb_1 + 1)]:
                name_cov += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(len(name_cov))), list(range(len(name_cov))))
        ConfigureVariables.configure_new_variable(
            name,
            name_cov,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
        )


    @staticmethod
    def configure_stochastic_ref(ocp, nlp, n_references: int):
        """
        Configure the reference kinematics.

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "ref"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_ref = [f"reference_{i}" for i in range(n_references)]
        nlp.variable_mappings[name] = BiMapping(list(range(n_references)), list(range(n_references)))
        ConfigureVariables.configure_new_variable(
            name,
            name_ref,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
        )


    @staticmethod
    def configure_stochastic_m(ocp, nlp, n_noised_states: int):
        """
        Configure the helper matrix M (from Gillis 2013 : https://doi.org/10.1109/CDC.2013.6761121).

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        name = "m"

        if "m" in nlp.variable_mappings and nlp.variable_mappings["m"].actually_does_a_mapping:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_m = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_m += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(
            list(range(n_noised_states * n_noised_states)),
            list(range(n_noised_states * n_noised_states)),
        )
        ConfigureVariables.configure_new_variable(
            name,
            name_m,
            ocp,
            nlp,
            as_states=False,
            as_controls=False,
            as_algebraic_states=True,
        )


    @staticmethod
    def configure_tau(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool, fatigue: FatigueList = None):
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
        as_algebraic_states: bool
            If the generalized forces should be an algebraic state
        fatigue: FatigueList
            If the dynamics with fatigue should be declared
        """

        name = "tau"
        name_tau = ConfigureVariables._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name, name_tau, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states, fatigue=fatigue, axes_idx=axes_idx
        )


    @staticmethod
    def configure_residual_tau(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the residual forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized forces should be a state
        as_controls: bool
            If the generalized forces should be a control
        as_algebraic_states: bool
            If the generalized forces should be an algebraic state
        """

        name = "residual_tau"
        name_residual_tau = ConfigureVariables._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name, name_residual_tau, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states, axes_idx=axes_idx
        )


    @staticmethod
    def configure_taudot(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        as_algebraic_states: bool
            If the generalized force derivatives should be an algebraic state
        """

        name = "taudot"
        name_taudot = ConfigureVariables._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(name, name_taudot, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states, axes_idx=axes_idx)


    @staticmethod
    def configure_translational_forces(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool, n_contacts: int = 1):
        """
        Configure contact forces as optimization variables (for now only in global reference frame with an unknown point of application))
        # TODO: Match this with ExternalForceSetTimeSeries (options: 'in_global', 'torque', ...)

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the contact force should be a state
        as_controls: bool
            If the contact force should be a control
        as_algebraic_states: bool
            If the contact force should be an algebraic state
        n_contacts: int
            The number of contacts to consider (There will be 3 components for each contact)
        """

        name_contact_forces = [f"Force{i}_{axis}" for i in range(n_contacts) for axis in ("X", "Y", "Z")]
        ConfigureVariables.configure_new_variable("contact_forces", name_contact_forces, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states)
        ConfigureVariables.configure_new_variable(
            "contact_positions", name_contact_forces, ocp, nlp, as_states, as_controls
        )


    @staticmethod
    def configure_rigid_contact_forces(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        as_algebraic_states: bool
            If the generalized force derivatives should be an algebraic state
        """

        name_contact_forces = [name for name in nlp.model.contact_names]
        ConfigureVariables.configure_new_variable(
            "rigid_contact_forces", name_contact_forces, ocp, nlp, as_states=as_states, as_controls=as_controls, as_algebraic_states=as_algebraic_states
        )


    @staticmethod
    def configure_soft_contact_forces(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        as_algebraic_states: bool
            If the generalized force derivatives should be an algebraic state
        """
        name_soft_contact_forces = [
            f"{name}_{axis}" for name in nlp.model.soft_contact_names for axis in ("MX", "MY", "MZ", "FX", "FY", "FZ")
        ]
        ConfigureVariables.configure_new_variable(
            "soft_contact_forces",
            name_soft_contact_forces,
            ocp,
            nlp,
            as_states=as_states,
            as_algebraic_states=as_algebraic_states,
            as_controls=as_controls,
        )


    @staticmethod
    def configure_muscles(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool, fatigue: FatigueList = None):
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
        as_algebraic_states: bool
            If the muscles should be an algebraic state
        fatigue: FatigueList
            The list of fatigue parameters
        """

        muscle_names = nlp.model.muscle_names
        ConfigureVariables.configure_new_variable(
            "muscles",
            muscle_names,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            combine_state_control_plot=True,
            fatigue=fatigue,
        )


    @staticmethod
    def configure_qv(ocp, nlp, dyn_func: Callable, **extra_params):
        """
        Configure the qv, i.e. the dependent joint coordinates, to be plotted

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[time, states, controls, param, algebraic_states, numerical_timeseries]
            The function to get the values of contact forces from the dynamics
        """

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.q_v_function = Function(
            "qv_function",
            [
                time_span_sym,
                nlp.states.cx,
                nlp.controls.cx,
                nlp.parameters.cx,
                nlp.algebraic_states.cx,
                nlp.numerical_timeseries.cx,
            ],
            [
                dyn_func()(
                    nlp.get_var("q_u", nlp.states.cx, nlp.controls.cx),
                    DM.zeros(nlp.model.nb_dependent_joints, 1),
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["q_v"],
        )

        all_multipliers_names = []
        for nlp_i in ocp.nlp:
            if hasattr(nlp_i.model, "has_holonomic_constraints"):  # making sure we have a HolonomicBiorbdModel
                nlp_i_multipliers_names = [nlp_i.model.name_dof[i] for i in nlp_i.model.dependent_joint_index]
                all_multipliers_names.extend(
                    [name for name in nlp_i_multipliers_names if name not in all_multipliers_names]
                )

        all_multipliers_names_in_phase = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
        axes_idx = BiMapping(
            to_first=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
            to_second=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
        )

        nlp.plot["q_v"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.q_v_function(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_multipliers_names,
        )

    @staticmethod
    def configure_qdotv(ocp, nlp, dyn_func: Callable, **extra_params):
        """
        Configure the qdot_v, i.e. the dependent joint velocities, to be plotted

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[time, states, controls, param, algebraic_states, numerical_timeseries]
            The function to get the values of contact forces from the dynamics
        """

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.q_v_function = Function(
            "qdot_v_function",
            [
                time_span_sym,
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.scaled.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [
                dyn_func()(
                    nlp.get_var("q_u", nlp.states.scaled.cx, nlp.controls.scaled.cx),
                    nlp.get_var("qdot_u", nlp.states.scaled.cx, nlp.controls.scaled.cx),
                    DM.zeros(nlp.model.nb_dependent_joints, 1),
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["qdot_v"],
        )

        all_multipliers_names = []
        for nlp_i in ocp.nlp:
            if hasattr(nlp_i.model, "has_holonomic_constraints"):  # making sure we have a HolonomicBiorbdModel
                nlp_i_multipliers_names = [nlp_i.model.name_dof[i] for i in nlp_i.model.dependent_joint_index]
                all_multipliers_names.extend(
                    [name for name in nlp_i_multipliers_names if name not in all_multipliers_names]
                )

        all_multipliers_names_in_phase = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
        axes_idx = BiMapping(
            to_first=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
            to_second=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
        )

        nlp.plot["qdot_v"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.q_v_function(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_multipliers_names,
        )

    @staticmethod
    def _get_kinematics_based_names(nlp, var_type: str) -> list[str]:
        """
        To modify the names of the variables added to the plots if there is quaternions

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        var_type: str
            A string that refers to the decision variable such as (q, qdot, qddot, tau, etc...)

        Returns
        ----------
        new_name: list[str]
            The list of str to display on figures
        """

        idx = nlp.phase_mapping.to_first.map_idx if nlp.phase_mapping else range(nlp.model.nb_q)

        if nlp.model.nb_quaternions == 0:
            new_names = [nlp.model.name_dof[i] for i in idx]
        else:
            new_names = []
            for i in nlp.phase_mapping.to_first.map_idx:
                if nlp.model.name_dof[i][-4:-1] == "Rot" or nlp.model.name_dof[i][-6:-1] == "Trans":
                    new_names += [nlp.model.name_dof[i]]
                else:
                    if nlp.model.name_dof[i][-5:] != "QuatW":
                        if var_type == "qdot":
                            new_names += [nlp.model.name_dof[i][:-5] + "omega" + nlp.model.name_dof[i][-1]]
                        elif var_type == "qddot":
                            new_names += [nlp.model.name_dof[i][:-5] + "omegadot" + nlp.model.name_dof[i][-1]]
                        elif var_type == "qdddot":
                            new_names += [nlp.model.name_dof[i][:-5] + "omegaddot" + nlp.model.name_dof[i][-1]]
                        elif var_type == "tau" or var_type == "taudot":
                            new_names += [nlp.model.name_dof[i]]

        return new_names

    @staticmethod
    def _apply_phase_mapping(ocp, nlp, name: str) -> BiMapping | None:
        """
        Apply the phase mapping to the variable

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        name: str
            The name of the variable to map

        Returns
        -------
        The mapping or None if no mapping is defined

        """
        if nlp.phase_mapping:
            if name in nlp.variable_mappings.keys():
                double_mapping_to_first = (
                    nlp.variable_mappings[name].to_first.map(nlp.phase_mapping.to_first.map_idx).T.tolist()[0]
                )
                double_mapping_to_first = [int(double_mapping_to_first[i]) for i in range(len(double_mapping_to_first))]
                double_mapping_to_second = (
                    nlp.variable_mappings[name].to_second.map(nlp.phase_mapping.to_second.map_idx).T.tolist()[0]
                )
                double_mapping_to_second = [
                    int(double_mapping_to_second[i]) for i in range(len(double_mapping_to_second))
                ]
            else:
                double_mapping_to_first = nlp.phase_mapping.to_first.map_idx
                double_mapping_to_second = nlp.phase_mapping.to_second.map_idx
            axes_idx = BiMapping(to_first=double_mapping_to_first, to_second=double_mapping_to_second)
        else:
            axes_idx = None
        return axes_idx

class States(FcnEnum):
    Q = (ConfigureVariables.configure_q, )
    Q_ROOTS = "q_roots"
    Q_JOINTS = "q_joints"
    QDOT = (ConfigureVariables.configure_qdot, )
    QDOT_ROOTS = "qdot_roots"
    QDOT_JOINTS = "qdot_joints"
    TAU = (ConfigureVariables.configure_tau, )
    MUSCLE_ACTIVATION = "muscle_activation"

class Controls(FcnEnum):
    QDDOT_JOINTS = "qddot_joints"
    TAU = (ConfigureVariables.configure_tau, )
    TAU_JOINTS = "tau_joints"
    TAUDOT = "taudot"
    MUSCLE_EXCITATION = "muscle_excitation"
    K = "k"
    C = "c"
    A = "a"
    COV = "cov"
    CHOLESKY_COV = "cholesky_cov"
    REF = "ref"
    SOFT_CONTACT_FORCES = "soft_contact_forces"  # This should be removed after PR #964

class AlgebraicStates(FcnEnum):
    RIGID_CONTACT_FORCES = "rigid_contact_forces"
    SOFT_CONTACT_FORCES = "soft_contact_forces"



class AutoConfigure:
    def __init__(
        self,
        states: list[States] | tuple[States] = (States.Q, States.QDOT),
        controls: list[Controls] | tuple[Controls] = (Controls.TAU),
        algebraic_states: list[AlgebraicStates] | tuple[Controls] = (),
        **extra_parameters: Any,
    ):
        """
        states: list[States] | tuple[States]
            The states to consider in the dynamics
        controls: list[Controls] | tuple[Controls]
            The controls to consider in the dynamics
        algebraic_states: list[AlgebraicStates] | tuple[Controls]
            The algebraic states to consider in the dynamics
        """
        self.states = states
        self.controls = controls
        self.algebraic_states = algebraic_states

    def initialize(
            self,
            ocp,
            nlp,
    ):
        for state in self.states:
            state(ocp, nlp, as_states=True, as_controls=False, as_algebraic_states=False)

        for control in self.controls:
            control(ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False)

        for algebraic_state in self.algebraic_states:
            algebraic_state(ocp, nlp, as_states=False, as_controls=False, as_algebraic_states=True)


