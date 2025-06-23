from typing import Callable, Any
import numpy as np
from casadi import DM, vertcat, Function

from .configure_new_variable import NewVariableConfiguration
from .fatigue.fatigue_dynamics import FatigueList
from ..misc.enums import PlotType, ContactType
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping, Mapping
from ..models.protocols.stochastic_biomodel import StochasticBioModel
from ..dynamics.ode_solvers import OdeSolver
from ..dynamics.dynamics_functions import DynamicsFunctions
from ..gui.plot import CustomPlot

from ..misc.parameters_types import Bool, Int, NpArrayDictOptional


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
        ConfigureVariables.configure_new_variable(
            name,
            name_q,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_q_roots(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized coordinates for the root segment

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
        name = "q_roots"
        name_q_roots = [nlp.model.name_dof[i] for i in range(nlp.model.nb_root)]
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name,
            name_q_roots,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_q_joints(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized coordinates for the segments other than the root segment

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
        name = "q_joints"
        name_q_joints = [nlp.model.name_dof[i] for i in range(nlp.model.nb_root, nlp.model.nb_q)]
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name,
            name_q_joints,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_q_u(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized coordinates for the independent dofs in the case of holonomic dynamics

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
        if as_states != True or as_controls != False or as_algebraic_states != False:
            raise RuntimeError("configure_q_u is intended to be used as a state.")

        name = "q_u"
        names_u = [nlp.model.name_dof[i] for i in nlp.model.independent_joint_index]
        ConfigureVariables.configure_new_variable(
            name,
            names_u,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_algebraic_states=False,
            # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
            # see _set_kinematic_phase_mapping method
            # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
        )

    @staticmethod
    def configure_qdot_u(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized coordinates for the independent velocities in the case of holonomic dynamics

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
        if as_states != True or as_controls != False or as_algebraic_states != False:
            raise RuntimeError("configure_q_u is intended to be used as a state.")

        name = "qdot_u"
        names_qdot = ConfigureVariables._get_kinematics_based_names(nlp, "qdot")
        names_udot = [names_qdot[i] for i in nlp.model.independent_joint_index]
        ConfigureVariables.configure_new_variable(
            name,
            names_udot,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_algebraic_states=False,
            # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
            # see _set_kinematic_phase_mapping method
            # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
        )

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
        ConfigureVariables.configure_new_variable(
            name,
            name_qdot,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_qdot_roots(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized velocities for the root segment

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
        name = "qdot_roots"
        name_qdot = ConfigureVariables._get_kinematics_based_names(nlp, "qdot")
        name_qdot_roots = [name_qdot[i] for i in range(nlp.model.nb_root)]
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name,
            name_qdot_roots,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_qdot_joints(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized velocities for the segments other than the root segment

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
        name = "qdot_joints"
        name_qdot = ConfigureVariables._get_kinematics_based_names(nlp, "qdot")
        name_qdot_joints = [name_qdot[i] for i in range(nlp.model.nb_root, nlp.model.nb_qdot)]
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name,
            name_qdot_joints,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

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
        ConfigureVariables.configure_new_variable(
            name,
            name_qddot,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_qddot_joints(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized acceleration for the segments other than the root segment

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
        name = "qddot_joints"
        name_qddot = ConfigureVariables._get_kinematics_based_names(nlp, "qddot")
        name_qddot_joints = [name_qddot[i] for i in range(nlp.model.nb_root, nlp.model.nb_q)]
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name,
            name_qddot_joints,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

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
        ConfigureVariables.configure_new_variable(
            name,
            name_qdddot,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_stochastic_k(
        ocp,
        nlp,
        as_states: bool,
        as_controls: bool,
        as_algebraic_states: bool,
        n_noised_controls: int,
        n_references: int,
    ):
        """
        Configure the optimal feedback gain matrix K.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """
        if as_states != False or as_controls != True or as_algebraic_states != False:
            raise RuntimeError("configure_stochastic_k is intended to be used as a control.")

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
    def configure_stochastic_c(
        ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool, n_noised_states: int, n_noise: int
    ):
        """
        Configure the stochastic variable matrix C representing the injection of motor noise (df/dw).
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """

        if as_states != False or as_controls != True or as_algebraic_states != False:
            raise RuntimeError("configure_stochastic_c is intended to be used as a control.")

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
    def configure_stochastic_a(
        ocp, nlp, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool, n_noised_states: Int
    ):
        """
        Configure the stochastic variable matrix A representing the propagation of motor noise (df/dx).
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """

        if as_states != False or as_controls != True or as_algebraic_states != False:
            raise RuntimeError("configure_stochastic_a is intended to be used as a control.")

        name = "a"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_a = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_a += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states**2)), list(range(n_noised_states**2)))

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
    def configure_stochastic_cov_implicit(
        ocp, nlp, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool, n_noised_states: Int
    ):
        """
        Configure the covariance matrix P representing the motor noise.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """

        if as_states != False or as_controls != True or as_algebraic_states != False:
            raise RuntimeError("configure_stochastic_cov_implicit is intended to be used as a control.")

        name = "cov"

        if name in nlp.variable_mappings:
            raise NotImplementedError(f"Algebraic states and mapping cannot be use together for now.")

        name_cov = []
        for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
            for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
                name_cov += [name_1 + "_&_" + name_2]
        nlp.variable_mappings[name] = BiMapping(list(range(n_noised_states**2)), list(range(n_noised_states**2)))
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
    def configure_stochastic_cholesky_cov(
        ocp, nlp, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool, n_noised_states: Int
    ):
        """
        Configure the diagonal matrix needed to reconstruct the covariance matrix using L @ L.T.
        This formulation allows insuring that the covariance matrix is always positive semi-definite.
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """

        if as_states != False or as_controls != True or as_algebraic_states != False:
            raise RuntimeError("configure_stochastic_cholesky_cov is intended to be used as a control.")

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
    def configure_stochastic_ref(
        ocp, nlp, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool, n_references: Int
    ):
        """
        Configure the reference kinematics.

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """

        if as_states != False or as_controls != True or as_algebraic_states != False:
            raise RuntimeError("configure_stochastic_ref is intended to be used as a control.")

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
    def configure_stochastic_m(
        ocp, nlp, as_states: Bool, as_controls: Bool, as_algebraic_states: Bool, n_noised_states: Int
    ):
        """
        Configure the helper matrix M (from Gillis 2013 : https://doi.org/10.1109/CDC.2013.6761121).

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        """

        if as_states != False or as_controls != False or as_algebraic_states != True:
            raise RuntimeError("configure_stochastic_m is intended to be used as an algebraic state.")

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
    def configure_tau(
        ocp,
        nlp,
        as_states: bool,
        as_controls: bool,
        as_algebraic_states: bool,
    ):
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
        """

        name = "tau"
        name_tau = ConfigureVariables._get_kinematics_based_names(nlp, name)
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name,
            name_tau,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_tau_joints(ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool):
        """
        Configure the generalized forces for the segments other than the root segment

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
        name = "tau_joints"
        name_tau = ConfigureVariables._get_kinematics_based_names(nlp, "tau")
        name_tau_joints = [name_tau[i] for i in range(nlp.model.nb_root, nlp.model.nb_tau)]
        axes_idx = ConfigureVariables._apply_phase_mapping(ocp, nlp, name)
        ConfigureVariables.configure_new_variable(
            name,
            name_tau_joints,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
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
            name,
            name_residual_tau,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
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
        ConfigureVariables.configure_new_variable(
            name,
            name_taudot,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_translational_forces(
        ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool, n_contacts: int = 1
    ):
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
        ConfigureVariables.configure_new_variable(
            "contact_forces",
            name_contact_forces,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
        )
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

        name_contact_forces = [name for name in nlp.model.rigid_contact_names]
        ConfigureVariables.configure_new_variable(
            "rigid_contact_forces",
            name_contact_forces,
            ocp,
            nlp,
            as_states=as_states,
            as_controls=as_controls,
            as_algebraic_states=as_algebraic_states,
        )

    @staticmethod
    def configure_lagrange_multipliers_variable(
        ocp, nlp, as_states: bool, as_controls: bool, as_algebraic_states: bool
    ) -> None:
        """
        Configure the lambdas for the holonomic constraints as algebraic states

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the lagrange multipliers should be a state
        as_controls: bool
            If the lagrange multipliers should be a control
        as_algebraic_states: bool
            If the lagrange multipliers should be an algebraic state
        """
        if nlp.model.has_holonomic_constraints:
            lambdas = []
            for i in range(nlp.model.nb_holonomic_constraints):
                lambdas.append(f"lambda_{i}")
            ConfigureVariables.configure_new_variable(
                "lambdas",
                lambdas,
                ocp,
                nlp,
                as_states=as_states,
                as_controls=as_controls,
                as_algebraic_states=as_algebraic_states,
                # Please note that lagrange multipliers (lambdas) are not ready yet for bimapping other than the
                # independent-dependent mapping from the holonomic configuration.
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
    def configure_muscles(
        ocp,
        nlp,
        as_states: bool,
        as_controls: bool,
        as_algebraic_states: bool,
    ):
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
        )

    @staticmethod
    def configure_rigid_contact_function(ocp, nlp, **extra_params) -> None:
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        contact_func: Callable[time, states, controls, param, algebraic_states, numerical_timeseries]
            The function to get the values of contact forces from the dynamics
        """

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.rigid_contact_forces_func = Function(
            "rigid_contact_forces_func",
            [
                time_span_sym,
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.scaled.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [
                nlp.model.get_rigid_contact_forces(
                    time_span_sym,
                    nlp.states.scaled.cx,
                    nlp.controls.scaled.cx,
                    nlp.parameters.scaled.cx,
                    nlp.algebraic_states.scaled.cx,
                    nlp.numerical_timeseries.cx,
                    nlp,
                    **extra_params,
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["rigid_contact_forces"],
        ).expand()

        all_contact_names = []
        for elt in ocp.nlp:
            all_contact_names.extend([name for name in elt.model.rigid_contact_names if name not in all_contact_names])

        if "rigid_contact_forces" in nlp.plot_mapping:
            contact_names_in_phase = [name for name in nlp.model.rigid_contact_names]
            axes_idx = BiMapping(
                to_first=nlp.plot_mapping["rigid_contact_forces"].map_idx,
                to_second=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
            )
        else:
            contact_names_in_phase = [name for name in nlp.model.rigid_contact_names]
            axes_idx = BiMapping(
                to_first=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
                to_second=[i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase],
            )

        nlp.plot["rigid_contact_forces"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.rigid_contact_forces_func(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_contact_names,
        )

    @staticmethod
    def configure_soft_contact_function(ocp, nlp) -> None:
        """
        Configure the soft contact sphere

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.soft_contact_forces_func = Function(
            "soft_contact_forces_func",
            [
                time_span_sym,
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.scaled.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [nlp.model.soft_contact_forces().expand()(nlp.states["q"].cx, nlp.states["qdot"].cx, nlp.parameters.cx)],
            ["t_span", "x", "u", "p", "a", "d"],
            ["soft_contact_forces"],
        ).expand()

        component_list = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]

        for i_sc in range(nlp.model.nb_soft_contacts):
            all_soft_contact_names = []
            all_soft_contact_names.extend(
                [
                    f"{nlp.model.soft_contact_names[i_sc]}_{name}"
                    for name in component_list
                    if nlp.model.soft_contact_names[i_sc] not in all_soft_contact_names
                ]
            )

            if "soft_contact_forces" in nlp.plot_mapping:
                soft_contact_names_in_phase = [
                    f"{nlp.model.soft_contact_names[i_sc]}_{name}"
                    for name in component_list
                    if nlp.model.soft_contact_names[i_sc] not in all_soft_contact_names
                ]
                phase_mappings = BiMapping(
                    to_first=nlp.plot_mapping["soft_contact_forces"].map_idx,
                    to_second=[i for i, c in enumerate(all_soft_contact_names) if c in soft_contact_names_in_phase],
                )
            else:
                soft_contact_names_in_phase = [
                    f"{nlp.model.soft_contact_names[i_sc]}_{name}"
                    for name in component_list
                    if nlp.model.soft_contact_names[i_sc] not in all_soft_contact_names
                ]
                phase_mappings = BiMapping(
                    to_first=[i for i, c in enumerate(all_soft_contact_names) if c in soft_contact_names_in_phase],
                    to_second=[i for i, c in enumerate(all_soft_contact_names) if c in soft_contact_names_in_phase],
                )
            nlp.plot[f"soft_contact_forces_{nlp.model.soft_contact_names[i_sc]}"] = CustomPlot(
                lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.soft_contact_forces_func(
                    np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
                )[(i_sc * 6) : ((i_sc + 1) * 6), :],
                plot_type=PlotType.INTEGRATED,
                axes_idx=phase_mappings,
                legend=all_soft_contact_names,
            )

    @staticmethod
    def configure_qv(ocp, nlp, **extra_params) -> None:
        """
        Configure the qv, i.e. the dependent joint coordinates, to be plotted

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
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
                nlp.model.compute_q_v()(
                    nlp.states["q_u"].cx,
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
    def configure_qdotv(ocp, nlp, **extra_params) -> None:
        """
        Configure the qdot_v, i.e. the dependent joint velocities, to be plotted

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
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
                nlp.model._compute_qdot_v()(
                    nlp.states.scaled["q_u"].cx,
                    nlp.states.scaled["qdot_u"].cx,
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
    def configure_lagrange_multipliers_function(ocp, nlp: NpArrayDictOptional, **extra_params) -> None:
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        time_span_sym = vertcat(nlp.time_cx, nlp.dt)
        nlp.lagrange_multipliers_function = Function(
            "lagrange_multipliers_function",
            [
                time_span_sym,
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.scaled.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [
                nlp.model.compute_the_lagrangian_multipliers()(
                    nlp.states.scaled["q_u"].cx,
                    nlp.states.scaled["qdot_u"].cx,
                    DM.zeros(nlp.model.nb_dependent_joints, 1),
                    DynamicsFunctions.get(nlp.controls["tau"], nlp.controls.scaled.cx),
                )
            ],
            ["t_span", "x", "u", "p", "a", "d"],
            ["lagrange_multipliers"],
        )

        all_multipliers_names = []
        for nlp_i in ocp.nlp:
            if hasattr(nlp_i.model, "has_holonomic_constraints"):  # making sure we have a HolonomicBiorbdModel
                nlp_i_multipliers_names = [nlp_i.model.name_dof[i] for i in nlp_i.model.dependent_joint_index]
                all_multipliers_names.extend(
                    [name for name in nlp_i_multipliers_names if name not in all_multipliers_names]
                )

        all_multipliers_names = [f"lagrange_multiplier_{name}" for name in all_multipliers_names]
        all_multipliers_names_in_phase = [
            f"lagrange_multiplier_{nlp.model.name_dof[i]}" for i in nlp.model.dependent_joint_index
        ]

        axes_idx = BiMapping(
            to_first=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
            to_second=[i for i, c in enumerate(all_multipliers_names) if c in all_multipliers_names_in_phase],
        )

        nlp.plot["lagrange_multipliers"] = CustomPlot(
            lambda t0, phases_dt, node_idx, x, u, p, a, d: nlp.lagrange_multipliers_function(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), x, u, p, a, d
            ),
            plot_type=PlotType.INTEGRATED,
            axes_idx=axes_idx,
            legend=all_multipliers_names,
        )

    @staticmethod
    def configure_variational_functions(ocp, nlp, **extra_params) -> None:
        """
        Configure the functions necessary for the variational integrator.
        # TODO: This should be done elsewhere, but for now it is the easiest way to bypass the dynamics declaration as the variational dynamics depends on the nodes i-1, i and i+1

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        dt = nlp.cx.sym("time_step")
        q_prev = nlp.cx.sym("q_prev", nlp.model.nb_q, 1)
        q_cur = nlp.cx.sym("q_cur", nlp.model.nb_q, 1)
        q_next = nlp.cx.sym("q_next", nlp.model.nb_q, 1)
        control_prev = nlp.cx.sym("control_prev", nlp.model.nb_q, 1)
        control_cur = nlp.cx.sym("control_cur", nlp.model.nb_q, 1)
        control_next = nlp.cx.sym("control_next", nlp.model.nb_q, 1)
        q0 = nlp.cx.sym("q0", nlp.model.nb_q, 1)
        qdot0 = nlp.cx.sym("qdot_start", nlp.model.nb_q, 1)
        q1 = nlp.cx.sym("q1", nlp.model.nb_q, 1)
        control0 = nlp.cx.sym("control0", nlp.model.nb_q, 1)
        control1 = nlp.cx.sym("control1", nlp.model.nb_q, 1)
        q_ultimate = nlp.cx.sym("q_ultimate", nlp.model.nb_q, 1)
        qdot_ultimate = nlp.cx.sym("qdot_ultimate", nlp.model.nb_q, 1)
        q_penultimate = nlp.cx.sym("q_penultimate", nlp.model.nb_q, 1)
        controlN_minus_1 = nlp.cx.sym("controlN_minus_1", nlp.model.nb_q, 1)
        controlN = nlp.cx.sym("controlN", nlp.model.nb_q, 1)

        three_nodes_input = [dt, q_prev, q_cur, q_next, control_prev, control_cur, control_next]
        two_first_nodes_input = [dt, q0, qdot0, q1, control0, control1]
        two_last_nodes_input = [dt, q_penultimate, q_ultimate, qdot_ultimate, controlN_minus_1, controlN]

        if nlp.model.has_holonomic_constraints:
            lambdas = nlp.cx.sym("lambda", nlp.model.nb_holonomic_constraints, 1)
            three_nodes_input.append(lambdas)
            two_first_nodes_input.append(lambdas)
            two_last_nodes_input.append(lambdas)
        else:
            lambdas = None

        nlp.dynamics_defects_func = Function(
            "ThreeNodesIntegration",
            three_nodes_input,
            [
                nlp.model.discrete_euler_lagrange_equations(
                    dt,
                    q_prev,
                    q_cur,
                    q_next,
                    control_prev,
                    control_cur,
                    control_next,
                    lambdas,
                )
            ],
        )

        nlp.dynamics_defects_func_first_node = Function(
            "TwoFirstNodesIntegration",
            two_first_nodes_input,
            [
                nlp.model.compute_initial_states(
                    dt,
                    q0,
                    qdot0,
                    q1,
                    control0,
                    control1,
                    lambdas,
                )
            ],
        )

        nlp.dynamics_defects_func_last_node = Function(
            "TwoLastNodesIntegration",
            two_last_nodes_input,
            [
                nlp.model.compute_final_states(
                    dt,
                    q_penultimate,
                    q_ultimate,
                    qdot_ultimate,
                    controlN_minus_1,
                    controlN,
                    lambdas,
                )
            ],
        )

        # if expand:
        # TODO: see how to restore the possiblity to not expand the functions
        nlp.dynamics_defects_func = nlp.dynamics_defects_func.expand()
        nlp.dynamics_defects_func_first_node = nlp.dynamics_defects_func_first_node.expand()
        nlp.dynamics_defects_func_last_node = nlp.dynamics_defects_func_last_node.expand()

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
    Q = (ConfigureVariables.configure_q,)
    Q_ROOTS = (ConfigureVariables.configure_q_roots,)
    Q_JOINTS = (ConfigureVariables.configure_q_joints,)
    QDOT = (ConfigureVariables.configure_qdot,)
    QDOT_ROOTS = (ConfigureVariables.configure_qdot_roots,)
    QDOT_JOINTS = (ConfigureVariables.configure_qdot_joints,)
    Q_U = (ConfigureVariables.configure_q_u,)
    QDOT_U = (ConfigureVariables.configure_qdot_u,)
    TAU = (ConfigureVariables.configure_tau,)
    MUSCLE_ACTIVATION = (ConfigureVariables.configure_muscles,)


class Controls(FcnEnum):
    QDDOT_JOINTS = (ConfigureVariables.configure_qddot_joints,)
    TAU = (ConfigureVariables.configure_tau,)
    RESIDUAL_TAU = (ConfigureVariables.configure_residual_tau,)
    TAU_JOINTS = (ConfigureVariables.configure_tau_joints,)
    TAUDOT = (ConfigureVariables.configure_taudot,)
    MUSCLE_EXCITATION = (ConfigureVariables.configure_muscles,)
    K = (ConfigureVariables.configure_stochastic_k,)
    C = (ConfigureVariables.configure_stochastic_c,)
    A = (ConfigureVariables.configure_stochastic_a,)
    COV = (ConfigureVariables.configure_stochastic_cov_implicit,)
    CHOLESKY_COV = (ConfigureVariables.configure_stochastic_cholesky_cov,)
    REF = (ConfigureVariables.configure_stochastic_ref,)
    LAMBDA = (ConfigureVariables.configure_lagrange_multipliers_variable,)


class AlgebraicStates(FcnEnum):
    M = (ConfigureVariables.configure_stochastic_m,)
    RIGID_CONTACT_FORCES = (ConfigureVariables.configure_rigid_contact_forces,)
    SOFT_CONTACT_FORCES = (ConfigureVariables.configure_soft_contact_forces,)


class AutoConfigure:
    def __init__(
        self,
        states: list[States],
        controls: list[Controls] = None,
        algebraic_states: list[AlgebraicStates] = None,
        functions: list[Callable] = None,
        **extra_parameters: Any,
    ):
        """
        states: list[States] | tuple[States]
            The states to consider in the dynamics
        controls: list[Controls] | tuple[Controls]
            The controls to consider in the dynamics
        algebraic_states: list[AlgebraicStates] | tuple[Controls]
            The algebraic states to consider in the dynamics
        functions: list[Callable] | tuple[Callable]
            Additional functions to add to the nlp (mainly for live plots)
        """
        self.states = states
        self.controls = controls
        self.algebraic_states = algebraic_states
        self.functions = functions

    def configure_contacts(self, ocp, nlp):

        # Add algebraic states for implicit contacts
        if ContactType.RIGID_IMPLICIT in nlp.model.contact_types:
            if self.algebraic_states is None:
                self.algebraic_states = [AlgebraicStates.RIGID_CONTACT_FORCES]
            else:
                self.algebraic_states += [AlgebraicStates.RIGID_CONTACT_FORCES]

        if ContactType.SOFT_IMPLICIT in nlp.model.contact_types:
            if self.algebraic_states is None:
                self.algebraic_states = [AlgebraicStates.SOFT_CONTACT_FORCES]
            else:
                self.algebraic_states += [AlgebraicStates.SOFT_CONTACT_FORCES]

        # Define the contact function for explicit contacts
        if ContactType.RIGID_EXPLICIT in nlp.model.contact_types:
            ConfigureVariables.configure_rigid_contact_function(ocp, nlp)

        if ContactType.SOFT_EXPLICIT in nlp.model.contact_types:
            ConfigureVariables.configure_soft_contact_function(ocp, nlp)

    def initialize(
        self,
        ocp,
        nlp,
    ):

        for state in self.states:
            state(ocp, nlp, as_states=True, as_controls=False, as_algebraic_states=False)

        for control in self.controls:
            control(ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False)

        # Contacts must be defined after states and controls, but before algebraic states
        self.configure_contacts(ocp, nlp)

        for algebraic_state in self.algebraic_states:
            algebraic_state(ocp, nlp, as_states=False, as_controls=False, as_algebraic_states=True)

        for function in self.functions:
            function(ocp, nlp)
